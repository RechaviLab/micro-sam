import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
import imageio.v3 as imageio
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: safe to use from the GUI's worker thread
import matplotlib.pyplot as plt


# load worm mask from .tiff file
# binary_mask_path = r"C:\data\RechaviLab\R1\microsam_output\R1_P0_-001_mask.tif"
# binary_mask = imageio.imread(binary_mask_path).astype(bool)
# plt.imshow(binary_mask)
mm_in_pixels = 320
# if 105 pixels is 1 mm, then 1 pixel is 1/105 mm, which is 1000/105 microns

# micrometer_per_pixel = 1000/mm_in_pixels  # 1 pixel = 1000/105 microns
# 688*micrometer_per_pixel
# scale_um_per_px = micrometer_per_pixel
# analyze_worm_geometry(binary_mask, sample_interval=10.0,scale_um_per_px=scale_um_per_px)
#!/usr/bin/env python3
"""
wormsizer_py.py - Python re-implementation of the WormSizer measurement stage
(Moore, Jordan & Baugh 2013, PLoS ONE 8(2): e57142) starting from an EXISTING
binary / labelled worm mask (i.e. the pre-processing + thresholding stages of the
original ImageJ plugin are skipped).

Pipeline (per worm)
  1. split mask into individual worms (connected components, or label image)
  2. area filter, optional closing, hole filling
  3. skeletonize -> keep longest geodesic path (= "pruned skeleton")
  4. smooth the path, extend both ends along their tangent to the worm edge
  5. length  = arc-length of the extended path
  6. sample the path every `interval_px`; at each sample cast rays along the
     local normal (both sides) until they leave the mask -> radii / width
  7. volume  = sum of conical frustums  pi*l/3 * (r1^2 + r1*r2 + r2^2)
     surface = sum of frustum lateral areas (WormSizer >= 1.2.0 feature)
  8. QC: radial-model reconstruction vs. mask (IoU), EDT cross-check of the
     widths, volume-vs-interval sensitivity (cf. Fig. 3 of the paper).

Usage (batch: a folder of .tif/.tiff masks, one worm per file)
  python wormsizer_py.py masks/ --um-per-px 2 --interval 10 --out results/
  python wormsizer_py.py masks/ --out results/ --debug-n 3      # + step-by-step figures, first 3 masks
  python wormsizer_py.py one_mask.tif --show

Or from Python
  from wormsizer_py import Params, analyze_mask, plot_worm, plot_debug, run_batch
  df, failed = run_batch(r'C:\\masks', Params(um_per_px=2, min_area_px=0), out_dir='results')
"""
# from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage import draw, io, measure, morphology


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    um_per_px: float = 2.0          # image scale
    interval_px: float = 10.0       # spacing of width samples along skeleton (paper: 10 px)
    min_area_px: float = 300.0      # worm area filter, in pixels (paper GUI default 300 .. inf)
    max_area_px: float = np.inf
    close_radius: int = 0           # morphological closing radius (paper GUI: 5); 0 = off
    min_skeleton_ratio: float = 0.0  # pruned/original skeleton length filter ("tubeness")
    smooth_sigma_px: float = 3.0    # gaussian smoothing of the skeleton path (0 = off)
    tangent_halfwin_px: float = 3.0  # half-window used to estimate local tangent / normal
    extend_tangent_px: float = 5.0  # path length used to estimate the end tangent
    max_extension_px: float = 40.0
    ray_step_px: float = 0.25       # step for ray marching / extension
    pad_px: int = 12                # zero padding around each worm crop


# --------------------------------------------------------------------------- #
# 1. Splitting the mask into worms
# --------------------------------------------------------------------------- #
def split_worms(mask_img, params: Params, mode: str = "auto"):
    """Yield dicts with a padded single-worm boolean crop and bookkeeping."""
    a = np.asarray(mask_img)
    if a.ndim == 3:
        a = a[..., 0]
    vals = np.unique(a[a != 0])
    if mode == "auto":
        mode = "labels" if len(vals) > 1 else "binary"
    if mode == "single":                  # one mask = one worm; stray fragments are
        lab = (a > 0).astype(np.int64)    # dropped later by clean_mask (largest component)
    elif mode == "binary":
        lab = measure.label(a > 0, connectivity=2)
    else:
        lab = a.astype(np.int64)

    H, W = lab.shape
    worms, skipped = [], []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        r0, c0 = sl[0].start, sl[1].start
        touches = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W
        if not (params.min_area_px <= area <= params.max_area_px):
            skipped.append((i, f"area {area}px outside filter"))
            continue
        P = params.pad_px
        m = np.pad(m, P)
        worms.append(dict(label=i, mask=m, offset=(r0 - P, c0 - P), area_px=area,
                          touches_border=bool(touches)))
    return worms, skipped


def clean_mask(m: np.ndarray, params: Params):
    had_holes = bool((ndi.binary_fill_holes(m) & ~m).any())
    if params.close_radius > 0:
        m = morphology.binary_closing(m, morphology.disk(params.close_radius))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n > 1:                                   # keep largest component
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m, had_holes


# --------------------------------------------------------------------------- #
# 2. Skeleton -> longest path (pruned skeleton)
# --------------------------------------------------------------------------- #
def skeleton_longest_path(mask: np.ndarray):
    """Returns (path_xy [N,2] float, all_skeleton_xy [M,2], ratio pruned/original)."""
    skel = morphology.skeletonize(mask)
    coords = np.argwhere(skel)                  # (row, col)
    n = len(coords)
    if n < 3:
        return None, coords[:, ::-1].astype(float), 0.0
    idx = -np.ones(mask.shape, dtype=np.int64)
    idx[coords[:, 0], coords[:, 1]] = np.arange(n)
    H, W = mask.shape
    rows, cols, wts = [], [], []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        r2, c2 = coords[:, 0] + dr, coords[:, 1] + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        j = np.full(n, -1)
        j[ok] = idx[r2[ok], c2[ok]]
        sel = j >= 0
        rows.append(np.arange(n)[sel]); cols.append(j[sel])
        wts.append(np.full(sel.sum(), np.hypot(dr, dc)))
    G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()

    ncomp, comp = connected_components(G, directed=False)
    big = np.argmax(np.bincount(comp))
    nodes = np.where(comp == big)[0]

    d0 = dijkstra(G, directed=False, indices=nodes[0])
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))                      # one tip
    da, pred = dijkstra(G, directed=False, indices=a, return_predecessors=True)
    da[~np.isfinite(da)] = -1
    b = int(np.argmax(da))                      # opposite tip (double-sweep = tree diameter)
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    path = coords[path][:, ::-1].astype(float)  # -> (x, y)
    return path, coords[:, ::-1].astype(float), len(path) / n


def smooth_path(p: np.ndarray, sigma: float):
    if sigma <= 0 or len(p) < 6:
        return p
    pad = int(3 * sigma) + 1
    pad = min(pad, len(p) - 1)
    # 'odd' reflection = point reflection, so the end points do not shrink inwards
    q = np.pad(p, ((pad, pad), (0, 0)), mode="reflect", reflect_type="odd")
    q = ndi.gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    return q[pad:-pad]


def arclength(p: np.ndarray):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def _extend_one_end(p, mask_f, params: Params):
    """Extend the LAST end of p along its tangent until the mask edge."""
    s = arclength(p)
    i = int(np.searchsorted(s, s[-1] - params.extend_tangent_px))
    i = min(i, len(p) - 2)
    t = p[-1] - p[i]
    nt = np.hypot(*t)
    if nt == 0:
        return p, np.empty((0, 2))
    t /= nt
    step = params.ray_step_px
    pos, extra, v_prev = p[-1].copy(), [], 1.0
    for _ in range(int(params.max_extension_px / step)):
        nxt = pos + t * step
        v = ndi.map_coordinates(mask_f, [[nxt[1]], [nxt[0]]], order=1, cval=0.0)[0]
        if v < 0.5:
            frac = (v_prev - 0.5) / max(v_prev - v, 1e-9)    # sub-pixel crossing
            extra.append(pos + t * step * frac)
            break
        pos, v_prev = nxt, v
        extra.append(pos.copy())
    extra = np.array(extra) if extra else np.empty((0, 2))
    return (np.vstack([p, extra]) if len(extra) else p), extra


def extend_path(p, mask_f, params: Params):
    p1, ext_end = _extend_one_end(p, mask_f, params)
    p2, ext_start = _extend_one_end(p1[::-1], mask_f, params)
    return p2[::-1], ext_start, ext_end


# --------------------------------------------------------------------------- #
# 3. Width sampling, volume, surface area
# --------------------------------------------------------------------------- #
def _interp_path(p, s, sq):
    return np.stack([np.interp(sq, s, p[:, 0]), np.interp(sq, s, p[:, 1])], axis=-1)


def _ray_length(mask_f, pt, d, step, max_len=200.0):
    ds = np.arange(0.0, max_len + step, step)
    xs, ys = pt[0] + d[0] * ds, pt[1] + d[1] * ds
    vals = ndi.map_coordinates(mask_f, [ys, xs], order=1, cval=0.0)
    out = np.where(vals < 0.5)[0]
    if len(out) == 0:
        return max_len
    k = out[0]
    if k == 0:
        return 0.0
    frac = (vals[k - 1] - 0.5) / max(vals[k - 1] - vals[k], 1e-9)
    return ds[k - 1] + frac * step


def sample_positions(L, interval):
    s = np.arange(0.0, L, interval)
    if L - s[-1] > 1e-6:
        s = np.r_[s, L]
    if len(s) < 2:
        s = np.array([0.0, L])
    return s


def measure_path(path, mask_f, edt, params: Params, interval_px=None):
    """All WormSizer quantities for a given centreline `path` (x,y in px)."""
    interval = params.interval_px if interval_px is None else interval_px
    s = arclength(path)
    L = s[-1]
    sq = sample_positions(L, interval)
    pts = _interp_path(path, s, sq)

    h = params.tangent_halfwin_px
    t = _interp_path(path, s, np.clip(sq + h, 0, L)) - _interp_path(path, s, np.clip(sq - h, 0, L))
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)

    rl = np.array([_ray_length(mask_f, pts[i], nrm[i], params.ray_step_px) for i in range(len(sq))])
    rr = np.array([_ray_length(mask_f, pts[i], -nrm[i], params.ray_step_px) for i in range(len(sq))])
    width = rl + rr                              # px (diameter)
    r_um = 0.5 * width * params.um_per_px        # radius used for the frustums
    l_um = np.diff(sq) * params.um_per_px

    r1, r2 = r_um[:-1], r_um[1:]
    vol_um3 = np.sum(np.pi * l_um / 3.0 * (r1**2 + r1 * r2 + r2**2))
    area_um2 = np.sum(np.pi * (r1 + r2) * np.sqrt(l_um**2 + (r1 - r2) ** 2))

    # width at the middle of the skeleton (for the cylinder approximation)
    pm = _interp_path(path, s, np.array([L / 2]))[0]
    tm = _interp_path(path, s, np.array([min(L / 2 + h, L)]))[0] - \
        _interp_path(path, s, np.array([max(L / 2 - h, 0)]))[0]
    tm /= max(np.hypot(*tm), 1e-9)
    nm = np.array([-tm[1], tm[0]])
    w_mid = (_ray_length(mask_f, pm, nm, params.ray_step_px) +
             _ray_length(mask_f, pm, -nm, params.ray_step_px))

    L_um = L * params.um_per_px
    return dict(
        length_um=L_um,
        mean_width_um=float(width.mean() * params.um_per_px),
        mid_width_um=w_mid * params.um_per_px,
        volume_pl=vol_um3 / 1e3,                          # 1 pl = 1000 um^3
        cylinder_volume_pl=np.pi * (0.5 * w_mid * params.um_per_px) ** 2 * L_um / 1e3,
        surface_area_um2=area_um2,
        # arrays for plotting
        _s=sq, _pts=pts, _nrm=nrm, _rl=rl, _rr=rr, _width_px=width,
        _edt_width_px=2.0 * (ndi.map_coordinates(edt, [pts[:, 1], pts[:, 0]], order=1) - 0.5),
    )


def render_model(shape, path, meas):
    """Rasterise the radial (frustum) model back onto the image grid."""
    s = arclength(path)
    dense = np.arange(0, s[-1] + 0.5, 0.5)
    r = np.interp(dense, meas["_s"], 0.5 * meas["_width_px"])
    P = _interp_path(path, s, dense)
    out = np.zeros(shape, bool)
    for (x, y), rad in zip(P, r):
        if rad > 0.3:
            rr, cc = draw.disk((y, x), rad + 1e-3, shape=shape)
            out[rr, cc] = True
    return out


def volume_vs_interval(path, mask_f, edt, params, intervals):
    return np.array([measure_path(path, mask_f, edt, params, iv)["volume_pl"] for iv in intervals])


# --------------------------------------------------------------------------- #
# 4. Top-level: analyse a mask image
# --------------------------------------------------------------------------- #
def analyze_mask(mask_img, params: Params = Params(), name="mask", mode="single"):
    """Returns (list of result dicts, list of skipped (label, reason))."""
    worms, skipped = split_worms(mask_img, params, mode)
    results = []
    for w in worms:
        m, had_holes = clean_mask(w["mask"], params)
        raw, skel_xy, ratio = skeleton_longest_path(m)
        if raw is None:
            skipped.append((w["label"], "skeleton too short")); continue
        if ratio < params.min_skeleton_ratio:
            skipped.append((w["label"], f"skeleton ratio {ratio:.2f} below filter")); continue

        mask_f = m.astype(float)
        edt = ndi.distance_transform_edt(m)
        sm = smooth_path(raw, params.smooth_sigma_px)
        path, ext_start, ext_end = extend_path(sm, mask_f, params)

        meas = measure_path(path, mask_f, edt, params)
        model = render_model(m.shape, path, meas)
        iou = float((model & m).sum() / max((model | m).sum(), 1))

        r0, c0 = w["offset"]
        cy, cx = ndi.center_of_mass(m)
        row = dict(
            image=name, worm_id=w["label"],
            centroid_x_px=cx + c0, centroid_y_px=cy + r0,
            length_um=meas["length_um"], mid_width_um=meas["mid_width_um"],
            mean_width_um=meas["mean_width_um"], volume_pl=meas["volume_pl"],
            cylinder_volume_pl=meas["cylinder_volume_pl"],
            surface_area_um2=meas["surface_area_um2"],
            mask_area_um2=float(m.sum() * params.um_per_px**2),
            skeleton_ratio=ratio, model_iou=iou,
            width_ray_vs_edt=float(np.nanmedian(meas["_width_px"][2:-2] / np.maximum(meas["_edt_width_px"][2:-2], 1e-6)))
            if len(meas["_width_px"]) > 5 else np.nan,
            touches_border=w["touches_border"], had_holes=had_holes,
        )
        results.append(dict(row=row, mask=m, skel_xy=skel_xy, path_raw=raw, path_smooth=sm,
                            path=path, ext_start=ext_start, ext_end=ext_end,
                            meas=meas, model=model, mask_f=mask_f, edt=edt, offset=w["offset"]))
    return results, skipped


# --------------------------------------------------------------------------- #
# 5. Plots
# --------------------------------------------------------------------------- #
def plot_worm(res, params: Params, out_png=None, show=False):
    import matplotlib.pyplot as plt
    m, meas, path, row = res["mask"], res["meas"], res["path"], res["row"]
    ys, xs = np.where(m)
    pad = 8
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.max() + pad, ys.min() - pad)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = axs.ravel()
    title = f"{row['image']}  worm {row['worm_id']}"
    fig.suptitle(title, fontsize=13)

    # (1) skeleton steps
    ax[0].imshow(m, cmap="gray", vmin=0, vmax=1.6)
    ax[0].plot(*res["skel_xy"].T, ".", c="red", ms=3, label="full skeleton")
    ax[0].plot(*res["path_raw"].T, "-", c="cyan", lw=1, label="longest path (pruned)")
    ax[0].plot(*res["path_smooth"].T, "-", c="lime", lw=1.5, label="smoothed")
    for e in (res["ext_start"], res["ext_end"]):
        if len(e):
            ax[0].plot(*e.T, "-", c="yellow", lw=3)
    ax[0].plot([], [], c="yellow", lw=3, label="tangent extension")
    ax[0].plot(*meas["_pts"].T, "o", mfc="none", mec="magenta", ms=5, label="width samples")
    ax[0].set_title("1 · skeleton → pruned → smoothed → extended")
    ax[0].legend(fontsize=7, loc="upper right")

    # (2) contour + radii
    ax[1].imshow(m, cmap="gray", vmin=0, vmax=2.5)
    for c in measure.find_contours(res["mask_f"], 0.5):
        ax[1].plot(c[:, 1], c[:, 0], "-", c="gold", lw=1.2)
    ax[1].plot(*path.T, "-", c="royalblue", lw=1.5)
    P, N = meas["_pts"], meas["_nrm"]
    a, b = P + N * meas["_rl"][:, None], P - N * meas["_rr"][:, None]
    for i in range(len(P)):
        ax[1].plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], "-", c="cyan", lw=1)
    ax[1].set_title("2 · contour (yellow), centreline (blue), sampled radii (cyan)")

    # (3) width profile
    s_um = meas["_s"] * params.um_per_px
    ax[2].plot(s_um, meas["_width_px"] * params.um_per_px, "o-", c="teal", label="ray width (used)")
    ax[2].plot(s_um, meas["_edt_width_px"] * params.um_per_px, "s--", c="orange", ms=3,
               label="2×EDT width (cross-check)")
    ax[2].axhline(meas["mid_width_um"], c="grey", ls=":", label=f"mid width {meas['mid_width_um']:.1f} µm")
    ax[2].set_xlabel("position along worm (µm)"); ax[2].set_ylabel("width (µm)")
    ax[2].set_title("3 · width profile (tapering)"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    # (4) model reconstruction vs mask
    both, only_m, only_model = m & res["model"], m & ~res["model"], res["model"] & ~m
    rgb = np.zeros(m.shape + (3,))
    rgb[both] = (0.85, 0.85, 0.85); rgb[only_m] = (0.95, 0.25, 0.25); rgb[only_model] = (0.25, 0.45, 0.95)
    ax[3].imshow(rgb)
    ax[3].set_title(f"4 · radial model vs mask  (IoU = {row['model_iou']:.3f})\n"
                    "grey=both  red=mask only  blue=model only", fontsize=10)

    # (5) volume vs interval
    L_px = meas["length_um"] / params.um_per_px
    ivs = np.unique(np.round(np.linspace(2, max(0.12 * L_px, 8), 12)))  # paper: up to ~10% of length
    vols = volume_vs_interval(path, res["mask_f"], res["edt"], params, ivs)
    cv = vols.std(ddof=1) / vols.mean() * 100
    ax[4].plot(ivs, vols, "o-", c="purple")
    ax[4].axvline(params.interval_px, c="grey", ls="--", label=f"chosen = {params.interval_px:g} px")
    ax[4].set_xlabel("sampling interval (px)"); ax[4].set_ylabel("volume (pl)")
    ax[4].set_title(f"5 · volume vs interval size (CV = {cv:.1f}%)")
    ax[4].legend(fontsize=8); ax[4].grid(alpha=.3)

    # (6) numbers
    ax[5].axis("off")
    txt = (f"length            {row['length_um']:9.1f} µm\n"
           f"mid width         {row['mid_width_um']:9.1f} µm\n"
           f"mean width        {row['mean_width_um']:9.1f} µm\n"
           f"VOLUME (frustums) {row['volume_pl']:9.1f} pl\n"
           f"cylinder volume   {row['cylinder_volume_pl']:9.1f} pl  "
           f"(×{row['cylinder_volume_pl']/row['volume_pl']:.2f})\n"
           f"surface area      {row['surface_area_um2']:9.0f} µm²\n\n"
           f"-- QC --\n"
           f"model IoU         {row['model_iou']:9.3f}\n"
           f"skeleton ratio    {row['skeleton_ratio']:9.2f}\n"
           f"ray/EDT width     {row['width_ray_vs_edt']:9.2f}\n"
           f"interval CV       {cv:9.1f} %\n"
           f"touches border    {row['touches_border']}\n"
           f"had holes         {row['had_holes']}")
    ax[5].text(0.02, 0.98, txt, va="top", family="monospace", fontsize=11)

    for k in (0, 1, 3):
        ax[k].set_xlim(*xlim); ax[k].set_ylim(*ylim); ax[k].set_aspect("equal")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if out_png:
        fig.savefig(out_png, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def plot_summary(df: pd.DataFrame, out_png=None, show=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    ax[0].hist(df.length_um, bins=30, color="steelblue"); ax[0].set_xlabel("length (µm)")
    ax[1].hist(df.volume_pl, bins=30, color="seagreen"); ax[1].set_xlabel("volume (pl)")
    ax[2].scatter(df.length_um, df.mid_width_um, s=8, alpha=.6)
    ax[2].set_xlabel("length (µm)"); ax[2].set_ylabel("mid width (µm)"); ax[2].set_title("shape (cf. Fig. 7)")
    ax[3].scatter(df.cylinder_volume_pl, df.volume_pl, s=8, alpha=.6)
    lim = [0, max(df.cylinder_volume_pl.max(), df.volume_pl.max()) * 1.05]
    ax[3].plot(lim, lim, "k--", lw=.8)
    ax[3].set_xlabel("cylinder-approx volume (pl)"); ax[3].set_ylabel("WormSizer volume (pl)")
    ax[3].set_title(f"mean cylinder/WormSizer = {(df.cylinder_volume_pl/df.volume_pl).mean():.2f}")
    ax[4].hist(df.model_iou, bins=30, color="darkorange"); ax[4].set_xlabel("radial-model IoU with mask")
    ax[5].hist(df.skeleton_ratio, bins=30, color="grey"); ax[5].set_xlabel("pruned / original skeleton")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5b. Debug mode: one figure per calculation stage
# --------------------------------------------------------------------------- #
def _end_tangent(p, params: Params):
    s = arclength(p)
    i = min(int(np.searchsorted(s, s[-1] - params.extend_tangent_px)), len(p) - 2)
    t = p[-1] - p[i]
    return t / max(np.hypot(*t), 1e-9)


def _angle_deg(p, stride=5):
    d = p[stride:] - p[:-stride]
    s = arclength(p)
    return (s[:-stride] + s[stride:]) / 2, np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))


def straighten_worm(res, params: Params, half_width_px: float):
    """Resample the mask along the centreline normals -> 'straightened worm' image."""
    path, mask_f = res["path"], res["mask_f"]
    s = arclength(path)
    dense = np.arange(0.0, s[-1], 1.0)
    h = params.tangent_halfwin_px
    L = s[-1]
    P = _interp_path(path, s, dense)
    t = _interp_path(path, s, np.clip(dense + h, 0, L)) - _interp_path(path, s, np.clip(dense - h, 0, L))
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    offs = np.arange(-half_width_px, half_width_px + 0.5, 0.5)
    X = P[None, :, 0] + n[None, :, 0] * offs[:, None]
    Y = P[None, :, 1] + n[None, :, 1] * offs[:, None]
    return ndi.map_coordinates(mask_f, [Y, X], order=1, cval=0.0), dense, offs


def plot_debug(res, params: Params, out_dir=None, show=False):
    """
    Five figures that expose every intermediate step of the measurement:
      01_mask_skeleton  input mask, EDT ridge, skeleton branches that were pruned
      02_centerline     tip extension zooms, smoothing displacement, tangent angle, orientation
      03_rays           rays coloured by width, ray intensity profiles, left/right radii
      04_volume         straightened worm + frustum outline, per-segment & cumulative volume
      05_sensitivity    volume/length vs interval, smoothing sigma and tangent window
    Saved as PNGs in `out_dir` (if given) and/or shown on screen.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    um = params.um_per_px
    m, mask_f, edt, meas, path = res["mask"], res["mask_f"], res["edt"], res["meas"], res["path"]
    raw, smo = res["path_raw"], res["path_smooth"]
    ys, xs = np.where(m)
    pad = 8
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.max() + pad, ys.min() - pad)
    contours = measure.find_contours(mask_f, 0.5)
    name = f"{res['row']['image']} worm {res['row']['worm_id']}"
    figs = {}

    def img_ax(ax, zoom=True):
        ax.set_facecolor("black")
        ax.imshow(m, cmap="gray", vmin=0, vmax=2.2)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], "-", c="gold", lw=1)
        if zoom:
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")

    # ---------------- 01: mask + skeleton ---------------- #
    fig, ax = plt.subplots(3, 1, figsize=(13, 10))
    img_ax(ax[0]); ax[0].set_title("input mask used for the calculation (after hole fill / largest component)")
    e = np.ma.masked_where(~m, edt)
    im = ax[1].imshow(e, cmap="magma")
    ax[1].plot(*res["skel_xy"].T, ".", c="cyan", ms=2)
    ax[1].set_xlim(*xlim); ax[1].set_ylim(*ylim); ax[1].set_aspect("equal")
    fig.colorbar(im, ax=ax[1], fraction=0.025, label="distance to edge (px)")
    ax[1].set_title("distance transform + skeleton  (skeleton should follow the bright ridge)")
    img_ax(ax[2])
    on_path = {(int(round(x)), int(round(y))) for x, y in raw}
    br = np.array([p for p in res["skel_xy"] if (int(round(p[0])), int(round(p[1]))) not in on_path])
    if len(br):
        ax[2].plot(*br.T, ".", c="orange", ms=4, label=f"pruned branch pixels ({len(br)})")
    ax[2].plot(*raw.T, "-", c="red", lw=1.5, label=f"kept longest path ({len(raw)} px)")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].set_title(f"skeleton pruning  (pruned/original ratio = {res['row']['skeleton_ratio']:.2f})")
    fig.suptitle(name); fig.tight_layout(); figs["01_mask_skeleton"] = fig

    # ---------------- 02: centreline ---------------- #
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    tips = [("START (path[0])", smo[::-1], res["ext_start"], path[0]),
            ("END (path[-1])", smo, res["ext_end"], path[-1])]
    ext_len = []
    for k, (lab, ps, ext, tip) in enumerate(tips):
        a = ax[0, k]
        img_ax(a, zoom=False)
        a.plot(*raw.T, "-", c="cyan", lw=1, label="raw skeleton path")
        a.plot(*smo.T, "-", c="lime", lw=1.5, label="smoothed")
        if len(ext):
            a.plot(*np.vstack([ps[-1], ext]).T, "o-", c="yellow", ms=3, lw=2, label="extension")
        t = _end_tangent(ps, params)
        a.annotate("", xy=ps[-1] + t * 10, xytext=ps[-1],
                   arrowprops=dict(arrowstyle="->", color="red", lw=2))
        half = 22
        a.set_xlim(tip[0] - half, tip[0] + half); a.set_ylim(tip[1] + half, tip[1] - half)
        el = float(np.sum(np.hypot(*np.diff(np.vstack([ps[-1], ext]), axis=0).T))) if len(ext) else 0.0
        ext_len.append(el)
        a.set_title(f"{lab}: tangent extension = {el:.1f} px ({el*um:.1f} µm)")
        a.legend(fontsize=7, loc="upper right")

    a = ax[0, 2]
    img_ax(a)
    sc = a.scatter(*path.T, c=arclength(path) * um, s=6, cmap="viridis")
    a.plot(*path[0], "s", c="red", ms=8); a.text(*path[0], " start", color="red", fontsize=10)
    a.plot(*path[-1], "^", c="red", ms=8); a.text(*path[-1], " end", color="red", fontsize=10)
    fig.colorbar(sc, ax=a, fraction=0.04, label="arc length (µm)")
    a.set_title(f"final centreline (length = {meas['length_um']:.1f} µm)")

    dev = np.hypot(*(smo - raw).T)
    ax[1, 0].plot(arclength(smo) * um, dev, c="green")
    ax[1, 0].set_xlabel("position (µm)"); ax[1, 0].set_ylabel("|smoothed − raw| (px)")
    ax[1, 0].set_title(f"smoothing displacement (max {dev.max():.2f} px)"); ax[1, 0].grid(alpha=.3)

    sr, ar = _angle_deg(raw); ss, as_ = _angle_deg(smo)
    ax[1, 1].plot(sr * um, ar, c="cyan", lw=1, label="raw"); ax[1, 1].plot(ss * um, as_, c="green", label="smoothed")
    ax[1, 1].set_xlabel("position (µm)"); ax[1, 1].set_ylabel("tangent angle (deg)")
    ax[1, 1].set_title("tangent angle along path (jaggedness → increase smoothing)")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)

    ax[1, 2].axis("off")
    ax[1, 2].text(0.02, 0.98,
                  f"skeleton path length (raw, px): {arclength(raw)[-1]:.1f}\n"
                  f"after smoothing (σ={params.smooth_sigma_px:g}):    {arclength(smo)[-1]:.1f}\n"
                  f"start extension:                {ext_len[0]:.1f}\n"
                  f"end extension:                  {ext_len[1]:.1f}\n"
                  f"final length (px):              {arclength(path)[-1]:.1f}\n"
                  f"final length (µm):              {meas['length_um']:.1f}\n\n"
                  f"tangent window (±px):           {params.tangent_halfwin_px:g}\n"
                  f"extension tangent length (px):  {params.extend_tangent_px:g}",
                  va="top", family="monospace", fontsize=11)
    fig.suptitle(name); fig.tight_layout(); figs["02_centerline"] = fig

    # ---------------- 03: rays ---------------- #
    P, N = meas["_pts"], meas["_nrm"]
    rl, rr = meas["_rl"], meas["_rr"]
    s_um = meas["_s"] * um
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    a = fig.add_subplot(gs[0, :])
    img_ax(a)
    segs = [[P[i] + N[i] * rl[i], P[i] - N[i] * rr[i]] for i in range(len(P))]
    lc = LineCollection(segs, array=meas["_width_px"] * um, cmap="viridis", linewidths=2)
    a.add_collection(lc)
    a.plot(*P.T, ".", c="white", ms=3)
    fig.colorbar(lc, ax=a, fraction=0.02, label="width (µm)")
    a.set_title("sampled rays coloured by width (white dots = sample points on the centreline)")

    a = fig.add_subplot(gs[1, 0])
    sel = sorted({0, max(len(P) // 4, 1), len(P) // 2, min(3 * len(P) // 4, len(P) - 1), len(P) - 1})
    for idx, col in zip(sel, plt.cm.tab10(np.arange(len(sel)))):
        d = np.linspace(-(rr[idx] + 6), rl[idx] + 6, 400)
        v = ndi.map_coordinates(mask_f, [P[idx, 1] + N[idx, 1] * d, P[idx, 0] + N[idx, 0] * d], order=1, cval=0)
        a.plot(d * um, v, c=col, label=f"s = {s_um[idx]:.0f} µm")
        a.plot([-rr[idx] * um, rl[idx] * um], [0.5, 0.5], "o", c=col)
    a.axhline(0.5, c="grey", ls=":"); a.axvline(0, c="k", lw=.5)
    a.set_xlabel("signed distance along normal (µm; negative = right side)")
    a.set_ylabel("interpolated mask value")
    a.set_title("ray profiles: edge = where the value crosses 0.5 (dots)"); a.legend(fontsize=8)

    a = fig.add_subplot(gs[1, 1])
    a.plot(s_um, rl * um, "o-", c="tab:blue", ms=3, label="left radius (+normal)")
    a.plot(s_um, rr * um, "o-", c="tab:red", ms=3, label="right radius (−normal)")
    a.set_xlabel("position (µm)"); a.set_ylabel("radius (µm)")
    a.set_title(f"left vs right radius (median |L−R|/(L+R) = {np.median(np.abs(rl-rr)/np.maximum(rl+rr,1e-9)):.2f})")
    a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle(name); fig.tight_layout(); figs["03_rays"] = fig

    # ---------------- 04: volume ---------------- #
    r_um = 0.5 * meas["_width_px"] * um
    l_um = np.diff(meas["_s"]) * um
    r1, r2 = r_um[:-1], r_um[1:]
    seg_vol = np.pi * l_um / 3 * (r1**2 + r1 * r2 + r2**2) / 1e3          # pl
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    a = fig.add_subplot(gs[0, :])
    hw = float(max(rl.max(), rr.max()) + 4)
    S, dense, offs = straighten_worm(res, params, hw)
    a.imshow(S, cmap="gray", vmin=0, vmax=1.5, aspect="equal",
             extent=(0, dense[-1] * um, hw * um, -hw * um))
    a.plot(s_um, rl * um, "-", c="gold", lw=1, label="measured edge (L / R)")
    a.plot(s_um, -rr * um, "-", c="gold", lw=1)
    a.plot(s_um, r_um, "--", c="cyan", lw=1.5, label="frustum radius ±r")
    a.plot(s_um, -r_um, "--", c="cyan", lw=1.5)
    for x in s_um:
        a.axvline(x, c="magenta", lw=.4, alpha=.6)
    a.set_xlabel("position along worm (µm)"); a.set_ylabel("offset from centreline (µm)")
    a.set_title(f"straightened worm with frustum model  ({len(l_um)} frustums, total volume = "
                f"{meas['volume_pl']:.1f} pl)")
    a.legend(loc="upper right", fontsize=8)

    a = fig.add_subplot(gs[1, 0])
    mid = (meas["_s"][:-1] + meas["_s"][1:]) / 2 * um
    a.bar(mid, seg_vol, width=l_um * 0.95, color="seagreen")
    a.set_xlabel("position (µm)"); a.set_ylabel("segment volume (pl)")
    a.set_title("volume contributed by each frustum"); a.grid(alpha=.3)

    a = fig.add_subplot(gs[1, 1])
    a.plot(meas["_s"][1:] * um, np.cumsum(seg_vol), "o-", c="seagreen", ms=3)
    a.axhline(meas["volume_pl"], c="grey", ls=":")
    a.axhline(meas["cylinder_volume_pl"], c="red", ls="--",
              label=f"cylinder approx = {meas['cylinder_volume_pl']:.1f} pl")
    a.set_xlabel("position (µm)"); a.set_ylabel("cumulative volume (pl)")
    a.set_title("cumulative volume vs cylinder approximation"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle(name); fig.tight_layout(); figs["04_volume"] = fig

    # ---------------- 05: sensitivity ---------------- #
    L_px = meas["length_um"] / um
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ivs = np.unique(np.round(np.linspace(2, max(0.12 * L_px, 8), 12)))
    vols = volume_vs_interval(path, mask_f, edt, params, ivs)
    ax[0].plot(ivs, vols, "o-", c="purple")
    ax[0].axvline(params.interval_px, c="grey", ls="--", label=f"chosen = {params.interval_px:g}")
    ax[0].set_xlabel("sampling interval (px)"); ax[0].set_ylabel("volume (pl)")
    ax[0].set_title(f"vs sampling interval (CV = {vols.std(ddof=1)/vols.mean()*100:.1f}%)")
    ax[0].legend(fontsize=8)

    sig = [0, 1, 2, 3, 5, 8]
    Ls, Vs = [], []
    for sg in sig:
        pth, _, _ = extend_path(smooth_path(raw, sg), mask_f, params)
        mm = measure_path(pth, mask_f, edt, params)
        Ls.append(mm["length_um"]); Vs.append(mm["volume_pl"])
    ax[1].plot(sig, Ls, "o-", c="tab:blue"); ax[1].set_ylabel("length (µm)", color="tab:blue")
    ax[1].set_xlabel("skeleton smoothing σ (px)")
    b = ax[1].twinx(); b.plot(sig, Vs, "s--", c="tab:green"); b.set_ylabel("volume (pl)", color="tab:green")
    ax[1].axvline(params.smooth_sigma_px, c="grey", ls="--")
    ax[1].set_title("vs smoothing of the centreline")

    hws = [1, 2, 3, 5, 8, 12]
    Vh = [measure_path(path, mask_f, edt, replace(params, tangent_halfwin_px=h))["volume_pl"] for h in hws]
    ax[2].plot(hws, Vh, "o-", c="tab:orange")
    ax[2].axvline(params.tangent_halfwin_px, c="grey", ls="--")
    ax[2].set_xlabel("tangent half-window (px)"); ax[2].set_ylabel("volume (pl)")
    ax[2].set_title("vs normal-direction estimate")
    for a_ in (ax[0], ax[2]):
        a_.grid(alpha=.3)
    fig.suptitle(name + " — flat curves = robust measurement"); fig.tight_layout(); figs["05_sensitivity"] = fig

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k, f in figs.items():
            f.savefig(os.path.join(out_dir, k + ".png"), dpi=120)
    if show:
        plt.show()
    for f in figs.values():
        plt.close(f)
    return list(figs)


# --------------------------------------------------------------------------- #
# 6. Batch processing
# --------------------------------------------------------------------------- #
def _read_mask(path) -> np.ndarray:
    """Read a (TIFF) mask file -> 2D boolean array."""
    try:
        import imageio.v2 as iio
        a = iio.imread(path)
    except ImportError:
        a = io.imread(path)
    a = np.squeeze(np.asarray(a))
    if a.ndim == 3 and a.shape[-1] in (3, 4):      # RGB(A) -> first channel
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {a.shape}")
    return a > 0


def collect_mask_files(path, recursive=False, exts=(".tif", ".tiff")):
    """All .tif/.tiff files (case-insensitive) in a folder, or [path] if it is a file."""
    from pathlib import Path
    path = Path(path)
    if path.is_file():
        return [path]
    it = path.rglob("*") if recursive else path.iterdir()
    return sorted(f for f in it if f.is_file() and f.suffix.lower() in exts)


def qc_flags(row, iou_min=0.9) -> str:
    """Heuristic flags - a prompt to look at the diagnostic figure, not a verdict."""
    f = []
    if row["model_iou"] < iou_min:
        f.append("low_iou")
    if row["touches_border"]:
        f.append("touches_border")
    if row["had_holes"]:
        f.append("had_holes")
    if row["skeleton_ratio"] < 0.9:
        f.append("branchy_skeleton")
    if not (0.8 <= row["width_ray_vs_edt"] <= 1.3):
        f.append("ray_edt_mismatch")
    return ",".join(f)


def run_batch(input_path, params: Params = Params(), out_dir="wormsizer_out", recursive=False,
              plots=True, debug_n=0, show=False, iou_min=0.9, progress_cb=None):
    """
    Measure every TIFF mask in a folder (each file = ONE worm).

    Writes to `out_dir`:
      results.csv   one row per mask (measurements + QC columns + qc_flags)
      failed.csv    files that could not be measured, with the error (only if any)
      params.json   settings used
      summary.png   distributions over the whole folder
      plots/        <name>.png  six-panel diagnostic per mask       (if plots=True)
      debug/<name>/ five step-by-step figures per mask              (first `debug_n` masks;
                                                                     -1 = all)
    Returns (DataFrame, list_of_failures).
    """
    import json
    import time
    from dataclasses import asdict

    files = collect_mask_files(input_path, recursive)
    if not files:
        raise FileNotFoundError(f"no .tif/.tiff files found in {input_path}")
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    rows, failed, n_debug = [], [], 0
    t0 = time.time()
    for k, f in enumerate(files, 1):
        if progress_cb:
            progress_cb(k - 1, len(files), f.name)   # may raise to abort (GUI cancel)
        name = f.stem
        tag = f"[{k}/{len(files)}] {f.name}"
        try:
            mask = _read_mask(f)
            res, skipped = analyze_mask(mask, params, name, mode="single")
            if not res:
                raise RuntimeError(skipped[0][1] if skipped else "no worm found in mask")
            r = res[0]
            row = r["row"]
            row["file"] = str(f)
            row["qc_flags"] = qc_flags(row, iou_min)
            rows.append(row)
            print(f"{tag}: L={row['length_um']:.0f} µm  V={row['volume_pl']:.1f} pl  "
                  f"IoU={row['model_iou']:.3f}  {row['qc_flags'] and '⚠ ' + row['qc_flags']}")
        except Exception as e:                                   # one bad file must not stop the batch
            failed.append(dict(file=str(f), error=repr(e)))
            print(f"{tag}: FAILED - {e!r}")
            continue
        try:
            if plots:
                plot_worm(r, params, os.path.join(out_dir, "plots", f"{name}.png"), show=show)
            if debug_n < 0 or n_debug < debug_n:
                plot_debug(r, params, os.path.join(out_dir, "debug", name), show=show)
                n_debug += 1
        except Exception as e:
            print(f"    (plotting failed for {f.name}: {e!r})")

    if progress_cb:
        progress_cb(len(files), len(files), "")
    df = pd.DataFrame(rows)
    if len(df):
        first = ["image", "file", "length_um", "mid_width_um", "mean_width_um", "volume_pl",
                 "cylinder_volume_pl", "surface_area_um2", "mask_area_um2", "model_iou", "qc_flags"]
        df = df[first + [c for c in df.columns if c not in first and c != "worm_id"]]
        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
        plot_summary(df, os.path.join(out_dir, "summary.png"), show=show)
    if failed:
        pd.DataFrame(failed).to_csv(os.path.join(out_dir, "failed.csv"), index=False)
    with open(os.path.join(out_dir, "params.json"), "w") as fh:
        json.dump({k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in asdict(params).items()}, fh, indent=2)

    n_flag = int((df["qc_flags"] != "").sum()) if len(df) else 0
    print(f"\nDone in {time.time() - t0:.0f}s: {len(df)} measured, {len(failed)} failed, "
          f"{n_flag} with QC flags.  Output in {out_dir}/")
    return df, failed


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of .tif/.tiff masks (each file = one worm), or a single file")
    ap.add_argument("--out", default="wormsizer_out")
    ap.add_argument("--um-per-px", type=float, default=2.0)
    ap.add_argument("--interval", type=float, default=10.0, help="sampling interval (px)")
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="skip masks smaller than this many px (default 0 = no filter)")
    ap.add_argument("--close-radius", type=int, default=0)
    ap.add_argument("--smooth", type=float, default=3.0, help="skeleton smoothing sigma (px)")
    ap.add_argument("--recursive", action="store_true", help="also search sub-folders")
    ap.add_argument("--no-plots", action="store_true", help="skip the per-mask diagnostic figures")
    ap.add_argument("--debug-n", type=int, default=0,
                    help="save step-by-step debug figures for the first N masks (-1 = all)")
    ap.add_argument("--iou-min", type=float, default=0.9, help="IoU below this raises the low_iou flag")
    ap.add_argument("--show", action="store_true", help="open figures on screen (not for big batches)")
    a = ap.parse_args()

    if not a.show:
        import matplotlib
        matplotlib.use("Agg")
    p = Params(um_per_px=a.um_per_px, interval_px=a.interval, min_area_px=a.min_area,
               close_radius=a.close_radius, smooth_sigma_px=a.smooth)
    run_batch(a.input, p, a.out, recursive=a.recursive, plots=not a.no_plots,
              debug_n=a.debug_n, show=a.show, iou_min=a.iou_min)


if __name__ == "__main__":
    main()