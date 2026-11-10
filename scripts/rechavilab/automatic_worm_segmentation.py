"""
Automatic worm segmentation with micro-sam.

Can be used in two ways:

1. Imported (this is what the GUI does):

       from automatic_worm_segmentation import load_segmenter, segment_folder
       predictor, segmenter = load_segmenter("vit_b_lm", "apg", "cpu")
       segment_folder(input_folder, output_folder, predictor, segmenter)

2. Run directly as a script on one folder (edit the SETTINGS block below).

Mask convention: 0 = background, 1, 2, 3... = objects.
Output file for  worm001.tif  is  worm001_mask.tif
"""

from pathlib import Path

import numpy as np
import imageio.v3 as imageio
import torch

from micro_sam import util
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)

IMAGE_EXTENSIONS = {".tif", ".tiff"}


# ============================================================
# STANDALONE SETTINGS (only used when running this file directly)
# ============================================================
INPUT_FOLDER = Path(r"C:\data\RechaviLab\R1")
OUTPUT_FOLDER = Path(r"C:\data\RechaviLab\R1\microsam_output")
MODEL_TYPE = "vit_b_lm"
SEGMENTATION_MODE = "apg"     # "apg" | "ais" | "amg"
DEVICE = "cpu"


# ============================================================
# FUNCTIONS
# ============================================================
def list_image_files(folder):
    """All .tif/.tiff files directly inside `folder` (sorted)."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_segmenter(model_type="vit_b_lm", segmentation_mode="apg", device="cpu"):
    """Load the micro-sam predictor + automatic segmenter ONCE."""
    if model_type is None:
        model_type = util._DEFAULT_MODEL
    return get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,
        device=torch.device(device),
        segmentation_mode=segmentation_mode,
        is_tiled=False,
    )


def segment_folder(input_folder, output_folder, predictor, segmenter,
                   progress_cb=None, log_cb=print):
    """
    Segment every TIFF in `input_folder`, writing <stem>_mask.tif into `output_folder`.

    progress_cb(done, total, current_name) is called before each image and once at the end.
    It may raise to abort the run (the exception is deliberately NOT caught here).
    A failing image is logged and skipped.

    Returns a list of (filename, error_string) for images that failed.
    """
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    files = list_image_files(input_folder)
    total = len(files)
    failures = []

    for i, image_path in enumerate(files):
        if progress_cb:
            progress_cb(i, total, image_path.name)      # outside try: lets cancel propagate
        output_path = output_folder / f"{image_path.stem}_mask.tif"
        try:
            automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmenter,
                input_path=image_path,
                output_path=output_path,
                ndim=2,
                verbose=False,
            )
            seg = np.asarray(imageio.imread(output_path))
            n_obj = len(np.unique(seg[seg != 0]))
            if log_cb:
                log_cb(f"  {image_path.name}: {n_obj} objects")
        except Exception as e:
            failures.append((image_path.name, repr(e)))
            if log_cb:
                log_cb(f"  ERROR {image_path.name}: {e!r}")

    if progress_cb:
        progress_cb(total, total, "")
    return failures


def main():
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder does not exist:\n{INPUT_FOLDER.resolve()}")
    predictor, segmenter = load_segmenter(MODEL_TYPE, SEGMENTATION_MODE, DEVICE)
    segment_folder(
        INPUT_FOLDER, OUTPUT_FOLDER, predictor, segmenter,
        progress_cb=lambda d, t, n: print(f"[{d}/{t}] {n}"),
    )
    print("Finished.")


if __name__ == "__main__":
    main()
