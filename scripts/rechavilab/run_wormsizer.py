from wormsizer import Params, run_batch

params = Params(um_per_px=2.0, interval_px=10, min_area_px=0)
df, failed = run_batch(r"C:\data\RechaviLab\Projects\BodySizeEvolution\ColdShockResistance\R1\microsam_output", params,
                       out_dir=r"C:\data\RechaviLab\Projects\BodySizeEvolution\ColdShockResistance\R1\microsam_output\results", debug_n=3)

