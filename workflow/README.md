# Mock data

To generate the mock catalog: <br />


1. Run `python makeSelectionFunction.py` to get `selfunc_m1qz_source.pkl` and `snr_m1qz_source.pkl`. <br />
2. Run `python z-dep-catalog.py -N 30000`. <br />
In `catalog/Catalog_30000Samples_afterSelection.npz`, we have the data for a observed GW catalog. <br />

3. Run `python gen_post_cat.py -i Catalog_30000Samples_afterSelection_unlensed.npz` to get posterior for m1, m2, z. <br />
Check the number of events after the selection by the file produced by step2, something like `m1m2z_posterior_PPD_afterSelection_xxx_unlensed.npz`. <br />





