# Mock data

To generate the mock catalog: <br />


1. Run `python makeSelectionFunction.py` to get `selfunc_m1qz_source.pkl` and `snr_m1qz_source.pkl`. <br />

2. Run `python makeCatalog.py -N 30000` to simulate an observed GW catalog. <br />

3. Run `python makePosterior.py -i Catalog_30000Samples_afterSelection_unlensed.npz` to generate the posterior samples for each GW event in the catalog. <br />

3. Run `python makePopulationPRior.py` to make different population priors for Bayes factor computation. <br />







