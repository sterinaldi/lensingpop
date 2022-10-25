# Mock data

To generate the mock data that include masses, redshift, and spin components, please do the following.

1. Run `python z-dep-catalog.py -N 30000`(will get ~3300 observed events) to get observed catalog (filtered by the detectability function). <br />
In the `Catalog_30000Samples_afterSelection.npz` file, we have m1, m2, redshift data. <br />
2. Run `python gen_post_cat.py -i Catalog_30000Samples_afterSelection.npz` to get posterior for m1, m2, z. <br />

Check the number of events after the selection by the file produced by step2, something like `m1m2z_posterior_PPD_afterSelection_xxx.npz`:  
3. Run `spin_pop.py -N xxx` to get spin catalog and posterior.

4. Combine the data into 1 file, run `combine_data.py -N xxx`.

It contains detector frame m1,m2 posteriors, redshift posterior and spin components' posteriors of the observed catalog.





