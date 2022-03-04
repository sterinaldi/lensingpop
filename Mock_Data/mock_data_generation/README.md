# Mock data

To generate the mock data in the parent folder, run the following:

Generate population catalog and observed catalog.
1. `python catalog.py --N 1000000`
, where `N` is the number of events.

2. `python gen_posterior.py --Npos 1000`
, where `Npos` is the number of posterior samples per event

