# Mock data

To generate the mock data in the parent folder, run the following:

Generate population catalog and observed catalog.
1. `python catalog.py`
Optional argument can be added `--N `, where `N` is the number of events.

2. `python gen_posterior.py`.
Optional argument `--N --Npos`, where `Npos` is the number of posterior samples per event and `N` need to match with step 1 to read the correct catalog file.

