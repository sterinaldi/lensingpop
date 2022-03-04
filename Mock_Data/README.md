# Mock data

Power-law + Peak + Delta population

Please find the catalog data in `PowerlawplusPeakplusDelta1000000Samples.npz` <br />
and observed catalog in `PowerlawplusPeakplusDelta1000000Samples_afterSelection.npz` (after applying the selection effect). 

`PowerlawplusPeakplusDelta1000000Samples.npz` has 1000000 events while `PowerlawplusPeakplusDelta1000000Samples_afterSelection.npz` has 1986 events

Chcke accessible data:

    data  = np.load("PowerlawplusPeakplusDelta1000000Samples.npz")
    print(data.files)
    #output:['m1', 'm2', 'redshift', 'snr', 'inclinationValue', 'polarisationValue', 'rightAscensionValue', 'declinationValue', 'GPStimeValue']

Load the data:
    
    m1 = data['m1']
    ...



Please find the mock posterior data in `m1m2posterior_PPD_afterSelection1000000.npz`.

It contains events' m1,m2 posteriors of the observed catalog.


The selection function in `gwdet` package is used. It gives the probability for detecting the event with parameters `m1, m2, z`
However, it is not compatible with current `scipy` version. 

Therefore, the selection function is saved in `gwdet_default_interpolator` and can be loaded by using `pickle`.


    with open("gwdet_default_interpolator", "rb") as f:

        pdet = pickle.load(f)


    pdet_value = pdet(np.array([m1,m2,redshiftValue]).T)

The input of the function should be `(N,3)`, where `N` is the number of events.
