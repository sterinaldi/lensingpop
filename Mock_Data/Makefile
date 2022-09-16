all: data plots

plots: selection_functions/selection_function_detector_frame.pdf
data: endo3_bbhpop-LIGO-T2100113-v12.hdf5

endo3_bbhpop-LIGO-T2100113-v12.hdf5:
	wget https://zenodo.org/record/5546676/files/endo3_bbhpop-LIGO-T2100113-v12.hdf5

selection_functions/selection_function_detector_frame.pdf: gen_selfunc.py endo3_bbhpop-LIGO-T2100113-v12.hdf5
	python3 $<
	echo "42"

quicktest: gen_selfunc_small.py
	python3 $<
