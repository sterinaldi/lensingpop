all: data plots

plots: rec_interpolators.pkl
data: endo3_bbhpop-LIGO-T2100113-v12.hdf5

endo3_bbhpop-LIGO-T2100113-v12.hdf5:
	wget https://zenodo.org/record/5546676/files/endo3_bbhpop-LIGO-T2100113-v12.hdf5

rec_interpolators.pkl: gen_selfunc.py endo3_bbhpop-LIGO-T2100113-v12.hdf5
	python3 $<
	echo "42"

