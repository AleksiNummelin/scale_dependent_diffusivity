###########################
# AUTHOR: ALEKSI NUMMELIN
# YEAR:   2016-2020
#
# This scrupt wil run the inversion with using both a full timeseries and
# a Monte Carlo approach with the dataset split into 'ens' chuncks
# of length 'dt_min'. The inversion is carried out for each chunck separately.
# If the length of the timeseries is shorter than ens*dt_min then
# the individual chuncks will overlap.
#
# When the whole timeseries is used the data is saved at once for each lag,
# but because the inversion is rather heavy when done with the ensemble
# approach, we will first save the data separately for each lag (tau)
# and (optionally) for smaller spatial regions (depends on the grid).
#
# Results are then combined across all lags (and sub-regions) at the end
# using postprocess_ensemble_inversion.py.
#
from xmitgcm import open_mdsdataset
import xarray as xr
import xesmf as xe
import numpy as np
import numpy.ma as ma
import sys
sys.path.append('/home/anummel1/Projects/MicroInv/MicroInverse/')
from MicroInverse import MicroInverse_utils as mutils
from scipy.signal import detrend
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import os
import calendar
from dask.distributed import Client, LocalCluster
#
# temporal lags and spatial resolution to use
taus = [1,2,3,4,5,6,8,10] 
dgs  = [0.25,0.3,0.4,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0,5.0] 
#
itype='conservative'
dt_mins = [365*10,365*5]
#
Stencil_size   = 5
Stencil_center = 2
# Parallelize loops over 'num_cores' using joblib
num_cores=32
#
for data_source in ['OI-SST','MITGCM']:
    for ens in [1,30]:
        monte_carlo= ens>1
        if monte_carlo: #ensemble inversion
            for dt_min in dt_mins:
                for dg in dgs:
                    print(data_source,var,dg,dt_min/365)
                    if data_source in ['OI-SST']:
                        regrid=(dg>0.25)
                        var='sst'
                        exec(open('inversion_SST.py').read())    
                    elif data_source in ['MITGCM']:
                        regrid=(dg>0.1)
                        exec(open('inversion_mitgcm.py').read())
        else: #normal inversion (use whole timeseries at once)
            for dg in dgs:
                print(data_source,var,dg,ens)
                if data_source in ['OI-SST']:
                    regrid=(dg>0.25)
                    var='sst'
                    exec(open('inversion_SST.py').read())
                elif data_source in ['MITGCM']:
                    regrid=(dg>0.1)
                    exec(open('inversion_mitgcm.py').read())
                        
#
# Finally combine the data and regrid the magnitude
# to a common 2deg grid to estimate the slopes.
fpath = '../data/processed/monte_carlo/'
years = [5,10]
percentiles=[5,25,50,75,95]
wpath = '../data/weights/'
imethod = 'patch'
exec(open('postprocess_ensemble_inversion.py').read())
