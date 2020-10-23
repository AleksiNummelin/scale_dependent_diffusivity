##################
# AUTHOR: Aleksi Nummelin
# YEAR:   2020
#
# The SST data has variability at short timescale and
# very large scales, which are mostly of atmospheric origin.
# We are interested in the small scale, ocean mesoscale
# dominated variability. Therefore, we will create a low
# pass filtered timeseries of SST, which can be later
# substracted from the original SSTs to create a highpassed
# SST field with mostly mesoscale signal left.
#
# In Nummelin et al. 2018 we experimented with different filter sizes
# and the ended up using (top-hat) filter with a size 4deg in lat and
# 8 deg in lon. This is also what we use here.
#
# This script will first calculate smoothing weights using the above
# mentioned filter and then apply the weights and create new
# low-pass filtered timeseries.
#
# DISCLAIMER: IF I WOULD REDO THE FILTERING NOW I WOULD USE COMBINATION OF
# XESM AND XARRAY WHICH DID NOT EXIST WHEN THIS WAS FIRST WRITTEN.

import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
#import sys
#sys.path.append('/home/anummel1/Projects/MicroInv/MicroInverse/')
from MicroInverse import MicroInverse_utils as mutils
#
import os
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import glob
#
def save_smooth(lonin,latin,timein,data_smooth,var,outpath,outfile):
    '''
    save data_smooth into a outfile
    '''
    tlen,ylen,xlen = data_smooth.shape
    print(outpath+outfile)
    ncfile = Dataset(outpath+outfile, 'w', format='NETCDF4')
    #
    t = ncfile.createDimension('time', None)
    y = ncfile.createDimension('y', ylen)
    x = ncfile.createDimension('x', xlen)
    #
    lat    = ncfile.createVariable(lat_name,'f4',('y',))
    lon    = ncfile.createVariable(lon_name,'f4',('x',))
    time   = ncfile.createVariable('time','f8',('time',))
    nc_var = ncfile.createVariable(var,'f4',('time','y','x',))
    #
    lat[:]    = latin[:]
    lon[:]    = lonin[:]
    time[:]   = timein[:]
    nc_var[:] = data_smooth[:]
    #
    ncfile.close()

def smooth_file(ff,n,filepath,fname,outpath,t_inds2,weights_out,var='sst'):
    '''
    Read in the a netcdf file and apply the smoothing weights
    '''
    #
    f0   = Dataset(filepath+fname)
    data = f0.variables[var][:].squeeze()
    #
    data_smooth=ma.masked_array(np.zeros(data.shape),mask=data.mask)
    jind,iind=ma.where(1-data[0,:,:].mask);
    for j in range(data.shape[0]): #loop over the time dimension
        data_smooth[j,jind,iind]=ma.sum(data[j,:,:].ravel()[list(t_inds2)]*weights_out[:,:,0],-1)
    #
    save_smooth(f0.variables[lon_name][:],f0.variables[lat_name][:],f0.variables['time'][:],data_smooth,var,outpath+'smooth_annual_files_y'+str(2*n*0.25)+'deg_x'+str(2*2*n*0.25)+'deg/',fname[:-3]+'_'+str(2*n*0.25)+'deg.nc')
    f0.close()


# CALCULATE WEIGHTS AND APPLY THEM
calc_weights    = True
apply_weights   = True
# width of the smoothing kernell in grid-cells (y-direction)
n = 8
# define paths
filepath = '../data/raw/'
wpath    = '../data/weights/'
outpath  = '../data/processed/'
# variable
var='sst'
lat_name='lat'
lon_name='lon'
# load one file so we can create the weights
fnames=sorted(glob.glob(filepath+'sst.day.mean.*.v2.nc'))
d1=Dataset(fnames[0])
sst=d1.variables[var][0,:,:].squeeze()
lat=d1.variables[lat_name][:]
lon=d1.variables[lon_name][:]
# first calculate weights
if calc_weights:
    dum,weights_out=mutils.smooth2D_parallel(lon,lat,sst,n=n,num_cores=30,use_weights=True,weights_only=True,use_median=False,save_weights=True,save_path=wpath)

# then apply the weights
if apply_weights:
   n_cores=8
   #
   d2=np.load(wpath+str(n)+'_degree_smoothing_weights_coslat_y'+str(n)+'_x'+str(2*n)+'.npz')
   #
   t_inds=ma.reshape(np.arange(sst.ravel().shape[0]),(sst.shape[0],sst.shape[1])) #create array of indices                                                                    
   #t_inds2=t_inds[list(weights_out[:,:,1]),list(weights_out[:,:,2])] 
   #
   folder1 = tempfile.mkdtemp()
   path1   =  os.path.join(folder1, 'weights_out.mmap')
   path2   =  os.path.join(folder1, 't_inds2.mmap')
   #
   weights_out = np.memmap(path1, dtype=float, shape=d2['weights_out'].shape, mode='w+')
   t_inds2     = np.memmap(path2, dtype=int,   shape=d2['weights_out'].shape[:2], mode='w+')
   #
   weights_out[:] = d2['weights_out'][:]
   t_inds2[:]     = t_inds[weights_out[:,:,1].astype('int'),weights_out[:,:,2].astype('int')]
   #this will give you [len(inds),n**2] shaped array of indices matching to corresponding points in data.ravel()
   Parallel(n_jobs=n_cores)(delayed(smooth_file)(ff,n,filepath,fname,outpath,t_inds2,weights_out,var=var) for ff,fname in enumerate(fnames))
   try:
       shutil.rmtree(folder1)
   except OSError:
       pass 
