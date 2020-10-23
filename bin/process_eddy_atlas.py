######################################################
#
# AUTHOR: ALEKSI NUMMELIN
# YEAR:   2016-2020
#
#
# This script will load the eddy trajectory atlas
# and bin the radius, average speed (of the core),
# as well as the velocity components of the core,
# into a regular lat,lon grid (0.25deg resolution).
#
# The speed and velocity components are calculated
# from the eddy locations, whereas the radius is part
# of the atlas.
# 
######################################################
#
import numpy as np
from netCDF4 import Dataset
from scipy import interpolate
from joblib import Parallel, delayed
import tempfile
import shutil
import os
import dist
#
load      = True
variables = ['radius','speed_average','u','v']
scale     = 1 # 
#
if load:
    data=Dataset('../data/raw/eddy_trajectory_2.0exp_*.nc')
    inds=np.where(np.diff(data.variables['track'][:]))[0]  
    #
    inds0=np.arange(len(inds)).astype(np.int)
    inds0[1:]=inds[:-1]+1
    times=data['time'][inds]-data['time'][inds0]
    #
    times_all=data['time'][:]
    lat_all=data['latitude'][:]
    lon_all=data['longitude'][:]
    lon_all[np.where(lon_all>180)]=lon_all[np.where(lon_all>180)]-360
    #
    lat=data.variables['latitude'][inds0]
    lon=data.variables['longitude'][inds0]
    #
    lat_end=data.variables['latitude'][inds]
    lon_end=data.variables['longitude'][inds]
#
#############################################
# --- CALCULATE THE SPEED OF THE CORE  --- #                                                                                                                            
############################################
#
def calc_core_bin(j,lat_all,lon_all,inds0,inds,gy,gx,core_map,flag=None,vardat=None):
    '''Bin eddy core properties'''
    #
    la        = lat_all[int(inds0[j]):int(inds[j]+1)]
    lo        = lon_all[int(inds0[j]):int(inds[j]+1)]
    #
    if flag in ['u']:
        var     = np.sign(lo[1:]-lo[:-1])*dist.distance((la[:-1],lo[:-1]),(la[:-1],lo[1:]))*1E3/(24*3600.)
    elif flag in ['v']:
        var     = np.sign(la[1:]-la[:-1])*dist.distance((la[:-1],lo[:-1]),(la[1:],lo[:-1]))*1E3/(24*3600.)
    elif flag in ['radius', 'speed_average']:
        var     = vardat[int(inds0[j]):int(inds[j]+1)]
    #elif flag in ['speed_average']
    #  var     = data['speed_average'][int(inds0[j]):int(inds[j]+1)]
    #
    fy=interpolate.interp1d(gy,np.arange(720))
    fx=interpolate.interp1d(gx,np.arange(1440))
    y=fy(0.5*(la[1:]+la[:-1])).astype(int)
    x=fx(0.5*(lo[1:]+lo[:-1])).astype(int)
    #
    for k in range(len(y)):
        c=len(np.where(~np.isnan(core_map[:,y[k],x[k]]))[0])
        if c>=core_map.shape[0]:
            #just in case there would be a very large amount in the same bin
            continue
        core_map[c,y[k],x[k]]=var[k]
        #
        if j%1000==0:
            print(j)

############################################################################################
# SORT OF NEAREST NEIGHBOUR APPROACH, WHERE VELOCITIES ARE BINNED IN A PRE-DETERMINED GRID #
#
for v,var in enumerate(variables):
    print(var)
    ny=int(720/scale)
    nx=int(1440/scale)
    ne=int(250*(scale**2))
    grid_x, grid_y = np.mgrid[-180:180:complex(0,nx), -90:90:complex(0,ny)]
    #
    folder2 = tempfile.mkdtemp()
    path1   = os.path.join(folder2, 'inds0.mmap')
    path2   = os.path.join(folder2, 'inds.mmap')
    path3   = os.path.join(folder2, 'lat_all.mmap')
    path4   = os.path.join(folder2, 'lon_all.mmap')
    path5   = os.path.join(folder2, 'core_map.mmap')
    path7   = os.path.join(folder2, 'gx.mmap')
    path8   = os.path.join(folder2, 'gy.mmap')
    #
    inds0_m    = np.memmap(path1, dtype=float, shape=(len(inds)), mode='w+')
    inds_m     = np.memmap(path2, dtype=float, shape=(len(inds)), mode='w+')
    lon_all_m  = np.memmap(path3, dtype=float, shape=(len(lat_all)), mode='w+')
    lat_all_m  = np.memmap(path4, dtype=float, shape=(len(lat_all)), mode='w+')
    core_map   = np.memmap(path5, dtype=float, shape=(ne,ny,nx), mode='w+')
    gx         = np.memmap(path7, dtype=float, shape=(nx), mode='w+')
    gy         = np.memmap(path8, dtype=float, shape=(ny), mode='w+')
    #
    inds0_m[:]            = inds0
    inds_m[:]             = inds
    lon_all_m[:]          = lon_all
    lat_all_m[:]          = lat_all
    core_map[:]           = np.ones((ne,ny,nx))*np.nan
    gx[:]                 = grid_x[:,0]
    gy[:]                 = grid_y[0,:]
    if var in ['radius','speed_average']:
        path9  = os.path.join(folder2, 'data.mmap')
        vardat = np.memmap(path9, dtype=float, shape=(len(lat_all)), mode='w+')
        if var in ['radius']:
            vardat[:] = data['speed_radius'][:]
        elif var in ['speed_average']:
            vardat[:] = data['speed_average'][:]
    else:
        vardat=None
    #
    #SERIAL VERSION MIGHT WORK BETTER DEPENDING ON YOUR MACHINE
    #for j in range(len(inds)):
    #   calc_core_bin(j,lat_all,lon_all,inds0,inds,gy,gx,core_map,flag=var,vardat=vardat)
    num_cores=10
    Parallel(n_jobs=num_cores)(delayed(calc_core_bin)(j,lat_all_m,lon_all_m,inds0_m,inds_m,gy,gx,core_map,flag=var,vardat=vardat) for j in range(len(inds)))
    #
    core_map    = np.array(core_map)
    core_count  = core_map.copy()
    core_count[np.where(~np.isnan(core_count))]=1
    core_count=np.nansum(core_count,0)
    mask=np.zeros(core_count.shape)
    mask[np.where(core_count==0)]=1
    #
    np.savez('../data/processed/eddy_core_'+str(var)+'_binned_scale'+str(scale)+'.npz', grid_x=grid_x.T, grid_y=grid_y.T, var_grid=np.nanmean(core_map,0),core_count=core_count, mask=mask)
    #
    try:
        shutil.rmtree(folder2)
    except OSError:
        pass

