#################################################
#
# AUTHOR: Aleksi Nummelin
# YEAR:   2016-2020
#
# Regridding of the GLOBCURRENT surface
# velocities by calculating the Helmholtz-Hodge
# decomposition of the velocity vectors.
#
# These velocities are then used to calculate the
# low-pass and high-pass velocities by smoothing
# the decomposed potential and stream function
# in HelmHoltzDecomposition_analysis.py script
#
#
#################################################
#
import sys
sys.path.append('../side_packages/naturalHHD/pynhhd-v1.1/')
sys.path.append('../side_packages/naturalHHD/pynhhd-v1.1/pynhhd/')
from pynhhd import nHHD
import xarray as xr
import numpy as np
import numpy #because of pynhhd
import xesmf as xe
import HelmHoltzDecomposition_analysis_utils as hutils

dgs = [0.5,0.6,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0]
dg_in = 0.25
for y, year in enumerate(range(2003,2013)):
    #
    data_in=xr.open_dataset('../data/raw/surface_currents_'+str(year)+'.nc')
    # we will use weekly averages, shorter timescales turned out to be too noisy
    data_in['time2']=data_in.time.to_index().week
    data=data_in.groupby('time2').mean(dim='time')
    # we need at least this precission
    lon, lat = np.meshgrid(np.arange(-179.875,180,0.25).astype(np.float64),np.arange(-89.875,90,0.25).astype(np.float64)) #longdouble))
    #
    nt,ny,nx = data.uo.shape
    for t,tt in enumerate(range(nt)):
        print(year,tt)
        # pickup data
        vfield=np.zeros((ny,nx,2))
        vfield[:,:,0]=data.uo.isel(time2=t).values
        vfield[:,:,1]=data.vo.isel(time2=t).values
        #
        vfield[np.where(np.isnan(vfield))]=0
        #
        ls_vels = np.zeros((len(dgs),ny,nx,2)) #large scale flow
        ss_vels = np.zeros((len(dgs),ny,nx,2)) #small scale flow
        #
        #############
        # DECOMPOSE #
        ############# 
        nhhd1 = nHHD(sphericalgrid=vfield[:,:,0].shape,spacings=(0.25,0.25),lat=lat,lon=lon)
        nhhd1.decompose(vfield,True,num_cores=25)
        r0 = hutils.gradient(nhhd1.nRu, lat, lon, [0.25,0.25], r=6371E3, rotated=True, glob=True)
        d0 = hutils.gradient(nhhd1.nD, lat, lon, [0.25,0.25], r=6371E3, rotated=False, glob=True)
        #
        #######################################################
        # REGRID - THESE ARE NOT USED IN ANY FURTHER ANALYSIS #
        # BUT KEPT HERE FOR FUTURE INTEREST                   #
        #######################################################
        if y==0 and t==0:
            dg_in=0.25
            #
            lat_in     = lat[:,0]
            lat_b_in   = np.concatenate([lat[:,0]-dg_in/2,lat[-1:,0]+dg_in/2],axis=0)
            lon_in     = lon[0,:]
            lon_b_in   = np.concatenate([lon[0,:]-dg_in/2,lon[0,-1:]+dg_in/2],axis=0)
            #
            lon_in[np.where(lon_in>180)] = lon_in[np.where(lon_in>180)]-360
            lon_b_in[np.where(lon_b_in>180)] = lon_b_in[np.where(lon_b_in>180)]-360
            ds_in      = xr.Dataset({'lat': (['lat'], lat_in),'lon': (['lon'], lon_in),'lat_b':(['lat_b'], lat_b_in), 'lon_b': (['lon_b'], lon_b_in), })
            #
            # set output grids
            #
            fpath='..data/weights/'
        for j,dg_out in enumerate(dgs):
            print(j,dg_out)
            # setup regridding
            ds_out=hutils.return_grid(dg_out)
            regridder_coarse = xe.Regridder(ds_in,ds_out,'conservative',filename=fpath+'conservative_from_'+str(dg_in)+'_to_'+str(dg_out)+'.nc',reuse_weights=True)
            regridder_smooth = xe.Regridder(ds_out,ds_in,'bilinear',filename=fpath+'bilinear_from_'+str(dg_out)+'_to_'+str(dg_in)+'.nc',reuse_weights=True,periodic=True)
            nRu_coarse = hutils.perform_regrid(regridder_coarse,nhhd1.nRu)
            nRu_smooth = hutils.perform_regrid(regridder_smooth,nRu_coarse)
            nD_coarse  = hutils.perform_regrid(regridder_coarse,nhhd1.nD)
            nD_smooth  = hutils.perform_regrid(regridder_smooth,nD_coarse)
            #RUs[j+1,:,:] = nRu_smooth
            #Ds[j+1,:,:]  = nD_smooth
            r = hutils.gradient(nRu_smooth, lat, lon,[0.25,0.25], r=6371E3, rotated=True, glob=True)
            d = hutils.gradient(nD_smooth, lat, lon,[0.25,0.25], r=6371E3, rotated=False, glob=True)
            ls_vels[j,:,:,:] = -(r+d)
            ss_vels[j,:,:,:] = -(r0+d0)+(r+d)
            #
         
        ########
        # SAVE #
        ########
        ul_out  = xr.DataArray(np.expand_dims(ls_vels[:,:,:,0],axis=1),dims=('dgs','time2','lat','lon'),coords={'dgs':(['dgs'], np.array(dgs)), 'time2': (['time2'], np.array([tt])) ,'lat': (['lat'], data.latitude.values),'lon': (['lon'], data.longitude.values)},name='ul_out')
        vl_out  = xr.DataArray(np.expand_dims(ls_vels[:,:,:,1],axis=1),dims=('dgs','time2','lat','lon'),coords={'dgs':(['dgs'], np.array(dgs)), 'time2': (['time2'], np.array([tt])) ,'lat': (['lat'], data.latitude.values),'lon': (['lon'], data.longitude.values)},name='vl_out')
        us_out  = xr.DataArray(np.expand_dims(ss_vels[:,:,:,0],axis=1),dims=('dgs','time2','lat','lon'),coords={'dgs':(['dgs'], np.array(dgs)), 'time2': (['time2'], np.array([tt])) ,'lat': (['lat'], data.latitude.values),'lon': (['lon'], data.longitude.values)},name='us_out')
        vs_out  = xr.DataArray(np.expand_dims(ss_vels[:,:,:,1],axis=1),dims=('dgs','time2','lat','lon'),coords={'dgs':(['dgs'], np.array(dgs)), 'time2': (['time2'], np.array([tt])) ,'lat': (['lat'], data.latitude.values),'lon': (['lon'], data.longitude.values)},name='vs_out')
        nRu_out = xr.DataArray(np.expand_dims(nhhd1.nRu,axis=0),dims=('time2','lat','lon'),coords={'time2': (['time2'], np.array([tt])) ,'lat': (['lat'], data.latitude.values),'lon': (['lon'], data.longitude.values)},name='r0_out')
        nD_out = xr.DataArray(np.expand_dims(nhhd1.nD,axis=0),dims=('time2','lat','lon'),coords={'time2': (['time2'], np.array([tt])) ,'lat': (['lat'], data.latitude.values),'lon': (['lon'], data.longitude.values)},name='d0_out')
        #
        data_out = ul_out.to_dataset()
        #data['ul_out'] = ul_out
        data_out['vl_out'] = vl_out
        data_out['us_out'] = us_out
        data_out['vs_out'] = vs_out
        data_out['nRu_out'] = nRu_out
        data_out['nD_out'] = nD_out
        #
        data_out.to_netcdf('../data/processed/surface_currents_'+str(year)+'_week'+str(tt).zfill(2)+'.nc')


# BY DEFINITION
# u = data['h'][:,:,0]+data['d'][:,:,0]+data['r'][:,:,0]
# v = data['h'][:,:,1]+data['d'][:,:,1]+data['r'][:,:,1]
# 
