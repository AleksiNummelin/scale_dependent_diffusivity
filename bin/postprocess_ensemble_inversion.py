import numpy as np
import glob
import xesmf as xe
from joblib import Parallel, delayed, parallel_backend
from scipy import stats, interpolate
#
import sys
sys.path.append('../side_packages/MicroInverse/')
from MicroInverse import MicroInverse_utils as mutils

def combine_parallel(e,Kx,Ky,U,V,taus,jinds,iinds,weight_coslat,dg):
    '''
    Combine the estimates accross different lags.
    See Nummelin et al. 2018 for motivation. 
    '''
    datain ={}
    datain['Kx'] = Kx
    datain['Ky'] = Ky
    datain['U']  = U
    datain['V']  = V
    #
    ntaus, ny, nx = Kx.shape
    #weight_coslat = np.reshape(weight_coslat[jinds,iinds], (ny,nx))
    datadum = mutils.combine_Taus(datain,weight_coslat,taus,K_lim=True,dg=dg)
    #
    return datadum['Kx'], datadum['Ky'], datadum['U'], datadum['V'], datadum['tau_opt']

def parallel_interpolation(dum,lon,lat):
    ''' '''
    #
    dum_in = dum.copy()
    ny,nx  = dum_in.shape
    jinds,iinds = np.where(np.isnan(dum_in))
    ii0,jj0 = np.meshgrid(np.arange(nx),np.arange(ny))
    c = 0
    while len(jinds)>1:
        print(c,len(jinds))
        jj  = jinds[0]
        ii  = iinds[0]
        i0  = ii-nx//5
        i1  = ii+nx//5
        j0  = jj-ny//5
        j1  = jj+ny//5
        ii1 = np.arange(i0,i1)
        jj1 = np.arange(j0,j1)
        ii1[np.where(ii1<0)]    = nx+ii1[np.where(ii1<0)]
        ii1[np.where(ii1>nx-1)] = ii1[np.where(ii1>nx-1)]-nx
        jj1[np.where(jj1<0)]    = 0
        jj1[np.where(jj1>ny-1)] = ny-1
        jj1 = np.unique(jj1)
        #
        ii2,jj2 = np.meshgrid(ii1,jj1)
        #
        Dsub = dum_in[jj1,:][:,ii1]
        isub = ii0[jj1,:][:,ii1]
        jsub = jj0[jj1,:][:,ii1]
        #
        jpos,ipos = np.where(np.isfinite(Dsub))
        #
        jneg,ineg   = np.where(np.isnan(Dsub))
        Dnew        = interpolate.griddata((isub[jpos,ipos], jsub[jpos,ipos]), Dsub[jpos,ipos], (isub, jsub), method='linear',fill_value=0)
        dum_in[jj2[jneg,ineg],ii2[jneg,ineg]] = Dnew[jneg,ineg]
        jinds,iinds = np.where(np.isnan(dum_in))
        c = c+1
    #
    return dum_in

def parallel_theilslopes(var,x,lat):
    '''
    Calculate Theil-Sen estimator for the
    linear slope.
    '''
    #
    nt,ny,nx = var.shape
    var = np.reshape(var,(nt,-1))
    out = np.ones(ny*nx)*np.nan
    #
    jinds = np.where(np.isfinite(np.sum(var,0)))[0]
    lat   = lat.flatten()
    for j in jinds:
        #nonnans = np.where(np.isfinite(var[:6,j]))
        #if len(nonnans)>3:  
        out[j] = stats.theilslopes(np.log10(var[:,j].squeeze()), np.log10(111E3*x*np.cos(np.radians(lat[j]))),alpha=0.9)[0]
    #
    return np.reshape(out,(ny,nx))


def data_to_common_grid(data_out,dgs,imethod = 'patch',dg_out=2.0,y=5,weight_name='',reuse=False,frobenius=False):
    '''
    Interpolate data to a common grid using xesmf 
    '''
    # 
    grid_dest = xe.util.grid_2d(0, 360, dg_out, -90, 90, dg_out)
    K_out = np.ones((len(dgs),int(ens),)+grid_dest.lon.shape)*np.nan
    #
    for d,dg in enumerate(dgs):
        print(d)
        #
        grid = data_out['grid_'+str(5)+'_'+str(dg)]
        #
        regridder = xe.Regridder(grid,grid_dest,imethod,filename=weight_name+imethod+'_'+str(dg)+'.nc',reuse_weights=reuse)
        sign      = np.nanmin([np.sign(data_out['Kx_'+str(y)+'_'+str(d)]),np.sign(data_out['Ky_'+str(y)+'_'+str(d)])],axis=0)
        #
        if not frobenius:
            dum = sign*np.sqrt(data_out['Kx_'+str(y)+'_'+str(d)]*data_out['Ky_'+str(y)+'_'+str(d)])
        else:
            dum = sign*np.sqrt(data_out['Kx_'+str(y)+'_'+str(d)]**2 + data_out['Ky_'+str(y)+'_'+str(d)]**2)
        dum[np.where(dum<0.1)] = 0 #do not accept negative diffusivities
        # fill nans
        dum_out = Parallel(n_jobs=15)(delayed(parallel_interpolation)(dum[e,:,:],grid.lon.values,grid.lat.values) for e in np.arange(ens)) 
        dum     = np.array(dum_out)
        mask    = (dum>0)
        print(dum.shape,mask.shape)
        out  = regridder(dum)/regridder(mask)
        out[np.where(out==0)] = np.nan
        K_out[d,:,:,:]        = out
    #
    return K_out, grid_dest

def lonlat_bounds(bounds,lon,lat):
    '''bounds={'lon_min'}'''
    if bounds['lon_min']>lon.min():
        bounds['lon_min']=lon.min()
    if bounds['lat_min']>lat.min():
        bounds['lat_min']=lat.min()
    if bounds['lon_max']<lon.max():
        bounds['lon_max']=lon.max()
    if bounds['lat_max']<lat.max():
        bounds['lat_max']=lat.max()
    #
    return bounds

def adjust_bounds(bounds,dg):
    '''
     
    '''
    for key in bounds.keys():
        if 'max' in key:
            bounds[key]=np.round(bounds[key],3)+dg/2.0
        elif 'min' in key:
            bounds[key]=np.round(bounds[key],3)-dg/2.0
    #
    return bounds

def init_bounds():
    bounds={}
    bounds['lon_min'] = np.inf
    bounds['lat_min'] = np.inf
    bounds['lon_max'] = -np.inf
    bounds['lat_max'] = -np.inf
    return bounds

ki = np.where(np.array(dgs)==1.0)[0][0]
frobenius=False
##############################################################################################
# MITGCM - SST
#
data={}
data_out={}
#
for d,dg in enumerate(dgs):
    for yy, y in enumerate(years): #loop over the years
        #
        bounds=init_bounds()
        #
        fnames=sorted(glob.glob(fpath+'run_sst_AMSRE_damped_1m_global_'+str(dg)+'_full_conservative_v2_monte_carlo_AvePeriod_'+str(y)+'_block_*.npz'))
        data0 = np.load(fnames[0])
        #data_out['Lat_'+str(y)+'_'+str(d)] = data0['Lat_vector'][:]
        #data_out['Lon_'+str(y)+'_'+str(d)] = data0['Lon_vector'][:]
        #
        for key in ['Kx','Ky','U','V','tau_opt']: # create empty output
            data_out[key+'_'+str(y)+'_'+str(d)] = np.ones((data0['ens'],data0['ny'],data0['nx']))*np.nan
        #
        data_out['Lat_'+str(y)+'_'+str(d)] = np.ones((data0['ny'],data0['nx']))*np.nan
        data_out['Lon_'+str(y)+'_'+str(d)] = np.ones((data0['ny'],data0['nx']))*np.nan
        #
        jinds0,iinds0 = np.meshgrid(range(data0['ny']),range(data0['nx']))
        for f,fname in enumerate(fnames): # loop over subregions
            print(d,f)
            data = np.load(fname)
            weight_coslat = np.tile(np.cos(np.radians(data['Lat_vector'][3:-3])),(data['Lon_vector'][3:-3].shape[0],1)).T
            iinds,jinds = data['iinds'], data['jinds'] #np.meshgrid(range(data['nx']),range(data['ny']))
            #
            bounds = lonlat_bounds(bounds,data['Lon_vector'][:],data['Lat_vector'][:])
            dlon,dlat = np.meshgrid(data['Lon_vector'][3:-3],data['Lat_vector'][3:-3])
            #
            data_out['Lon_'+str(y)+'_'+str(d)][jinds,iinds] = dlon.flatten()
            data_out['Lat_'+str(y)+'_'+str(d)][jinds,iinds] = dlat.flatten()
            #
            ens  = data['ens']
            with parallel_backend('threading', n_jobs=3): 
                out = Parallel()(delayed(combine_parallel)(e,data['Kx'][:,e,],data['Ky'][:,e,],data['U'][:,e,],data['V'][:,e,],data['taus'][:],jinds,iinds,weight_coslat,dg) for e in range(ens))
            #out = Parallel(n_jobs=10)(delayed(combine_parallel)(e,data['Kx'][:,e,],data['Ky'][:,e,],data['U'][:,e,],data['V'][:,e,],data['taus'][:],jinds,iinds,weight_coslat,dg) for e in range(ens))
            #
            out = np.array(out)
            for k,key in enumerate(['Kx','Ky','U','V','tau_opt']):
                data_out[key+'_'+str(y)+'_'+str(d)][:,jinds,iinds] = np.reshape(out[:,k,:,:],(data['ens'],-1))
            #
            data.close()
        #
        bounds = adjust_bounds(bounds,dg)
        #
        data_out['grid_'+str(y)+'_'+str(dg)] = xe.util.grid_2d(bounds['lon_min'],bounds['lon_max'], dg, bounds['lat_min'], bounds['lat_max'], dg) 
        data_out['bounds_'+str(y)+'_'+str(dg)] = bounds
 
for yy, y in enumerate(years):
    # Interpolate data to a common grid
    K_out,grid_dest = data_to_common_grid(data_out,dgs,imethod = imethod, dg_out=2.0,y=y,weight_name=wpath+'MITgcm_'+imethod+'_',reuse=(yy>0),frobenius=frobenius)
    #
    # calculate the Theil-Sen estimate of the slope below 1 deg resolution.
    S_out = Parallel(n_jobs=10)(delayed(parallel_theilslopes)(K_out[:ki,e,:,:],np.array(dgs[:ki]),grid_dest.lat.values) for e in np.arange(ens))  
    S_out = np.array(S_out)
    #
    S_out_mean = parallel_theilslopes(np.nanmedian(K_out[:ki,],1),np.array(dgs[:ki]),grid_dest.lat.values)
    #
    if frobenius:
        vv='_v2'
    else:
        vv=''        
    #
    np.savez('../data/processed/MITgcm_results_'+str(y)+'_'+imethod+vv+'.npz', Kx_25=data_out['Kx_'+str(y)+'_0'][:], Ky_25=data_out['Ky_'+str(y)+'_0'][:], K_out = K_out, S_out=S_out, S_out_mean = S_out_mean, lat=grid_dest.lat.values, lon=grid_dest.lon.values,dgs=dgs,lat25 = data_out['Lat_'+str(y)+'_'+str(0)], lon25 = data_out['Lon_'+str(y)+'_'+str(0)])

###########################################################################################
# SST 
#
for d,dg in enumerate(dgs):
    print(dg)
    for y in years: #loop over the years
        #
        if dg==0.25:
            fnames = sorted(glob.glob(fpath+'uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_monte_carlo_AvePeriod_'+str(y)+'_block_???.npz'))
        else:
            fnames = sorted(glob.glob(fpath+'uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_coarse_ave_'+str(dg)+'_conservative_monte_carlo_AvePeriod_'+str(y)+'_block*.npz'))
        #
        data0 = np.load(fnames[0])
        data_out['Lat_'+str(y)+'_'+str(d)] = data0['Lat_vector'][:]
        data_out['Lon_'+str(y)+'_'+str(d)] = data0['Lon_vector'][:]
        # bounds
        bounds = init_bounds()
        bounds = lonlat_bounds(bounds,data0['Lon_vector'][:],data0['Lat_vector'][:])
        bounds = adjust_bounds(bounds,dg)
        data_out['grid_'+str(y)+'_'+str(dg)] = xe.util.grid_2d(bounds['lon_min'],bounds['lon_max'], dg, bounds['lat_min'], bounds['lat_max']-dg/2, dg)
        data_out['bounds_'+str(y)+'_'+str(dg)] = bounds
        #
        for key in ['Kx','Ky','U','V','tau_opt']: # create empty output
            data_out[key+'_'+str(y)+'_'+str(d)] = np.ones((data0['ens'],data0['ny'],data0['nx']))*np.nan
        #
        jinds0,iinds0 = np.meshgrid(range(data0['ny']),range(data0['nx']))
        for f,fname in enumerate(fnames): # loop over subregions
            print(d,f)
            data = np.load(fname)
            nd,ne,ny,nx   = data['Kx'].shape
            weight_coslat = np.tile(np.cos(np.radians(data['Lat_vector'])),(data['nx'],1)).T
            iinds,jinds   = data['iinds'], data['jinds'] #np.meshgrid(range(data['nx']),range(data['ny']))
            weight_coslat = np.reshape(weight_coslat[jinds,iinds],(ny,nx))
            ens  = data['ens']
            with parallel_backend('threading', n_jobs=3): 
                out = Parallel()(delayed(combine_parallel)(e,data['Kx'][:,e,],data['Ky'][:,e,],data['U'][:,e,],data['V'][:,e,],data['taus'][:],jinds,iinds,weight_coslat,dg) for e in range(ens))
            #out = Parallel(n_jobs=10)(delayed(combine_parallel)(e,data['Kx'][:,e,],data['Ky'][:,e,],data['U'][:,e,],data['V'][:,e,],data['taus'][:],jinds,iinds,weight_coslat,dg) for e in range(ens))
            #
            out = np.array(out)
            for k,key in enumerate(['Kx','Ky','U','V','tau_opt']):
                data_out[key+'_'+str(y)+'_'+str(d)][:,jinds,iinds] = np.reshape(out[:,k,:,:],(data['ens'],-1))
            #
            data.close()

for yy,y in enumerate(years):
    # Interpolate data to a common grid
    K_out,grid_dest = data_to_common_grid(data_out,dgs,imethod = imethod, dg_out=2.0,y=y,weight_name=wpath+'sst_'+imethod+'_',reuse=(yy>0),frobenius=frobenius)
    # calculate the Theil-Sen estimate of the slope below 1 deg resolution.    
    S_out = Parallel(n_jobs=10)(delayed(parallel_theilslopes)(K_out[:ki,e,:,:],np.array(dgs[:ki]),grid_dest.lat.values) for e in np.arange(ens))  
    S_out = np.array(S_out)
    #
    S_out_mean = parallel_theilslopes(np.nanmedian(K_out[:ki,],1),np.array(dgs[:ki]),grid_dest.lat.values)
    #
    if frobenius:
        vv='_v2'
    else:
        vv=''
    np.savez('../data/processed/SST_results_'+str(y)+'_'+imethod+vv+'.npz', Kx_25=data_out['Kx_'+str(y)+'_0'][:], Ky_25=data_out['Ky_'+str(y)+'_0'][:], K_out = K_out, S_out=S_out, S_out_mean = S_out_mean, lat=grid_dest.lat.values, lon=grid_dest.lon.values,dgs=dgs,lat25 = data_out['Lat_'+str(y)+'_'+str(0)], lon25 = data_out['Lon_'+str(y)+'_'+str(0)])
