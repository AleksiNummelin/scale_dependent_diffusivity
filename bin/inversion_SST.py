############################################
# AUTHOR: ALEKSI NUMMELIN
# YEAR:   2016-2020
#
# This script will be be called
# from outside by run_inversion.py
#
# This is the inversion workhorse and will:
# 
# 1) Load the SST data
# 2) Calculate the highpass filtered anomalies
# 3) Optionally regrid the anomalies to a coarser grid
# 4) Carry out the inversion (optionally with ensemble approach)
# 5) Save the inversted data (diffusivity, velocity, relaxation timescale)
# 
#############################################

def load_data(block_cols,block_rows,regrid=True,dg=None,itype='conservative',dg0=0.25,dt=1,datapath='../data/',var='sst',y0=None,y1=None):
    '''load data in parallel'''
    # open smooth data
    data1=xr.open_mfdataset(datapath+'processed/smooth_annual_files_y4.0deg_x8.0deg*.nc',concat_dim='time',combine='nested')
    data1=data1.chunk({'time':365,'y':720,'x':720})
    # open original data
    data=xr.open_mfdataset(datapath+'raw/sst.day.mean.*.v2.nc').squeeze()
    data=data.chunk({'time':365,'lat':720,'lon':720})
    # get shape
    nt=data.dims['time']
    ny=data.dims['lat']
    nx=data.dims['lon']
    # time bounds - much better approaches exist
    if y0==None:
       y0=int(str(data.time.isel(time=0).values)[:4])
    if y1==None:
       y1=int(str(data.time.isel(time=-1).values)[:4])
    #
    years=[]
    for dates in data.time.values.astype(str):
        years.append(int(dates[:4]))
    #
    years=np.array(years)
    #
    #setup regridding
    if regrid:
       tr_out=np.ones((nt,len(block_rows),len(block_cols)))*np.nan
       print('regrid')
       lat_out     = np.arange(np.floor(data.lat.min())+dg/2, np.ceil(data.lat.max()), dg)
       lon_out     = np.arange(np.floor(data.lon.min())+dg/2, np.ceil(data.lon.max()), dg)
       lat_in      = data.lat.values
       lon_in      = data.lon.values
       #
       if itype in ['conservative']:
          lat_b_in   = np.arange(data.lat.min()-dg0/2, data.lat.max()+dg0, dg0)
          lon_b_in   = np.arange(data.lon.min()-dg0/2, data.lon.max()+dg0, dg0)
          #
          ds_in      = xr.Dataset({'lat': (['lat'], lat_in),'lon': (['lon'], lon_in),'lat_b':(['lat_b'], lat_b_in), 'lon_b': (['lon_b'], lon_b_in), })
       else:
          ds_in      = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out), })
       #
       if itype in ['conservative']:
          lat_b_out  = np.arange(np.floor(data.lat.min()), np.ceil(data.lat.max())+dg/2, dg)
          lon_b_out  = np.arange(np.floor(data.lon.min()), np.ceil(data.lon.max())+dg/2, dg)
          ds_out     = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out),'lat_b':(['lat_b'], lat_b_out), 'lon_b': (['lon_b'], lon_b_out), })
          #
          if var in ['sst']:
               mask       = xr.DataArray(np.isfinite(data.sst.isel(time=0).values),dims=('lat','lon'),coords={'lat': (['lat'], data.lat.values),'lon': (['lon'], data.lon.values)},name='mask')
          elif var in ['sla']:
               mask       = xr.DataArray(np.isfinite(data.sla.isel(time=0).values),dims=('lat','lon'),coords={'lat': (['lat'], data.lat.values),'lon': (['lon'], data.lon.values)},name='mask')
       else:
          ds_out     = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out),})
       #
       regridder   = xe.Regridder(ds_in, ds_out, itype, filename='/home/anummel1/Projects/MicroInv/xesmf_weights/'+var+'_'+str(dg0)+'_to_'+str(dg)+'.nc', reuse_weights=True)
       for y,year in enumerate(range(y0,y1+1),dt):
           print(y)
           tinds=np.where(np.logical_and(years>=year,years<=year+dt))[0]
           print('load')
           if var in ['sst']:
              sst_in   = data.sst.isel(time=tinds).values-data1.sst.isel(time=tinds).values
           elif var in ['sla']:
              sst_in    = data.sla.isel(time=tinds).values 
           print(regrid)
           sst_out  = regridder(sst_in)
           print('select')
           sst_out=sst_out[:,block_rows,:][:,:,block_cols]
           print('pickup values')
           if itype in ['conservative']:
              mask_out=regridder(mask)
              mask_out=mask_out[block_rows,:][:,block_cols]
              mask_out.data[np.where(mask_out<0.5)]=0
              tr_out[tinds,:,:]=sst_out/mask_out.values
           else:
              tr_out[tinds,:,:]=sst_out
    elif not regrid:
       print('load')
       tinds    = np.where(np.logical_and(years>=y0,years<=y1))[0]
       if var in ['sst']:
          sst_out0 = data.sst.isel(time=tinds, lat=block_rows, lon=block_cols).values
          sst_out1 = data1.sst.isel(time=tinds, y=block_rows, x=block_cols).values
          tr_out   = sst_out0-sst_out1
       elif var in ['sla']:
          tr_out   = data.sla.isel(time=tinds, lat=block_rows, lon=block_cols).values
    #
    return tr_out

# open dask cluster
cluster = LocalCluster(n_workers=12)
client = Client(cluster)
#
data_dir0 = '../data/raw/'
savepath  = '../data/processed/'
#
fnames=sorted(os.listdir(data_dir0+'sst.day.mean.*.v2.nc'))
# spatial and time-resolution of the raw data
dg0=0.25
Dt_secs=3600*24
if not regrid:
   dg=dg0
   itype=None

# lags to be considered. Nummelin et al. 2018 showed that 10 days is usualy enough
taus=[1,2,3,4,5,6,8,10]
num_cores = 32
#
data0=xr.open_dataset(data_dir0+fnames[0]).squeeze()
# add lat,lon bounds
lat_in         = data0.lat.values
lon_in         = data0.lon.values
lat_b_in       = np.arange(data0.lat.min()-dg0/2, data0.lat.max()+dg0, dg0)
lon_b_in       = np.arange(data0.lon.min()-dg0/2, data0.lon.max()+dg0, dg0)
ds_in          = xr.Dataset({'lat': (['lat'], lat_in),'lon': (['lon'], lon_in),'lat_b':(['lat_b'], lat_b_in), 'lon_b': (['lon_b'], lon_b_in), })
#
if regrid:
   #create the target grid
   xx = data0.lon.values
   yy = data0.lat.values
   #
   itype='conservative'
   #
   lat_b_out=np.arange(np.floor(np.min(yy)), np.ceil(np.max(yy))+dg/2, dg)
   lon_b_out=np.arange(np.floor(np.min(xx)), np.ceil(np.max(xx))+dg/2, dg)
   lat_out=np.arange(np.floor(np.min(yy))+dg/2, np.ceil(np.max(yy)), dg)
   lon_out=np.arange(np.floor(np.min(xx))+dg/2, np.ceil(np.max(xx)), dg)
   #
   # target grid as an empty Dataset
   ds_out = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out),'lat_b':(['lat_b'], lat_b_out), 'lon_b': (['lon_b'], lon_b_out), })
   # create the regridder
   regridder = xe.Regridder(ds_in, ds_out, itype, filename='../data/weights/'+var+'_'+str(dg0)+'_to_'+str(dg)+'.nc', reuse_weights=False)
else:
   lat_out = lat_in
   lon_out = lon_in
#
# these settings will depend on the memory avaible on your machine
# the inversion will loop over the partitions and each
# partition will need to fit into the memory!
if dg==0.25:
   Partition_rows=4
   Partition_cols=4
elif dg<0.5:
   Partition_rows=2
   Partition_cols=2
else:
   Partition_rows=1
   Partition_cols=1
#
ny = len(lat_out)
nx = len(lon_out)
#
Block_row_size = int(np.ceil(1.0*ny/Partition_rows));
Block_col_size = int(np.ceil(1.0*nx/Partition_cols));
blknum         = 0
# 5-point stencil
Stencil_center = 2
Stencil_size   = 5

# intialize output
if not monte_carlo:
    Kx  = np.ones((len(taus),ny,nx))*np.nan
    Ky  = np.ones((len(taus),ny,nx))*np.nan
    Kxy = np.ones((len(taus),ny,nx))*np.nan
    Kyx = np.ones((len(taus),ny,nx))*np.nan
    U   = np.ones((len(taus),ny,nx))*np.nan
    V   = np.ones((len(taus),ny,nx))*np.nan
    R   = np.ones((len(taus),ny,nx))*np.nan
    DR  = np.ones((ny,nx))*np.nan
#
# timestep of the data (daily)
Dt_secs=3600*24 
#
#main loop over the partititions
for b_row in range(Partition_rows):
  rowStart = b_row*Block_row_size;
  for b_col in range(Partition_cols):
    colStart  = b_col*Block_col_size
    blknum    = blknum+1;
    print('calculating block '+str(blknum)+' of '+str(Partition_rows*Partition_cols)+' rows '+str(rowStart)+'-'+str(rowStart+Block_row_size)+ ' ,cols '+str(colStart)+'-'+str(colStart+Block_col_size))
    #
    block_rows      = np.arange(rowStart-1,rowStart+Block_row_size+1).astype('int')
    block_cols      = np.arange(colStart-1,colStart+Block_col_size+1).astype('int')
    #
    block_rows[np.where(block_rows<0)]    = 0
    block_rows[np.where(block_rows>ny-1)] = ny-1
    block_cols[np.where(block_cols<0)]    = block_cols[np.where(block_cols<0)]+nx
    block_cols[np.where(block_cols>nx-1)] = block_cols[np.where(block_cols>nx-1)]-nx
    block_rows = np.unique(block_rows) #clean the array from dublicates
    #
    xx2 = lon_out[block_cols]
    yy2 = lat_out[block_rows]
    # load the data and regrid if desired - this will take a while
    tr_out      = load_data(block_cols,block_rows,regrid=regrid,dg=dg,itype=itype,dg0=0.25,datapath=data_dir0, var=var)
    #
    nt2,ny2,nx2 = tr_out.shape
    # detrend
    j1,i1    = np.where(np.isfinite(np.sum(tr_out,0)))
    trc_anom = np.ones(tr_out.shape)*np.nan
    trc_anom[:,j1,i1] = detrend(tr_out[:,j1,i1],axis=0)
    # due to legacy reasons the inversion expects the following order
    x_grid = np.swapaxes(np.swapaxes(trc_anom,0,2),0,1)
    #
    block_lon,block_lat = np.meshgrid(xx2,yy2)
    jind        = block_rows[1:-1]
    iind        = block_cols[1:-1]
    iinds,jinds = np.meshgrid(iind,jind)
    jinds       = jinds.flatten()
    iinds       = iinds.flatten()
    # Loop over the desired lags.
    # If monte_carlo is defined i.e. 'ens'>1 then the inversion
    # will split the data (in time) into 'ens' number of chuncks
    # and carry out the inversion for each chunck separately
    for tt,Tau in enumerate(taus):
        print(Tau)
        if (not monte_carlo):
            U_block,V_block,Kx_block,Ky_block,Kxy_block,Kyx_block,R_block,res_block = mutils.inversion(x_grid,block_rows,block_cols,block_lon,block_lat,nx2,ny2,nt2,Stencil_center,Stencil_size,Tau,Dt_secs,num_cores=num_cores,b_9points=False)
        elif monte_carlo:
            print('monte carlo with '+str(ens)+' members')
            U_block,V_block,Kx_block,Ky_block,Kxy_block,Kyx_block,R_block,res_block = mutils.inversion(x_grid,block_rows,block_cols,block_lon,block_lat,nx2,ny2,nt2,Stencil_center,Stencil_size,Tau,Dt_secs,num_cores=num_cores, b_9points = False, dt_min = dt_min, ens = ens)
        if monte_carlo:
            if tt==0:
               Kx = Kx_block[np.newaxis,]
               Ky = Ky_block[np.newaxis,]
               U  = U_block[np.newaxis,]
               V  = V_block[np.newaxis,]
               R  = R_block[np.newaxis,]
               res = res_block[np.newaxis,]
            else:
               Kx = np.concatenate([Kx,  Kx_block[np.newaxis,]],axis=0)
               Ky = np.concatenate([Ky,  Ky_block[np.newaxis,]],axis=0)
               U  = np.concatenate([U,   U_block[np.newaxis,]],axis=0)
               V  = np.concatenate([V,   V_block[np.newaxis,]],axis=0)
               R  = np.concatenate([R,   R_block[np.newaxis,]],axis=0)
               res= np.concatenate([res, res_block[np.newaxis,]],axis=0) 

        elif not monte_carlo:
            Kx[tt,jinds,iinds] = Kx_block.flatten()
            Ky[tt,jinds,iinds] = Ky_block.flatten()
            U[tt,jinds,iinds]  = U_block.flatten()
            V[tt,jinds,iinds]  = V_block.flatten()
            R[tt,jinds,iinds]  = R_block.flatten()
    if monte_carlo:
         if regrid and var in ['sst']:
             fname = 'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_coarse_ave_'+str(dg)+'_'+itype+'_monte_carlo_AvePeriod_'+str(dt_min//365)+'_block_'+str(blknum).zfill(3)
         elif regrid and var not in ['sst']:
             fname = 'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB_coarse_ave_'+str(dg)+'_'+itype+'_monte_carlo_AvePeriod_'+str(dt_min//365)+'_block_'+str(blknum).zfill(3)
         elif not regrid and var in ['sst']:
             fname = 'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_monte_carlo_AvePeriod_'+str(dt_min//365)+'_block_'+str(blknum).zfill(3)
         elif not regrid and var not in ['sst']:
             fname = 'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB_monte_carlo_AvePeriod_'+str(dt_min//365)+'_block_'+str(blknum).zfill(3)
         #
         if not os.path.exists(savepath+'monte_carlo/'):
             os.makedirs(savepath+'monte_carlo/')
         np.savez(savepath+'monte_carlo/'+fname,Kx=Kx,Ky=Ky,U=U,V=V,R=R,res=res,taus=taus,ens=ens,Lat_vector=lat_out,Lon_vector=lon_out,jinds=jinds,iinds=iinds,ny=ny,nx=nx)

print('saving...')
# close cluster
client.close()
cluster.close()
if not monte_carlo:
   if regrid:
       if var in ['sst']:
           np.savez(savepath+'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_coarse_ave_'+str(dg)+'_'+itype+'_monte_carlo_AvePeriod_'+str(dt_min//365)+'.npz',Kx=Kx,Ky=Ky,Kxy=Kxy,Kyx=Kyx,R=R,U=U,V=V,taus=taus,res=res, Lat_vector=lat_out, Lon_vector=lon_out,percentiles=percentiles,ens=ens)
       else:
           np.savez(savepath+'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB_coarse_ave_'+str(dg)+'_'+itype+'_monte_carlo_AvePeriod_'+str(dt_min//365)+'.npz',Kx=Kx,Ky=Ky,Kxy=Kxy,Kyx=Kyx,res=res,R=R,U=U,V=V,taus=taus, Lat_vector=lat_out, Lon_vector=lon_out,percentiles=percentiles,ens=ens)
   else:
       if var in ['sst']:
           np.savez(savepath+'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_monte_carlo_AvePeriod_'+str(dt_min//365)+'.npz',Kx=Kx,Ky=Ky,Kxy=Kxy,Kyx=Kyx,res=res,R=R,U=U,V=V,taus=taus, Lat_vector=lat_out, Lon_vector=lon_out,percentiles=percentiles,ens=ens)
       else:
           np.savez(savepath+'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB_monte_carlo_AvePeriod_'+str(dt_min//365)+'.npz',Kx=Kx,Ky=Ky,Kxy=Kxy,Kyx=Kyx,res=res,R=R,U=U,V=V,taus=taus, Lat_vector=lat_out, Lon_vector=lon_out,percentiles=percentiles,ens=ens)

