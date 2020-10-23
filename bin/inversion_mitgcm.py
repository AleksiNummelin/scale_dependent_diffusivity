###############################
# 
# AUTHOR: ALEKSI NUMMELIN
# YEARS:  2016-2020
#
# 
#
def load_data(c,count,dc,Partition_cols,block_cols,block_rows,lon_out,lat_out,lon_b_out,lat_b_out,tr_out,dt=94,dt_max=842240,regrid=False,dg=None,itype='conservative'):
    '''load data in parallel'''
    #
    print('open '+str(count)+'-'+str(min(count+dc,dt_max+dt)))
    data=open_mdsdataset(data_dir,prefix=['tracer_snapshots'],iters=range(count,min(count+dc,dt_max+dt),dt), delta_t=900, ref_date='1993-01-01')
    #
    if regrid:
       data = data.rename({'YC':'lat','XC':'lon'})
       if Partition_cols>1:
           if lon_out[-1]-lon_out[0]<0:
               block_cols_in = np.concatenate([np.where(data.lon.values>=lon_out[0])[0],np.where(data.lon.values<=lon_out[-1])[0]])
           else:
               block_cols_in = np.where(np.logical_and(data.lon.values>=lon_out[0],data.lon.values<=lon_out[-1]))[0]
       else:
           block_cols_in = np.concatenate([np.where(data.lon.values>=lon_out[0])[0],np.arange(data.lon.shape[0]),np.where(data.lon.values<=lon_out[-1])[0]])
       print(block_cols_in[0],block_cols_in[-1])
       block_rows_in = np.where(np.logical_and(data.lat.values>=lat_out[0],data.lat.values<=lat_out[-1]))[0]
       data = data.isel(lon=block_cols_in,lat=block_rows_in)
       print('regrid')
       #
       if itype in ['conservative']:
          ds_out    = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out),'lat_b':(['lat_b'], lat_b_out), 'lon_b': (['lon_b'], lon_b_out), })
          #
          lon_b = list(data.XG.values[block_cols_in])
          lon_b.append(data.XG.values[block_cols_in][-1]+0.1)
          lat_b = list(data.YG.values[block_rows_in])
          lat_b.append(data.YG.values[block_rows_in][-1]+0.1)
          data['lon_b'] = np.array(lon_b)
          data['lat_b'] = np.array(lat_b)
          mask = xr.DataArray(np.ones(data.TRAC01.shape[1:]),dims=('lat','lon'),coords={'lat': (['lat'], data.lat.values),'lon': (['lon'], data.lon.values)},name='mask')
          mask.data[np.where(data.TRAC01.isel(time=0).values==0)]=0
       else:
          ds_out      = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out),})
       #
       if c==0:
          regridder   = xe.Regridder(data, ds_out, itype, reuse_weights=False)
       else:
          regridder   = xe.Regridder(data, ds_out, itype, reuse_weights=True)
       ds_out      = regridder(data.TRAC01)
       print('pickup values')
       if itype in ['conservative']:
           mask_out=regridder(mask)
           mask_out.data[np.where(mask_out<0.5)]=0
           tr_out[c*dc//dt:c*dc//dt+data.time.shape[0],:,:]=ds_out.values/mask_out.values
       else:
           tr_out[c*dc//dt:c*dc//dt+data.time.shape[0],:,:]=ds_out.values
    else:
       print('select')
       data2=data.TRAC01.isel(XC=block_cols,YC=block_rows)
       print('pickup values')
       tr_out[count//dt-1:count//dt-1+data2.time.shape[0],:,:]=data2.values
    #
    del data
#
cluster = LocalCluster(n_workers=4)
client = Client(cluster)
#
data_dir  = '../data/raw/run_sst_AMSRE_damped_1m/'
savepath  = '../data/processed/'
# specific settings of the MITGCM output
dt     = 94
dt_max = 842240
dc     = dt*360
#
Dt_secs = 3600*23.5
#
skip_files = 90
count0     = 94 #daily data
#
data0=open_mdsdataset(data_dir,prefix=['tracer_snapshots'],iters=range(count0,count0+dt,dt), delta_t=900, ref_date='1993-01-01')
#
xx = data0.XC.values; #xx[np.where(xx>180)]=xx[np.where(xx>180)]-360
yy = data0.YC.values
if regrid:
   itype='conservative' #'bilinear'
   lat_out=np.arange(np.floor(np.min(yy))+dg/2, np.ceil(np.max(yy)), dg)
   lon_out=np.arange(np.floor(np.min(xx))+dg/2, np.ceil(np.max(xx)), dg)
   lat_b_out=np.arange(np.floor(np.min(yy)), np.ceil(np.max(yy))+dg/2, dg)
   lon_b_out=np.arange(np.floor(np.min(xx)), np.ceil(np.max(xx))+dg/2, dg)
   #
   ny = len(lat_out)
   nx = len(lon_out)
   data0['lat'] = yy
   data0['lon'] = xx
   lon_b=list(data0.XG.values)
   lon_b.append(data0.XG.values[-1]+0.1)
   lat_b=list(data0.YG.values)
   lat_b.append(data0.YG.values[-1]+0.1)
   data0['lon_b'] = np.array(lon_b)
   data0['lat_b'] = np.array(lat_b)
   ds_out = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out),'lat_b':(['lat_b'], lat_b_out), 'lon_b': (['lon_b'], lon_b_out), })
   regridder = xe.Regridder(data0, ds_out, itype, reuse_weights=True)
   data1=regridder(data0.TRAC01)
else:
   ny = len(yy)
   nx = len(xx)

if not monte_carlo:
    Kx = np.ones((len(taus),ny,nx))*np.nan
    Ky = np.ones((len(taus),ny,nx))*np.nan
    Kxy= np.ones((len(taus),ny,nx))*np.nan
    U  = np.ones((len(taus),ny,nx))*np.nan
    V  = np.ones((len(taus),ny,nx))*np.nan
    R  = np.ones((len(taus),ny,nx))*np.nan

#
if regrid:
   nn=2 #int(np.ceil(dg/0.1)+2)
   if dg<0.5:
     Partition_rows=4
     Partition_cols=4
   elif dg<1.0:
     Partition_rows=2
     Partition_cols=2
   elif dg<1.5:
     Partition_rows=2
     Partition_cols=2
   else:
     Partition_rows=1
     Partition_cols=1
else:
  Partition_rows=10
  Partition_cols=15
  nn=1

Block_row_size = int(np.ceil(1.0*ny/Partition_rows));
Block_col_size = int(np.ceil(1.0*nx/Partition_cols));
blknum = 0
for b_row in range(Partition_rows):
    rowStart = b_row*Block_row_size;
    for b_col in range(Partition_cols):
        colStart  = b_col*Block_col_size
        blknum=blknum+1;
        if dg==0.25 and blknum<8:
            print(dg, blknum)
            continue
        #
        print('calculating block '+str(blknum)+' of '+str(Partition_rows*Partition_cols)+' rows '+str(rowStart)+'-'+str(rowStart+Block_row_size)+ ' ,cols '+str(colStart)+'-'+str(colStart+Block_col_size))
        #
        block_rows      = np.arange(rowStart-nn-1,rowStart+Block_row_size+nn+2).astype('int') #need halo of 1 for 1 point stencil
        block_cols      = np.arange(colStart-nn-1,colStart+Block_col_size+nn+2).astype('int')
        block_rows[ma.where(block_rows<0)]    = 0
        block_rows[ma.where(block_rows>ny-1)] = ny-1
        block_cols[ma.where(block_cols<0)]    = block_cols[ma.where(block_cols<0)]+nx
        block_cols[ma.where(block_cols>nx-1)] = block_cols[ma.where(block_cols>nx-1)]-nx
        block_rows=np.unique(block_rows)
        #
        nt2 = dt_max//dt
        ny2 = len(block_rows)
        nx2 = len(block_cols)
        print('processing')
        #
        tr_out = np.zeros((nt2,ny2,nx2))
        lat_out = data1.lat[block_rows].values
        lon_out = data1.lon[block_cols].values
        lon_b_out = np.concatenate([lon_out-dg/2,lon_out[-1:]])
        lon_b_out[np.where(lon_b_out<0)]=lon_b_out[np.where(lon_b_out<0)]+360
        lat_b_out = np.concatenate([lat_out-dg/2,lat_out[-1:]])
        for c,count in enumerate(range(count0+dt*skip_files,dt_max,dc)):
            print(c)
            load_data(c,count,dc,Partition_cols,block_cols,block_rows,lon_out,lat_out,lon_b_out,lat_b_out,tr_out,dt=dt,dt_max=dt_max,regrid=regrid,dg=dg,itype=itype)
        #
        tr_out=tr_out[:-skip_files-1,:,:] #
        nt2=tr_out.shape[0]
        #
        if regrid:
            block_cols = block_cols[nn:-nn]
            block_rows = block_rows[nn:-nn]
            tr_out     = tr_out[:,nn:-nn,nn:-nn]
            xx2 = lon_out[nn:-nn]
            yy2 = lat_out[nn:-nn]
            ny2 = len(block_rows)
            nx2 = len(block_cols)
        else:
            xx2 = xx[block_cols]
            yy2 = yy[block_rows]
        #
        print('data loaded...')
        # PREPARE THE DATA FOR MICROINVERSE
        tr_mean  = np.mean(tr_out,0)
        jj,ii    = np.where(tr_mean==0) # set 0's to nans
        tr_mean[jj,ii] = np.nan
        trc_anom = np.ones((nt2,ny2,nx2))*np.nan
        jj,ii    = np.where(np.isfinite(tr_mean)) # detrend - no nan's allowed
        trc_anom[:,jj,ii] = detrend(tr_out[:,jj,ii]-tr_mean[jj,ii],axis=0)
        # 
        x_grid = np.swapaxes(np.swapaxes(trc_anom,0,2),0,1)
        #
        block_lon,block_lat = np.meshgrid(xx2,yy2)
        jind = block_rows[1:-1]
        iind = block_cols[1:-1]
        #
        iinds,jinds = np.meshgrid(iind,jind)
        jinds = jinds.flatten()
        iinds = iinds.flatten()
        # Loop over lags
        for tt,Tau in enumerate(taus):
            print('Calculating lag '+str(Tau))
            print(x_grid.shape, block_lon.shape, block_lat.shape, ny2, nx2)
            U_block,V_block,Kx_block,Ky_block,Kxy_block,Kyx_block,R_block,res_block = mutils.inversion(x_grid,block_rows,block_cols,block_lon,block_lat,nx2,ny2,nt2,Stencil_center,Stencil_size,Tau,Dt_secs,num_cores=num_cores, dt_min = dt_min, ens = ens)
            #
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
                Kx[tt,jinds,iinds]  = Kx_block.flatten()
                Ky[tt,jinds,iinds]  = Ky_block.flatten()
                Kxy[tt,jinds,iinds] = Kxy_block.flatten()
                U[tt,jinds,iinds]   = U_block.flatten()
                V[tt,jinds,iinds]   = V_block.flatten()
                R[tt,jinds,iinds]   = R_block.flatten()
        if monte_carlo:
            if not os.path.exists(savepath+'monte_carlo/'):
                os.makedirs(savepath+'monte_carlo/')
            if regrid:
               savefile = savepath+'monte_carlo/run_sst_AMSRE_damped_1m_global_'+str(dg)+'_full_'+itype+'_v2_monte_carlo_AvePeriod_'+str(dt_min//365)+'_block_'+str(blknum).zfill(3)+'.npz'
               np.savez(savepath,Kx=Kx,Ky=Ky,R=R,U=U,V=V,taus=taus,res=res,ens=ens,Lat_vector=lat_out,Lon_vector=lon_out,jinds=jinds,iinds=iinds,ny=ny,nx=nx)
            else:
               savefile = savepath+'monte_carlo/run_sst_AMSRE_damped_1m_global_full_monte_carlo_AvePeriod_'+str(dt_min//365)+'_block_'+str(blknum).zfill(3)+'.npz'
               np.savez(savepath,Kx=Kx,Ky=Ky,R=R,U=U,V=V,taus=taus,res=res,ens=ens,Lat_vector=lat_out,Lon_vector=lon_out,jinds=jinds,iinds=iinds,ny=ny,nx=nx)


print('saving...')
client.close()
cluster.close()
if not monte_carlo:
    if regrid:
        np.savez(savepath+'run_sst_AMSRE_damped_1m_global_'+str(dg)+'_full_'+itype+'_v2.npz',Kx=Kx,Ky=Ky,R=R,U=U,V=V,taus=taus)
    else:
        np.savez(savepath+'run_sst_AMSRE_damped_1m_global_full_v2.npz',Kx=Kx,Ky=Ky,R=R,U=U,V=V,taus=taus)

