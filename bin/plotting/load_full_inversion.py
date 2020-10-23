########################################################
#
# AUTHOR: ALEKSI NUMMELIN
# YEAR:   2018-2020
#
# THIS SCRIPT WILL LOAD THE INVERSION THAT USED THE FULL
# LENGTH TIMESERIES AND POSTPROCESS THE RESULTS
# COMBINING THE SOLUTIONS ACROSS DIFFERENT TAUS
#
#########################################################
#
#DEFINE SOME FUNCTIONS
#
def load_inversion_data(main_path,taus=[1,2,3,4,6,8,10],ext=''):
    ''' '''
    data={}
    for key in ['Kx','Ky','U','V','R','Lat_vector','Lon_vector']:
       if key not in ['Lat_vector','Lon_vector']:
          data[key]=np.zeros(((len(taus),600,1440)))
       for t,tau in enumerate(taus):
           dum=np.load(main_path+str(tau)+ext+'.npz')
           if key in ['Lat_vector','Lon_vector'] and t==0:
              data[key]=dum[key][:].copy()
           elif key not in ['Lat_vector','Lon_vector']:
              data[key][t,:,:]=dum[key+'_global'][:].copy()
    ##
    weight_coslat=np.tile(np.cos(np.pi*np.reshape(data['Lat_vector'],(len(data['Lat_vector']),1))/180.),(1,len(data['Lon_vector'])))
    data2=mutils.combine_Taus(data,weight_coslat,taus,K_lim=True,dx=None,dy=111E3*0.25,timeStep=24*3600)
    for key in ['Lat_vector','Lon_vector']:
      data2[key] = data[key]
      if key in ['Lon_vector']:
         data2[key][np.where(data2[key]>180)]=data2[key][np.where(data2[key]>180)]-360
    for key in ['Kx','Ky','U','V','R']:
        data2[key][np.where(data2[key]==0)]=np.nan
    for key in ['Kx','Ky']:
        data2[key][np.where(data2[key]<0)]=np.nan
    #
    return data2

def calc_slopes(nx,j,dgs,lat,keys,data_out,slopes,interps,theil=True,alpha=0.95):
    ''' '''
    if j%50==0:
       print(j)
    for kk,K in enumerate(keys): #(['Kx','Ky']):
        for i in range(nx):
          if len(np.where(data_out[kk,:,j,i]>0)[0])>2: #np.isfinite(data_out[kk,0,j,i]): 
            if K in ['Ky']:
                if theil:
                    y=data_out[kk,:,j,i]
                    res=stats.theilslopes(np.log10(y[np.where(y>0)]),np.log10(111E3*np.array(dgs)[np.where(y>0)]),alpha=alpha)
                else:
                    res=monte_carlo_slopes(np.log10(data_out[kk,:len(dgs),j,i]),np.log10(111E3*np.array(dgs)))
            else:
                if theil:
                    y=data_out[kk,:,j,i]
                    res=stats.theilslopes(np.log10(y[np.where(y>0)]),np.log10(111E3*np.array(dgs)[np.where(y>0)]*np.cos(lat[j]*np.pi/180)),alpha=alpha)
                else:
                    res=monte_carlo_slopes(np.log10(data_out[kk,:len(dgs),j,i]),np.log10(111E3*np.array(dgs)*np.cos(lat[j]*np.pi/180)))
            #
            slopes[kk,:,j,i]=np.array([res[3],res[0],res[2]])
            interps[kk,j,i]=res[1]

def wrap_regrid(dg_in,datain,ds_in=None,lat_range=np.array([-70,80]),lon_range=np.array([-180,180]),dg_out=1/3,imethod='conservative', reuse_weights=True):
    ''' '''
    lat_out    = np.arange( -90+dg_out/2, 90, dg_out)
    lon_out    = np.arange(0+dg_out/2,360, dg_out)
    lat_b_out  = np.arange( -90, 90+dg_out/2, dg_out)
    lon_b_out  = np.arange(0, 360+dg_out/2, dg_out)
    lon_out[np.where(lon_out>180)]=lon_out[np.where(lon_out>180)]-360
    lon_b_out[np.where(lon_b_out>180)]=lon_b_out[np.where(lon_b_out>180)]-360   
    #
    ds_out     = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out),'lat_b':(['lat_b'], lat_b_out), 'lon_b': (['lon_b'], lon_b_out), })
    #
    # grid
    if ds_in==None: #imethod in ['conservative'] and ds_in==None:
       #
       lat_in     = np.arange(lat_range.min()+dg_in/2, lat_range.max()        , dg_in)
       lat_b_in   = np.arange(lat_range.min()        , lat_range.max()+dg_in/2, dg_in)
       #
       lon_in     = np.arange(lon_range.min()+dg_in/2, lon_range.max()        , dg_in)
       lon_b_in   = np.arange(lon_range.min()        , lon_range.max()+dg_in/2, dg_in)
       #
       lon_in[np.where(lon_in>180)] = lon_in[np.where(lon_in>180)]-360
       lon_b_in[np.where(lon_b_in>180)] = lon_b_in[np.where(lon_b_in>180)]-360
       #
       ds_in      = xr.Dataset({'lat': (['lat'], lat_in),'lon': (['lon'], lon_in),'lat_b':(['lat_b'], lat_b_in), 'lon_b': (['lon_b'], lon_b_in), })
    #
    regridder = xe.Regridder(ds_in,ds_out,imethod,reuse_weights=reuse_weights)
    dum=datain[:].copy()
    dum[np.where(np.isnan(dum))]=0
    mask=np.ones(dum.shape)
    mask[np.where(dum==0)]=0
    k_out = regridder(dum)
    mask  = regridder(mask)
    #
    return k_out/mask, ds_out

# ######################################
# LOAD DIFFUSIVITIES
#
data0_all={}
data_all_out={}
slopes_all={}
interps_all={}
dg_out=2.0
export_disk = '../data/posprocess/'
for var in ['sst','sst_model']:
    if var in ['sst','ssh']:
       dgs=[0.25,0.3,0.4,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0]
       data_all_out[var+'_dgs']=dgs
       main_path0='../../data/processed/uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_Tau'
       #
       data0_all['data_sst_avhrr'] = load_inversion_data(main_path0,[2,3,4,5,6,7,8,10])
       #
       data=np.load('/uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_1993_2016.npz')
       weight_coslat=np.tile(np.cos(np.pi*np.reshape(data['Lat_vector'],(len(data['Lat_vector']),1))/180.),(1,len(data['Lon_vector'])))
       data_sst=mutils.combine_Taus(data,weight_coslat,data['taus'],K_lim=True,dx=None,dy=111E3*0.25,timeStep=24*3600)
       data_sst['Lat_vector']=data['Lat_vector']
       data_sst['Lon_vector']=data['Lon_vector']
       data0_all['data_sst_avhrr_1993_2016']=data_sst
       #
       #dg_out=2.0 # 1./3 #moves up
       lat_out    = np.arange( -90+dg_out/2, 90, dg_out)
       lon_out    = np.arange(0+dg_out/2,360, dg_out)
       data_out   = np.zeros((3,len(dgs),len(lat_out),len(lon_out)))
       #
       #for kk,K in enumerate(['Kx','Ky']):
       for d,dg in enumerate(dgs):
            if dg>dg_out:
               imethod='bilinear'
            else:
               imethod='conservative'
            if dg!=0.25:
                if var in ['sst']:
                   data   = np.load(export_disk+'uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_coarse_ave_'+str(dg)+'_conservative.npz')
                   lat_in = np.arange( -90+dg/2, 90, dg)
                   lon_in = np.arange(0+dg/2, 360, dg)
                   lon_in[np.where(lon_in>180)]=lon_in[np.where(lon_in>180)]-360
                #
                weight_coslat=np.tile(np.cos(np.pi*np.reshape(lat_in,(len(lat_in),1))/180.),(1,len(lon_in)))
                datain=mutils.combine_Taus(data, weight_coslat, data["taus"], K_lim=True, dx=None, dy=111E3*dg, timeStep=24*3600)
                lat_range=np.array([-90,90])
            else:
                datain = load_inversion_data(main_path0,[2,3,4,5,6,7,8,10])
            #
            for kk, K in enumerate(['Kx','Ky','Ktot']):
                if kk==0:
                    reuse_weights=True
                else:
                    reuse_weights=True
                #
                if K in ['Kx','Ky','Kxy']:
                    k_out,ds_out=wrap_regrid(dg,datain[K],lat_range=lat_range,lon_range=np.array([0,360]),dg_out=dg_out,imethod=imethod,reuse_weights=reuse_weights)
                elif K in ['Ktot']:
                    if frobenius:
                        dum = np.min([np.sign(datain['Kx']),np.sign(datain['Ky'])],axis=0)*np.sqrt(datain['Kx']**2+datain['Ky']**2) #
                    else:
                        dum = np.min([np.sign(datain['Kx']),np.sign(datain['Ky'])],axis=0)*np.sqrt(datain['Kx']*datain['Ky'])
                    #
                    k_out,ds_out=wrap_regrid(dg,dum,lat_range=lat_range,lon_range=np.array([0,360]),dg_out=dg_out,imethod=imethod,reuse_weights=reuse_weights)
                #
                data_out[kk,d,:,:]=k_out
       #
       data_all_out[var]=data_out
       data_all_out[var+'ds_out']=ds_out
       #
    elif var in ['sst_model']:
       dgs=[0.25,0.3,0.4,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
       data_all={}
       #
       data0=open_mdsdataset(export_disk+'run_sst_AMSRE_damped_1m/',prefix=['tracer_snapshots'],iters=range(94,94*2,94), delta_t=900, ref_date='1993-01-01')
       xx0 = data0.XC.values
       yy0 = data0.YC.values
       #
       for j,dg in enumerate(dgs):
          if dg==0.1:
            data_all[str(dg)]=np.load(export_disk+'run_sst_AMSRE_damped_1m_global_full_v2.npz')
            data_all['xx'+str(dg)]=xx0
            data_all['yy'+str(dg)]=yy0
          else:
            data_all[str(dg)]=np.load(export_disk+'run_sst_AMSRE_damped_1m_global_'+str(dg)+'_full_conservative_v2.npz')
            data_all['xx'+str(dg)]=np.arange(np.floor(np.min(xx0))+dg/2, np.ceil(np.max(xx0)), dg)
            data_all['yy'+str(dg)]=np.arange(np.floor(np.min(yy0))+dg/2, np.ceil(np.max(yy0)), dg)
       #
       data1={}; data2={}; data3={};  data4={};  data5={};  data6={};
       data7={}; data8={}; data9={}; data10={}; data11={}; data12={};
       data13={}; data14={}; data15={}; data16={}; data17={}
       for j,dg in enumerate(dgs):
           for key in ['Kx','Ky','U','V','R']:
               exec('data'+str(j+1)+'["'+key+'"]=data_all["'+str(dg)+'"]["'+key+'"][:]')
       #
       for j,dg in enumerate(dgs):
           print(j,dg)
           xx = data_all['xx'+str(dg)][:].copy()
           yy = data_all['yy'+str(dg)][:].copy()
           weight_coslat=np.tile(np.cos(np.pi*np.reshape(yy,(len(yy),1))/180.),(1,len(xx)))
           exec('data'+str(j+1)+' = combine_Taus(data'+str(j+1)+',weight_coslat,data_all["'+str(dg)+'"]["taus"],K_lim=True,dx=None,dy=111E3*dg,timeStep=23.5*3600)')
           exec('data'+str(j+1)+'["taus"] = data_all["'+str(dg)+'"]["taus"]')
           exec('data'+str(j+1)+'["xx"] = xx')
           exec('data'+str(j+1)+'["yy"] = yy')
       #
       lat_out    = np.arange( -90+dg_out/2, 90, dg_out)
       lon_out    = np.arange(0+dg_out/2,360, dg_out)
       data_out   = np.zeros((3,len(dgs),len(lat_out),len(lon_out)))
       data0_all['sst_model'] = data1
       #
       for kk,K in enumerate(['Kx','Ky','Ktot']):
           #
           for d,data in enumerate([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15]):
              print(d,K)
              if dgs[d]>dg_out:
                  imethod='bilinear'
              else:
                  imethod='conservative'
              if kk==0:
                reuse_weights = True
              else:
                reuse_weights = True
              #
              if K in ['Kx','Ky','Kxy']:
                  k_out,ds_out = wrap_regrid(dgs[d],data[K],lat_range=np.array([data['yy'].min()-dgs[d]/2,data['yy'].max()+dgs[d]/2]),lon_range=np.array([0,360]),dg_out=dg_out,imethod=imethod,reuse_weights=reuse_weights)
              elif K in ['Ktot']:
                  if frobenius:
                      dum = np.min([np.sign(data['Kx']),np.sign(data['Ky'])],axis=0)*np.sqrt(data['Kx']**2+data['Ky']**2) #
                  else:
                      dum = np.min([np.sign(data['Kx']),np.sign(data['Ky'])],axis=0)*np.sqrt(data['Kx']*data['Ky']) #
                  k_out,ds_out = wrap_regrid(dgs[d],dum,lat_range=np.array([data['yy'].min()-dgs[d]/2,data['yy'].max()+dgs[d]/2]),lon_range=np.array([0,360]),dg_out=dg_out,imethod=imethod,reuse_weights=reuse_weights)
              #
              data_out[kk,d,:,:] = k_out
       #
       data_all_out[var]          = data_out
       data_all_out[var+'ds_out'] = ds_out
       data_all_out[var+'_dgs']   = dgs
    #
    # ##############################################
    #  CALCULATE DIFFUSIVITY SLOPES
    #
    folder1     = tempfile.mkdtemp()
    path1       = os.path.join(folder1, 'dum1.mmap')
    path2       = os.path.join(folder1, 'dum2.mmap')
    path3       = os.path.join(folder1, 'dum3.mmap')
    path4       = os.path.join(folder1, 'dum4.mmap')
    slopes      = np.memmap(path1, dtype=float, shape=((3,)+(3,)+data_all_out[var].shape[2:]), mode='w+')
    interps     = np.memmap(path2, dtype=float, shape=((3,)+data_all_out[var].shape[2:]), mode='w+')
    data_out_mm = np.memmap(path3, dtype=float, shape=(data_all_out[var].shape), mode='w+')
    lat         = np.memmap(path4, dtype=float, shape=(ds_out.lat.values.shape), mode='w+')
    #
    slopes[:]=np.ones((3,)+(3,)+data_all_out[var].shape[2:])*np.nan
    interps[:]=np.ones((3,)+data_all_out[var].shape[2:])*np.nan
    data_out_mm[:]=data_all_out[var]
    lat[:]=ds_out.lat.values
    ny,nx=data_all_out[var].shape[2:]
    num_cores=15
    i1deg=np.where(np.logical_and(np.array(dgs)>=0.25,np.array(dgs)<=1.0))[0] #[-1]+1 #use everything between 0.25-1 deg
    jinds,iinds = np.where(np.nansum(np.nansum(data_out_mm[:,i1deg[0]:i1deg[-1]+1,:,:],0),0)!=0)
    n=4
    #
    Parallel(n_jobs=num_cores)(delayed(calc_slopes)(nx,j,2*np.array(dgs)[i1deg[0]:i1deg[-1]+1],lat,['Kx','Ky','Ktot'],data_out_mm[:,i1deg[0]:i1deg[-1]+1,:,:],slopes,interps,alpha=0.9) for j in range(ny))
    slopes=np.array(slopes)
    interps=np.array(interps)
    slopes_all[var]=slopes
    interps_all[var]=interps
    try:
       shutil.rmtree(folder1)
    except OSError:
       pass
    #
    print(var+' DONE!') 



