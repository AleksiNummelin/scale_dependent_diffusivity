###########################
#
# AUTHOR: ALEKSI NUMMELIN
# YEAR:   2018-2020
#
# THIS SCRIPT WILL PREPROCESS
# THE DATA FOR MAKING THE FINAL
# FIGURES FOR NUMMELIN ET AL. 2020.
#
print('Calculate Slopes!')
dg_in=0.25
#
lat_in     = lat2[:,0]
lat_b_in   = np.concatenate([lat2[:,0]-dg_in/2,lat2[-1:,0]+dg_in/2],axis=0)
lon_in     = lon2[0,:]
lon_b_in   = np.concatenate([lon2[0,:]-dg_in/2,lon2[0,-1:]+dg_in/2],axis=0)
#
lon_in[np.where(lon_in>180)] = lon_in[np.where(lon_in>180)]-360
lon_b_in[np.where(lon_b_in>180)] = lon_b_in[np.where(lon_b_in>180)]-360
ds_in      = xr.Dataset({'lat': (['lat'], lat_in),'lon': (['lon'], lon_in),'lat_b':(['lat_b'], lat_b_in), 'lon_b': (['lon_b'], lon_b_in), })
#
# set output grids
fpath='/home/anummel1/Projects/MicroInv/xesmf_weights/'
#
ds_out_2deg=hutils.return_grid(2.0, llon=0,ulon=360)
regridder_2deg = xe.Regridder(ds_in,ds_out_2deg,'conservative',filename=fpath+'conservative_from_'+str(dg_in)+'_to_'+str(2.0)+'_HHscaling.nc',reuse_weights=True)
mask3=np.round(regridder_2deg(mask2))
#
ny2,nx2 = ds_out_2deg.lat.shape[0], ds_out_2deg.lon.shape[0]
data_out_ll = np.zeros((3,len(dgs0),ny2,nx2))
data_out_ll2 = np.zeros((3,len(dgs0),ny2,nx2))
data_out_ss = np.zeros((3,len(dgs0),ny2,nx2))
data_out_ss2 = np.zeros((3,len(dgs0),ny2,nx2))
#
for nn,data_out0 in enumerate([Kxm[:,1,:,:],Kym[:,1,:,:],Km_tot[:,:,:]]):
    data_out_ll[nn,:,:,:]=regridder_2deg(data_out0)
 
for nn,data_out0 in enumerate([Kxm5[:,:,:],Kym5[:,:,:],Km_tot5[:,:,:]]):
    data_out_ll2[nn,:,:,:]=regridder_2deg(data_out0)
#
for nn,data_out0 in enumerate([k30x,k30y,k30]): #([Kx_MixLen2,Ky_MixLen2,Kxy_MixLen2]):
    data_out_ss[nn,:,:,:]=regridder_2deg(data_out0)

print('error metrics!')
# #################################
# THEIL-SEN SLOPE ESTIMATE
#
data_out_ss2=data_out_ss
lon3,lat3=np.meshgrid(ds_out_2deg.lon,ds_out_2deg.lat)
ki='theil'
log10range=np.concatenate([np.arange(0,10),np.arange(10,100,10),np.arange(100,1000,100)],axis=0)/1E3
#
aranges={}
branges={}
aranges['0.25_0']=log10range
aranges['0.75_0']=log10range
aranges['1.25_0']=log10range
aranges['2.25_0']=log10range
#
branges['0.25_0']=log10range
branges['0.75_0']=log10range
branges['1.25_0']=log10range
branges['2.25_0']=log10range
#
aranges['0.25_1']=log10range
aranges['0.75_1']=log10range
aranges['1.25_1']=log10range
aranges['2.25_1']=log10range
#
branges['0.25_1']=log10range
branges['0.75_1']=log10range
branges['1.25_1']=log10range
branges['2.25_1']=log10range
#
jinds_tropics = np.where(abs(lat3[:,0])<=15)[0]
jinds_midlats = np.where(np.logical_and(abs(lat3[:,0])>=15,abs(lat3[:,0])<=60))[0]
#
jjs=[0.25,1.25]
abest0=np.ones((2,2,2,len(jjs),3))
bbest0=np.ones((2,2,2,len(jjs),3))
err0={}
std0={}
interp0={}
abest_x0=np.nan; bbest_x0=np.nan
abest_y0=np.nan; bbest_y0=np.nan
for jnd,jinds in enumerate([jinds_tropics,jinds_midlats]):
    for l,ll in enumerate([data_out_ll,data_out_ll2]):
        for j,jj in enumerate(jjs):
            a_range=aranges[str(jj)+'_'+str(l)]
            b_range=branges[str(jj)+'_'+str(l)]
            num_c=np.min([18,len(b_range)])
            j11=np.where(jj==np.array(data_ens['sst_model_10']['dgs']))[0][0]
            j12=np.where(jj==np.array(data_ens['sst_10']['dgs']))[0][0]
            j2=np.where(jj*3==dgs0*0.25)[0][0]
            print(j,j11,j12,j2)
            dum1 = np.nanmedian(data_ens['sst_model_10']['K_out'][j11,],axis=0)
            dum2 = np.nanmedian(data_ens['sst_10']['K_out'][j12,],axis=0)
            dum1[np.where(data_ens['sst_model_10_StoN'][j11,]<3)]=np.nan
            dum2[np.where(data_ens['sst_10_StoN'][j12,]<3)]=np.nan
            #
            for l2,dum in enumerate([dum1,dum2]):
                abest_tot0, bbest_tot0, err_tot0, std_tot0, interp_tot0 = hutils.optimize_mixing(dum[jinds,:], ll[2,j2,jinds,:], data_out_ss[2,j2,jinds,:],lat3[jinds,], a_range, b_range,ki=ki,num_cores=num_c)
                #
                abest0[jnd,l2,l,j,:] = np.array([abest_x0,abest_y0,abest_tot0])
                bbest0[jnd,l2,l,j,:] = np.array([bbest_x0,bbest_y0,bbest_tot0])
                #
                err0[str(jj)+'_'+str(l)+'_'+str(l2)+'_'+str(jnd)] = err_tot0
                std0[str(jj)+'_'+str(l)+'_'+str(l2)+'_'+str(jnd)] = std_tot0
                interp0[str(jj)+'_'+str(l)+'_'+str(l2)+'_'+str(jnd)] = interp_tot0
#
#################################
# MAXIMUM PERCENTAGE ERROR
#
aranges2=aranges
branges2=branges
#
err1={}
abest1=np.ones((2,2,2,len(jjs),3))
bbest1=np.ones((2,2,2,len(jjs),3))
for jnd,jinds in enumerate([jinds_tropics,jinds_midlats]):
    for l,ll in enumerate([data_out_ll,data_out_ll2]):
        for j,jj in enumerate(jjs):
            a_range=aranges2[str(jj)+'_'+str(l)]
            b_range=branges2[str(jj)+'_'+str(l)]
            num_c=np.min([18,len(b_range)])
            #j1=np.where(jj==np.array(data_all_out['sst_model_dgs']))[0][0]
            j11=np.where(jj==np.array(data_ens['sst_model_10']['dgs']))[0][0]
            j12=np.where(jj==np.array(data_ens['sst_10']['dgs']))[0][0]
            j2=np.where(jj*3==dgs0*0.25)[0][0]
            print(j,j11,j12,j2)
            #
            dum1 = np.nanmedian(data_ens['sst_model_10']['K_out'][j11,],axis=0)
            dum2 = np.nanmedian(data_ens['sst_10']['K_out'][j12,],axis=0)
            dum1[np.where(data_ens['sst_model_10_StoN'][j11,]<3)]=np.nan
            dum2[np.where(data_ens['sst_10_StoN'][j12,]<3)]=np.nan
            #
            for l2,dum in enumerate([dum1,dum2]):
                abest_tot1, bbest_tot1, err_tot1, std_tot1, interp_tot1 = hutils.optimize_mixing(dum[jinds,:], ll[2,j2,jinds,:], data_out_ss[2,j2,jinds,:],lat3[jinds,], a_range, b_range,ki='mpe',num_cores=num_c)
                abest1[jnd,l2,l,j,:] = np.array([abest_x0,abest_y0,abest_tot1])
                bbest1[jnd,l2,l,j,:] = np.array([bbest_x0,bbest_y0,bbest_tot1])
                #
                err1[str(jj)+'_'+str(l)+'_'+str(l2)+'_'+str(jnd)] = err_tot1

############################################
# PARAMETERIZATION SLOPES
slopes_nHHd = {}
interps_nHHd = {}
keys = ['LargeScale','LargeScale2', 'SmallScale','SmallScale2']
#
for nn,data_out in enumerate([data_out_ll,data_out_ll2,data_out_ss]):
    #
    folder1     = tempfile.mkdtemp()
    path1       = os.path.join(folder1, 'dum1.mmap')
    path2       = os.path.join(folder1, 'dum2.mmap')
    path3       = os.path.join(folder1, 'dum3.mmap')
    path4       = os.path.join(folder1, 'dum4.mmap')
    slopes_mm   = np.memmap(path1, dtype=np.float, shape=((3,)+(3,)+(ny2,nx2)), mode='w+')
    interps_mm  = np.memmap(path2, dtype=np.float, shape=((3,)+(ny2,nx2)), mode='w+')
    data_out_mm = np.memmap(path3, dtype=np.float, shape=(data_out.shape), mode='w+')
    lat_mm      = np.memmap(path4, dtype=np.float, shape=(ny2), mode='w+')
    #
    slopes_mm[:]=np.ones((3,)+(3,)+(ny2,nx2))*np.nan
    interps_mm[:]=np.ones((3,)+(ny2,nx2))*np.nan
    data_out_mm[:]=data_out.copy()
    lat_mm[:]=ds_out_2deg.lat.values #ds_in.lat.values
    num_cores=15
    n0=0
    n=5
    #
    comps=['x','y','xy']
    Parallel(n_jobs=num_cores)(delayed(hutils.calc_slopes)(comps,nx2,j,dgs0[n0:n]*0.25,lat_mm,data_out_mm[:,n0:n,:,:],slopes_mm,interps_mm,alpha=0.5) for j in range(ny2))
    slopes_nHHd[keys[nn]]=np.array(slopes_mm).squeeze()
    interps_nHHd[keys[nn]]=np.array(interps_mm).squeeze()
    try:
         shutil.rmtree(folder1)
     except OSError:
         pass



