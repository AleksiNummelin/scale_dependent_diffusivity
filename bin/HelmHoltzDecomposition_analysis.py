import xarray as xr
import numpy as np
from scipy import signal
from dask.distributed import Client, LocalCluster
import HelmHoltzDecomposition_analysis_utils as hutils

######################
# LOAD VELOCITY DATA
######################
#
######################
# SMOOTHING APPROACH
#
cluster = LocalCluster(n_workers=4)
client = Client(cluster)
#
for year in ['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012']:
    print(year)
    data=xr.open_mfdataset('../data/processed/surface_currents_'+year+'_*.nc')
    lon2,lat2=np.meshgrid(data.lon,data.lat)
    rlat2=np.radians(lat2)
    r=6371E3   
    #
    nt,nd,ny,nx=data.ul_out.shape
    #
    # filter sizes (grid cell widths)
    dgs0=np.array([3,5,7,9,11,13,15,19,23,27,31,39,51,59])   
    # filter sizes in degrees
    dgs=0.25*dgs0
    #
    # INITIALIZE VARIABLES
    U_s=np.zeros((len(dgs),3,ny,nx))
    V_s=np.zeros((len(dgs),3,ny,nx))
    U_l=np.zeros((len(dgs),3,ny,nx))
    V_l=np.zeros((len(dgs),3,ny,nx))  
    #
    KEs=np.zeros((len(dgs),3,ny,nx))
    KEsx=np.zeros((len(dgs),3,ny,nx))
    KEsy=np.zeros((len(dgs),3,ny,nx))
    KEl=np.zeros((len(dgs),3,ny,nx))
    KEtot=np.zeros((len(dgs),3,ny,nx))
    KEtriad=np.zeros((len(dgs),3,ny,nx))
    KE_transfer=np.zeros((len(dgs),3,ny,nx))
    #
    div=np.zeros((len(dgs),3,ny,nx))
    curl=np.zeros((len(dgs),3,ny,nx))
    shear=np.zeros((len(dgs),3,ny,nx))
    strain=np.zeros((len(dgs),3,ny,nx))
    #
    Kxm=np.zeros((len(dgs),3,ny,nx))
    Kym=np.zeros((len(dgs),3,ny,nx))
    Kxym=np.zeros((len(dgs),3,ny,nx))
    #
    nRu_ll = np.zeros((len(dgs),nd,ny,nx))
    nD_ll  = np.zeros((len(dgs),nd,ny,nx))
    nRu_ss = np.zeros((len(dgs),nd,ny,nx))
    nD_ss  = np.zeros((len(dgs),nd,ny,nx))
    #
    mask=1-np.isnan(data.nRu_out.values[0,:,:])
    jinds,iinds=np.where(np.isnan(data.nRu_out.values[0,:,:]))
    #
    # RECONSTRUCT VELOCITIES
    #
    r0 = hutils.gradient(data.nRu_out, lat2, lon2,[0.25,0.25], r=6371E3, rotated=True, glob=True)
    d0 = hutils.gradient(data.nD_out, lat2, lon2,[0.25,0.25], r=6371E3, rotated=False, glob=True)
    u0 = -(r0+d0)[:,:,:,0]
    v0 = -(r0+d0)[:,:,:,1]
    KE0 = np.percentile(np.sqrt(u0**2+v0**2),[25,50,75],axis=0)
    #
    # USE G
    for d,dg in enumerate(dgs0):
       #
       print(d,dg*0.25)
       win = gaussian(dg,dg, cutoff=2, mfac=4*2*np.log(2)) #gaussian filter which seems to most adequately capture the effect 
       #
       mask2 = signal.convolve2d(mask*np.cos(rlat2), win, mode='same', boundary='wrap')
       for t,tt in enumerate(data.time2.values):
           print(t)
           nRu_smooth = signal.convolve2d(data.nRu_out.isel(time2=t).fillna(0).values*np.cos(rlat2), win, mode='same', boundary='wrap')
           nD_smooth  = signal.convolve2d(data.nD_out.isel(time2=t).fillna(0).values*np.cos(rlat2), win, mode='same', boundary='wrap')
           nRu_ll[d,t,:,:] = nRu_smooth/mask2
           nD_ll[d,t,:,:]  = nD_smooth/mask2
       #
       nRu_ll[d,:,jinds,iinds] = 0
       nD_ll[d,:,jinds,iinds]  = 0
       #
       r_ll = hutils.gradient(nRu_ll[d,:,:,:], lat2, lon2,[0.25,0.25], r=6371E3, rotated=True, glob=True)
       d_ll = hutills.gradient(nD_ll[d,:,:,:], lat2, lon2,[0.25,0.25], r=6371E3, rotated=False, glob=True)
       # Lage scale velocity components
       u_ll = -(r_ll+d_ll)[:,:,:,0]
       v_ll = -(r_ll+d_ll)[:,:,:,1]
       # Small scale velocity components
       u_ss = u0 - u_ll
       v_ss = v0 - v_ll 
       # 25-50-75 \% percentiles (in time)
       U_s[d,:,:,:]=np.percentile(u_ss,[25,50,75],axis=0)
       V_s[d,:,:,:]=np.percentile(v_ss,[25,50,75],axis=0)
       U_l[d,:,:,:]=np.percentile(u_ll,[25,50,75],axis=0)
       V_l[d,:,:,:]=np.percentile(v_ll,[25,50,75],axis=0)
       #
       # we remove the time mean to get rid of the advective part 
       KEs[d,:,:,:]  = np.percentile(np.sqrt((u_ss-np.nanmedian(u_ss,0))**2+(v_ss-np.nanmedian(v_ss,0))**2),[25,50,75],axis=0)
       KEsx[d,:,:,:] = np.percentile(np.sqrt((u_ss-np.nanmedian(u_ss,0))**2),[25,50,75],axis=0) #
       KEsy[d,:,:,:] = np.percentile(np.sqrt((v_ss-np.nanmedian(v_ss,0))**2),[25,50,75],axis=0) #
       KEl[d,:,:,:]  = np.percentile(np.sqrt(u_ll**2+v_ll**2),[25,50,75],axis=0)
       #
       KEtot[d,:,:,:] = np.percentile(np.sqrt((u_ll+u_ss)**2+(v_ll+v_ss)**2),[25,50,75],axis=0)
       KEtriad[d,:,:,:] = np.percentile(-2*(u_ll*u_ss+v_ll*v_ss),[25,50,75],axis=0) #this is the nonlinear interaction term
       #
       #note that KEtot-KEs-KEl gives the nonlinear interaction term -2*(u_ll*u_ss+v_ll*v_ss)
       #
       # 
       di,cu,sh,st     = hutils.divcurl_shearstrain(np.stack([u_ll, v_ll],axis=-1),lat2,lon2,[0.25,0.25],r=6371E3)
       div[d,:,:,:]    = np.percentile(di,[25,50,75],axis=0)
       curl[d,:,:,:]   = np.percentile(cu,[25,50,75],axis=0)
       shear[d,:,:,:]  = np.percentile(sh,[25,50,75],axis=0)
       strain[d,:,:,:] = np.percentile(st,[25,50,75],axis=0)
       #
       shearstrain = np.sqrt(sh**2+st**2)
       p = shearstrain+st
       q = shearstrain-st
       delta = di/shearstrain
       # THIS IS THE LeSommer et al. 2014 parameterization for shear driven mixing
       # NOTE THAT WE WILL NOTE MULTIPLY WITH THE LENGTH SCALE HERE - WE WILL DETERMINE THAT LATER
       Kxm[d,:,:,:]  = np.percentile(0.5*(1+delta)*p,[25,50,75],axis=0)
       Kym[d,:,:,:]  = np.percentile(0.5*(1+delta)*q,[25,50,75],axis=0)
       Kxym[d,:,:,:] = np.percentile(0.5*(1+delta)*sh,[25,50,75],axis=0)
    
    #
    np.savez('../data/processed/mean_currents_'+year+'_v4.npz',u0=u0,v0=v0,U_s=U_s,V_s=V_s,U_l=U_l,V_l=V_l,KEs=KEs,KEsx=KEsx,KEsy=KEsy,KEl=KEl,KEtot=KEtot,KEtriad=KEtriad,dgs=dgs)
    #
    np.savez('../data/processed/LargeScaleMixing_'+year+'_v4.npz',Kxm=Kxm,Kym=Kym,Kxym=Kxym,div=div,curl=curl,shear=shear,strain=strain,mask=mask,dgs=dgs)

client.close()
cluster.close()
