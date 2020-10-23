import sys
#sys.path.append('/home/anummel1/Projects/MicroInv/naturalHHD/pynhhd-v1.1/')
#sys.path.append('/home/anummel1/Projects/MicroInv/naturalHHD/pynhhd-v1.1/pynhhd/')
#from pynhhd import nHHD
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial,signal,stats,interpolate,integrate
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from sklearn import linear_model
import xesmf as xe
#import numpy
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import os
#from dask.distributed import Client, LocalCluster
sys.path.append('/home/anummel1/Projects/MicroInv/MicroInverse/')
from MicroInverse import MicroInverse_utils as mutils
import bottleneck as bn
#
#from matplotlib.colors import from_levels_and_colors
#import matplotlib as mpl
#import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#from cartopy.util import add_cyclic_point
#import string

def efunc(x,a,b,c,d):
    '''exponential fit'''
    return a*np.exp(b*x)+c*x+d

def func(x, a, b, c,d):
    return a * np.exp(-b * x) + c*x +d

def func_grad(x, a, b, c):
    return c-a*b*np.exp(-b*x)

def func_grad2(x,a,b,c,r,d):
    L0 = ((np.radians(x)*r)**2)*np.cos(d)
    return -L0*(c-a*b*np.exp(-b*x))

def gaussian(x_fwhm, y_fwhm, cutoff=2, amplitude=1, theta=0, mfac=4*2*np.log(2)):
    """
    Two dimensional Gaussian function - here defined in terms of
    full width half maximum (i.e not std), otherwise we are
    following astropy.
    # mfac = 4*2*np.log(2) # full width, half maximum
    # mfact = 0.5          # std
    #
    # although cutoff of 3 is usually suggested, here cutoff of 2 seems to be adequate
    See astropy.modeling.functional_models - gaussian
    """
    #
    x = np.arange(0, x_fwhm*cutoff+1, 1, float)
    y = np.arange(0, y_fwhm*cutoff+1, 1, float)
    x,y = np.meshgrid(x,y)
    #
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    x_fwhm2 = x_fwhm**2
    y_fwhm2 = y_fwhm**2
    xdiff = x - x_fwhm*cutoff//2
    ydiff = y - y_fwhm*cutoff//2
    a = mfac * ((cost2 / x_fwhm2) + (sint2 / y_fwhm2))
    b = mfac * ((sin2t / x_fwhm2) - (sin2t / y_fwhm2))
    c = mfac * ((sint2 / x_fwhm2) + (cost2 / y_fwhm2))
    #
    return amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)))
                                                                            

def spatial_filter(datain, std=2, mode='wrap'):
    '''
    smooth spatial filter that takes care of the land boundaries
    '''
    mask = np.isfinite(datain)
    dum = datain.copy()
    dum[np.where(1-mask)] = 0
    dum = gaussian_filter(dum,std,mode=mode)/gaussian_filter(mask.astype(np.float),std,mode=mode)
    dum[np.where(1-mask)] = np.nan
    dum[np.where(dum==0)] = np.nan
    
    return dum

def divcurl_shearstrain(vfield,lat,lon,dx,r=6371E3):
    '''
    Calculate divergence, curl, shear (strain), (normal) strain on a spherical surface
    We assume (*,*,ny,nx) i.e. lat,lon have to be the two last dimensions
    
    '''
    dudy = np.gradient(vfield[:,:,:,0]*np.cos(np.radians(lat)), np.radians(dx[0]), axis=-2)/(r*np.cos(np.radians(lat)))
    dudx = np.gradient(vfield[:,:,:,0], np.radians(dx[1]), axis=-1)/(r*np.cos(np.radians(lat)))
    dudx[:,:,0] = np.gradient(np.concatenate([vfield[:,:,-1:,0],vfield[:,:,:2,0]],axis=-1), np.radians(dx[1]), axis=-1)[:,:,1]/(r*np.cos(np.radians(lat[:,0])))
    dudx[:,:,-1] = np.gradient(np.concatenate([vfield[:,:,-2:,0],vfield[:,:,:1,0]],axis=-1), np.radians(dx[1]), axis=-1)[:,:,1]/(r*np.cos(np.radians(lat[:,0])))
    #
    dvdy = np.gradient(vfield[:,:,:,1]*np.cos(np.radians(lat)), np.radians(dx[0]), axis=-2)/(r*np.cos(np.radians(lat)))
    dvdx = np.gradient(vfield[:,:,:,1], np.radians(dx[1]), axis=-1)/(r*np.cos(np.radians(lat)))
    dvdx[:,:,0] = np.gradient(np.concatenate([vfield[:,:,-1:,1],vfield[:,:,:2,1]],axis=-1), np.radians(dx[1]), axis=-1)[:,:,1]/(r*np.cos(np.radians(lat[:,0])))
    dvdx[:,:,-1] = np.gradient(np.concatenate([vfield[:,:,-2:,1],vfield[:,:,:1,1]],axis=-1), np.radians(dx[1]), axis=-1)[:,:,1]/(r*np.cos(np.radians(lat[:,0])))
    #
    div    = dudx+dvdy
    curl   = dvdx-dudy
    shear  = dvdx+dudy
    strain = dudx-dvdy
    #
    return div,curl,shear,strain

def gradient(sfield, lat, lon, dx, r=6371E3, rotated=False, glob=True):
    '''
    gradient on a sphere
    '''
    if glob:
        ddy = np.gradient(sfield, np.radians(dx[0]), axis=-2)/r
        ddx = np.gradient(sfield, np.radians(dx[1]), axis=-1)/(r*np.cos(np.radians(lat)))
        ddx[:,:,0] = np.gradient(np.concatenate([sfield[:,:,-1:],sfield[:,:,:2]],axis=-1), np.radians(dx[1]), axis=-1)[:,:,1]/(r*np.cos(np.radians(lat[:,0])))
        ddx[:,:,-1] = np.gradient(np.concatenate([sfield[:,:,-2:],sfield[:,:,:1]],axis=-1), np.radians(dx[1]), axis=-1)[:,:,1]/(r*np.cos(np.radians(lat[:,0])))
    else:
        ddy = np.gradient(sfield, np.radians(dx[0]), axis=-2)/r
        ddx = np.gradient(sfield, np.radians(dx[1]), axis=-1)/(r*np.cos(np.radians(lat)))
    if rotated:
        grad=np.stack((-ddy, ddx), axis=-1)
    else:
        grad=np.stack((ddx, ddy), axis = -1)

    return grad


def calc_slopes(keys,nx,j,dgs,lat,data_out,slopes,interps,theil=True,alpha=0.95,r=6371E3):
    '''
     
    '''
    if j%50==0:
       print(j)
    for kk,K in enumerate(keys):
        #print(kk,K)
        for i in range(nx):
            y=data_out[kk,:,j,i]
            ninds=np.where(np.isfinite(y))[0]
            if np.isfinite(data_out[kk,0,j,i]) and len(ninds)>2:
                if K in ['y']:
                    if theil:
                        #y=data_out[kk,:,j,i]
                        res=stats.theilslopes(np.log10(y[ninds]),np.log10(r*np.radians(np.array(dgs[ninds]))),alpha=alpha)
                elif K in ['x']:
                    if theil:
                        #y=data_out[kk,:,j,i]
                        res=stats.theilslopes(np.log10(y[ninds]),np.log10(r*np.radians(np.array(dgs[ninds]))*np.cos(np.radians(lat[j]))),alpha=alpha)
                elif K in ['xy']:
                    if theil:
                        #y=data_out[kk,:,j,i]
                        dxy=np.sqrt((r*np.radians(np.array(dgs)))**2+(r*np.radians(np.array(dgs))*np.cos(np.radians(lat[j])))**2)
                        res=stats.theilslopes(np.log10(y[ninds]),np.log10(dxy[ninds]),alpha=alpha)
                #
                slopes[kk,:,j,i]=np.array([res[3],res[0],res[2]])
                interps[kk,j,i]=res[1]

def calc_slopes_parallalel(dgs,ny2,nx2,lat,data_out,dims=['x'],num_cores=15,n0=1,n1=5):
    '''
    
    '''
    folder1     = tempfile.mkdtemp()
    path1       = os.path.join(folder1, 'dum1.mmap')
    path2       = os.path.join(folder1, 'dum2.mmap')
    path3       = os.path.join(folder1, 'dum3.mmap')
    path4       = os.path.join(folder1, 'dum4.mmap')
    slopes      = np.memmap(path1, dtype=np.float, shape=((3,)+(3,)+(ny2,nx2)), mode='w+')
    interps     = np.memmap(path2, dtype=np.float, shape=((3,)+(ny2,nx2)), mode='w+')
    data_out_mm = np.memmap(path3, dtype=np.float, shape=(data_out.shape), mode='w+')
    lat_mm      = np.memmap(path4, dtype=np.float, shape=(ny2), mode='w+')
    #
    slopes[:]=np.ones((3,)+(3,)+(ny2,nx2))*np.nan
    interps[:]=np.ones((3,)+(ny2,nx2))*np.nan
    data_out_mm[:]=data_out.copy()
    lat_mm[:]=lat
    #
    Parallel(n_jobs=num_cores)(delayed(calc_slopes)(dims,nx2,j,dgs[n0:n1],lat_mm,data_out_mm[:,n0:n1,:,:],slopes,interps,alpha=0.5) for j in range(ny2))
    slopes=np.array(slopes).squeeze()
    interps=np.array(interps).squeeze()
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    #
    return slopes


def find_weighted_k0_loop(jinds,iinds,dgs,L,k0, KE, perr, rlat, dx2=0.05,r=6371E3, extrap=False, smallscale=True):
    '''
    
    '''
    if not extrap:
        dgs2=np.arange(min(dgs),max(dgs)+dx2,dx2)
    else:
        dgs2=np.arange(0,max(dgs)+dx2,dx2)
    #
    for jj in range(len(jinds)):
        j=jinds[jj]
        i=iinds[jj]
        if smallscale:
            try:
                popt, pcov = curve_fit(func, dgs, KE[:,j,i])
                perr[j,i]  = np.mean(abs(KE[:,j,i] - func(dgs, *popt))/KE[:,j,i])
            except RuntimeError:
                continue
            # KE spectra is cumulative (i.e. how much energy below given length scale)
            # therefore we consider its gradient (which we know analytically)
        if smallscale:
            L[0,j,i] = dgs2[0]
            k0[0,j,i] = 0 #
            for d,dg in enumerate(dgs[1:]):
                x = np.arange(dgs[0],dg+0.01,0.01)
                L0 = (np.radians(x)*r)*np.cos(rlat[j,i])
                # integrate the functional fit to the data
                k0[d+1,j,i] = integrate.simps(L0*func_grad(x,*popt[:-1]),x)
        else:
            L0 = ((np.radians(dgs)*r)**2)*np.cos(rlat[j,i])
            for d,dg in enumerate(dgs):
                # integrate KE*L0 above the filter size (L)
                k0[d,j,i]=integrate.simps(KE[d:,j,i]*np.sqrt(L0[d]),np.sqrt(L0)[d:])
 
def find_weighted_k0(dgs,KE,lon2,lat2,r=6371E3,num_cores=15,extrap=False,smallscale=True):
    '''
    We will take the mixing length to depend on the pdf of the KE spectra
    i.e. diffusivity at given length scale will be proportional to median KE
    over all the length scales smaller than the given length scale.
    This makes more sense than multiplying the KE with the length scale directly
    or applying some ad-hoc criteria to limit the length scale.
    #
    # 
    '''
    KE[np.where(KE==0)]=np.nan
    ny,nx = KE.shape[-2:]
    jinds,iinds = np.where(np.isfinite(np.sum(KE,0)))
    folder1      = tempfile.mkdtemp()
    path1        = os.path.join(folder1, 'dum1.mmap')
    path2        = os.path.join(folder1, 'dum2.mmap')
    path3        = os.path.join(folder1, 'dum3.mmap')
    path4        = os.path.join(folder1, 'dum4.mmap')
    KE_m         = np.memmap(path1, dtype=np.float, shape=KE.shape, mode='w+')
    L            = np.memmap(path2, dtype=np.float, shape=(len(dgs),ny,nx), mode='w+')
    k0           = np.memmap(path3, dtype=np.float, shape=(len(dgs),ny,nx), mode='w+')
    perr         = np.memmap(path4, dtype=np.float, shape=(ny,nx), mode='w+')
    #
    KE_m[:] = KE
    L[:] = np.ones((len(dgs),ny,nx))*np.nan
    k0[:] = np.ones((len(dgs),ny,nx))*np.nan
    perr[:] = np.ones((ny,nx))*np.nan
    n=len(jinds)//num_cores
    #
    Parallel(n_jobs=num_cores)(delayed(find_weighted_k0_loop)(jinds[j*n:(j+1)*n],iinds[j*n:(j+1)*n],dgs,L,k0,KE_m,perr,np.radians(lat2),r=r,extrap=extrap,smallscale=smallscale) for j in range(num_cores))
    L = np.array(L)
    k0 = np.array(k0)
    perr = np.array(perr)
    #
    Li = np.ones(L.shape)*np.nan
    k0i = np.ones(L.shape)*np.nan
    for d in range(min(int(extrap),1-int(smallscale),1),len(dgs)):
        print(d)
        njinds,niinds = np.where(np.isfinite(k0[d,:,:]))
        k0i[d,:,:] = np.reshape(interpolate.griddata((lon2[njinds,niinds],lat2[njinds,niinds]),k0[d,njinds,niinds], (lon2.flatten(),lat2.flatten()),method='linear'),(ny,nx))
    #
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    if smallscale:
        return Li, k0i, perr
    else:
        return Li, k0i

def find_largest_scale_loop(jinds,iinds,dgs,L, KE, dx2=0.01, eps=1/np.e):
    '''
    '''
    dgs2=np.arange(min(dgs),max(dgs)+dx2,dx2)
    for jj in range(len(jinds)):
        j=jinds[jj]
        i=iinds[jj]
        try:
            popt, pcov = curve_fit(func, dgs, KE[:,j,i])
        except RuntimeError:
            continue
        dum = np.gradient(func(dgs2,*popt),dx2)/func(dgs2,*popt)
        einds = np.where(dum>eps)[0]
        if len(einds)>0 and dum[0]>eps:
            L[j,i] = dgs2[einds[-1]]
        elif len(einds)==0 and dum[0]>eps:
            L[j,i] = dgs2[-1]
        elif len(einds)==0 and dum[0]<=eps:
            L[j,i] = dgs2[0]



def find_largest_scale(dgs,KE,lon2,lat2,num_cores=15):
    '''
    Find the scale at which kinetick energy levels of i.e. the largest scale
    of eddies
    '''
    KE[np.where(KE==0)]=np.nan
    ny,nx = KE.shape[-2:]
    jinds,iinds = np.where(np.isfinite(np.sum(KE,0)))
    folder1      = tempfile.mkdtemp()
    path1        = os.path.join(folder1, 'dum1.mmap')
    path2        = os.path.join(folder1, 'dum2.mmap')
    KE_m         = np.memmap(path1, dtype=np.float, shape=KE.shape, mode='w+')
    L            = np.memmap(path2, dtype=np.float, shape=(ny,nx), mode='w+')
    #
    KE_m[:] = KE
    L[:] = np.ones((ny,nx))*np.nan
    #num_cores=15
    n=len(jinds)//num_cores
    #
    Parallel(n_jobs=num_cores)(delayed(find_largest_scale_loop)(jinds[j*n:(j+1)*n],iinds[j*n:(j+1)*n],dgs,L,KE_m) for j in range(num_cores))
    L = np.array(L)
    #
    njinds,niinds = np.where(np.isfinite(L))
    Li = interpolate.griddata((lon2[njinds,niinds],lat2[njinds,niinds]),L[njinds,niinds], (lon2.flatten(),lat2.flatten()),method='linear')
    Li = np.reshape(Li,lat2.shape)
    Li[np.where(np.isnan(np.sum(KE,0)))] = np.nan
    #
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    return Li
   

def smart_wmean(din,win):
    '''
    weighted mean 
    '''
    neginds=np.where(din.flatten()<0)[0]
    naninds=np.where(np.isnan(din.flatten()))[0]
    nonnaninds=np.where(np.isfinite(din.flatten()))[0]
    lmean=np.nanmean(win)
    if len(nonnaninds) > len(naninds) and len(neginds)/(len(nonnaninds)-len(naninds)) < 0.75:
        din[np.where(din<0)]=np.nan
        din[np.where(din>5E4*lmean/25E3)]=0
        dout=np.nansum(win*np.nanmean(din,-1))/np.nansum(win*np.isfinite(np.nanmean(din,-1)))
    else:
        dout=np.nan
    #                                                                                                                                             
    return dout


def return_grid(dg_out,llon=-180,ulon=180):
    '''
     
    '''
    lat_out    = np.arange( -90+dg_out/2, 90, dg_out)
    lon_out    = np.arange(llon+dg_out/2,ulon, dg_out)
    lat_b_out  = np.arange( -90, 90+dg_out/2, dg_out)
    lon_b_out  = np.arange(llon, ulon+dg_out/2, dg_out)
    lon_out[np.where(lon_out>180)]=lon_out[np.where(lon_out>180)]-360
    lon_b_out[np.where(lon_b_out>180)]=lon_b_out[np.where(lon_b_out>180)]-360
    ds_out     = xr.Dataset({'lat': (['lat'], lat_out),'lon': (['lon'], lon_out),'lat_b':(['lat_b'], lat_b_out), 'lon_b': (['lon_b'], lon_b_out), })
    #                                                                                                                                        
    return ds_out

def perform_regrid(regridder,datain):
    '''
    
    '''
    dum=datain[:].copy()
    dum[np.where(np.isnan(dum))]=0
    mask=np.ones(dum.shape)
    mask[np.where(dum==0)]=0
    k_out = regridder(dum)
    mask  = regridder(mask)
    return k_out/mask


def rotate_symmetric_tensor(n,ny,nx,Kout,kalong,kacross,angle):    
    ''' '''
    for j in range(ny):
        if j%100==0:
            print(j)
        for i in range(nx):
            if np.isfinite(kalong[j,i]):
               Ktot=np.array([[kalong[j,i],0], [0,kacross[j,i]]])
               R=mutils.rot_m(angle[j,i])
               Kout[n,:,:,j,i]=R.T@Ktot@R


def expand_mask(mask):
    '''
    mask should be 1 on land, 0 in ocean
    '''
    jinds,iinds=np.where(mask)
    ny,nx = mask.shape
    # 
    njinds=jinds+1
    sjinds=jinds-1
    eiinds=iinds+1
    wiinds=iinds-1
    njinds[np.where(njinds==ny)]=ny-1
    sjinds[np.where(sjinds==-1)]=0
    wiinds[np.where(wiinds==-1)]=nx-1
    eiinds[np.where(eiinds==nx)]=0
    #
    mask[njinds,iinds]=1
    mask[njinds,eiinds]=1
    mask[njinds,wiinds]=1
    mask[sjinds,iinds]=1
    mask[sjinds,eiinds]=1
    mask[sjinds,wiinds]=1
    mask[jinds,eiinds]=1
    mask[jinds,wiinds]=1
    #
    return mask


def K_mixlen_from_k0(dgs,k0,KEl,KEs,V_l,U_l,a4=0.1):
    #
    ny, nx = k0.shape[-2:]
    kacross = k0/(1+a4*(KEl/KEs))
    kalong = k0*(1+a4*(KEl/KEs))
    # ROTATION BACK TO X-Y PLANE
    Kout = np.ones((2,2,ny,nx))*np.nan
    angle = np.arctan2(V_l,U_l)
    #
    temp_folder = tempfile.mkdtemp()
    K_MixLen = np.ones((len(dgs),2,2,ny,nx))*np.nan
    file1 = os.path.join(temp_folder, 'dum.mmap')
    file2 = os.path.join(temp_folder, 'dum2.mmap')
    file3 = os.path.join(temp_folder, 'dum3.mmap')
    file4 = os.path.join(temp_folder, 'dum4.mmap')
    #
    dump(K_MixLen, file1)
    K_MixLen = load(file1, mmap_mode='w+')
    dump(kalong, file2)
    dump(kacross, file3)
    dump(angle, file4)
    kalong = load(file2, mmap_mode='r+')
    kacross = load(file3, mmap_mode='r+')
    angle = load(file4, mmap_mode='r+')
    #
    Parallel(n_jobs=len(dgs))(delayed(rotate_symmetric_tensor)(n,ny,nx,K_MixLen,kalong[n,1,:,:],kacross[n,1,:,:],angle[n,1,:,:]) for n in range(len(dgs)))
    #
    Kx_MixLen = np.array(K_MixLen[:,0,0,:,:])
    Ky_MixLen = np.array(K_MixLen[:,1,1,:,:])
    Kxy_MixLen = np.array(K_MixLen[:,0,1,:,:])
    #
    shutil.rmtree(temp_folder)
    return Kx_MixLen, Ky_MixLen, Kxy_MixLen

def magnitude_as_square_of_diagonals_loop(K_mag,ji,ii,x,y,xy):
    ''' '''
    for j in range(len(x)):
        if np.isfinite(x[j]+y[j]+xy[j]):
            r, k, rinv = np.linalg.svd(np.array([[x[j],xy[j]],[xy[j],y[j]]]))
            K_mag[j,ji,ii] = np.sign(k).min()*np.sqrt(k[0]*k[1])

def magnitude_as_square_of_diagonals_loop2(K_mag,a,jinds,iinds,x,y,xy):
    ''' '''
    for j in range(len(jinds)):
        jj=jinds[j]
        ii=iinds[j]
        if np.isfinite(x[jj,ii]+y[jj,ii]+xy[jj,ii]):
            r, k, rinv = np.linalg.svd(np.array([[x[jj,ii],xy[jj,ii]],[xy[jj,ii],y[jj,ii]]]))
            K_mag[a,jj,ii] = np.sign(k).min()*np.sqrt(k[0]*k[1])

def magnitude_as_square_of_diagonals(Kx,Ky,Kxy,num_cores=8):
    '''
    '''
    #
    folder1     = tempfile.mkdtemp()
    path1       = os.path.join(folder1, 'dum1.mmap')
    #
    K_mag       = np.memmap(path1, dtype=np.float, shape=Kx.shape, mode='w+')
    K_mag[:]    = np.ones(Kx.shape)*np.nan
    #
    jinds, iinds = np.where(np.isfinite(Kx[0,:,:]))
    #
    #(n_jobs=num_cores)(delayed(magnitude_as_square_of_diagonals_loop)(K_mag,jinds[a],iinds[a],Kx[:,jinds[a],iinds[a]],Ky[:,jinds[a],iinds[a]],Kxy[:,jinds[a],iinds[a]]) for a in range(len(jinds)))
    Parallel(n_jobs=num_cores)(delayed(magnitude_as_square_of_diagonals_loop2)(K_mag,a,jinds,iinds,Kx[a,:,:],Ky[a,:,:],Kxy[a,:,:]) for a in range(Kx.shape[0]))
    #
    K_mag = np.array(K_mag)
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    return K_mag



def optimize_mixing_loop(a1,a,b_range,X,y1,y2,err,std,interp,coslat,ki):
    '''
     
    '''
    if ki in ['ransac']:
        for b1,b in enumerate(b_range):
            ransac = linear_model.RANSACRegressor()
            y=a*y1+b*y2
            try:
                ransac.fit(X,y)
                err[a1,b1] = abs(ransac.estimator_.coef_[0]-1)
            except ValueError:
                err[a1,b1]=np.nan
            #err[a1,b1] = abs(ransac.estimator_.coef_[0]-1)
    elif ki in ['theil']:
        for b1,b in enumerate(b_range):
            try:
                res=stats.theilslopes(a*y1.squeeze()+b*y2.squeeze(),X.squeeze(),alpha=0.5)
                err[a1,b1]=(res[0]-1)
                std[a1,b1]=res[3]-res[2]
                interp[a1,b1]=res[1]
            except ValueError:
                err[a1,b1]=np.nan
                std[a1,b1]=np.nan
                interp[a1,b1]=np.nan
    elif ki in ['wasserstein']:
        for b1,b in enumerate(b_range):
            err[a1,b1]=stats.wasserstein_distance(a*y1.squeeze()+b*y2.squeeze(),X.squeeze(),u_weights=coslat, v_weights=coslat)
    elif ki in ['mpe']:
        for b1,b in enumerate(b_range):
            err[a1,b1]=np.sum(coslat*abs((a*y1.squeeze()+b*y2.squeeze())-X.squeeze())/X.squeeze())/np.sum(coslat)

def optimize_mixing(K_target, Kll, Kss, lat, a_range, b_range, ki='full',num_cores=10):
    '''
    Optimize mixing with different options. Theil-Sen estimator is probably most robust
    
    K_target : Diffusivity against which we optimize
    Kll      : Stirring by large scale flow
    Kss      : Mixing by small scale flow
    Lat      : Latitude
    a_range  : Range of stirring coefficients
    b_range  : Range of mixing coefficients
    ki       : Type of optimization, 'ransac','theil', 'full', 'rel', 'zonal'
               'ransac' and 'theil' try to optimize the most likely slope to be close to 1
               which is likely to be more robust than trying to minimize the squared errors
               which is what the other methods do.
    '''
    #err=np.ones((len(a_range),len(b_range)))*np.nan
    coslat=np.cos(np.radians(lat))
    jinds1,iinds1=np.where(np.isfinite(K_target+Kll+Kss))
    coslat2=coslat[jinds1,iinds1]
    if ki in ['ransac','theil','wasserstein','mpe']:
        X=K_target[jinds1,iinds1][np.newaxis,].T
        y1=Kll[jinds1,iinds1][np.newaxis,].T
        y2=Kss[jinds1,iinds1][np.newaxis,].T
        folder1     = tempfile.mkdtemp()
        path1       = os.path.join(folder1, 'dum1.mmap')
        path2       = os.path.join(folder1, 'dum2.mmap')
        path3       = os.path.join(folder1, 'dum3.mmap')
        err_mm      = np.memmap(path1, dtype=np.float, shape=(len(a_range),len(b_range)), mode='w+')
        std_mm      = np.memmap(path2, dtype=np.float, shape=(len(a_range),len(b_range)), mode='w+')
        interp_mm   = np.memmap(path3, dtype=np.float, shape=(len(a_range),len(b_range)), mode='w+')
        err_mm[:]   = np.ones((len(a_range),len(b_range)))*np.nan
        std_mm[:]   = np.ones((len(a_range),len(b_range)))*np.nan
        interp_mm[:]= np.ones((len(a_range),len(b_range)))*np.nan
        Parallel(n_jobs=num_cores)(delayed(optimize_mixing_loop)(a1,a,b_range,X,y1,y2,err_mm,std_mm,interp_mm,coslat2,ki) for a1,a in enumerate(a_range))
        err=np.array(err_mm)
        std=np.array(std_mm)
        interp=np.array(interp_mm)
        try:
           shutil.rmtree(folder1)
        except OSError:
           pass
    else:
        err=np.ones((len(a_range),len(b_range)))*np.nan
        for a1,a in enumerate(a_range):
            print(a1)
            for b1,b in enumerate(b_range):
                if ki in ['full']:
                    err[a1,b1] = bn.nansum(coslat*(K_target-(a*Kll+b*Kss))**2)
                elif ki in ['rel']:
                    err[a1,b1] = bn.nansum(coslat*abs(K_target-(a*Kll+b*Kss))/abs(K_target))
                elif ki in ['zonal']:
                    err[a1,b1] = bn.nansum(coslat[:,0]*(abs(bn.nanmedian(K_target,-1)-bn.nanmedian(a*Kll+b*Kss,-1)))) #/bn.nanmedian(K_target,-1))
    #print(err)
    abest,bbest = np.where(abs(err)==np.nanmin(abs(err)))
    #
    return a_range[abest[0]], b_range[bbest[0]], err, std, interp


def optimize_mixing_and_slope(dgs,K_target, slopes_target, Kll, Kss, lat, a_range, b_range, ki='rel'):
    #
    err=np.ones((len(a_range),len(b_range)))
    ny2,nx2 = Kll.shape[-2:]
    coslat=np.cos(np.radians(lat))
    for a1,a in enumerate(a_range):
        for b1,b in enumerate(b_range):
            print('calculating slopes')
            folder1     = tempfile.mkdtemp()
            path1       = os.path.join(folder1, 'dum1.mmap')
            path2       = os.path.join(folder1, 'dum2.mmap')
            path3       = os.path.join(folder1, 'dum3.mmap')
            path4       = os.path.join(folder1, 'dum4.mmap')
            slopes_mm   = np.memmap(path1, dtype=np.float, shape=((1,)+(3,)+(ny2,nx2)), mode='w+')
            interps_mm  = np.memmap(path2, dtype=np.float, shape=((1,)+(ny2,nx2)), mode='w+')
            data_out_mm = np.memmap(path3, dtype=np.float, shape=((1,)+Kll.shape), mode='w+')
            lat_mm      = np.memmap(path4, dtype=np.float, shape=(ny2), mode='w+')
            #
            slopes_mm[:]=np.ones((1,)+(3,)+(ny2,nx2))*np.nan
            interps_mm[:]=np.ones((1,)+(ny2,nx2))*np.nan
            data_out_mm[:]=a*Kll.copy()+b*Kss.copy()
            lat_mm[:]=lat[:,0].copy()
            num_cores=15
            n0=0
            n=5
            print(data_out_mm.shape)
            #
            Parallel(n_jobs=num_cores)(delayed(calc_slopes)(['x'],nx2,j,dgs[n0:n],lat_mm,data_out_mm[:,n0:n,:,:],slopes_mm,interps_mm,alpha=0.5) for j in range(ny2))
            print('done')
            #
            print(slopes_mm.shape)
            print(slopes_target.shape)
            err[a1,b1] = bn.nansum(coslat*abs(K_target-(a*Kll[0,:,:]+b*Kss[0,:,:]))/abs(K_target) + coslat*abs((np.array(slopes_mm)[0,1,:,:]-slopes_target)/slopes_target))
            try:
                shutil.rmtree(folder1)
            except OSError:
                pass
    #
    abest,bbest = np.where(err==np.min(err))
    return a_range[abest[0]], b_range[bbest[0]], err


def func_p(x,a,b,c,d):
    return a*(x**b)+x*c+d

def func_p_grad(x,a,b,c):
    return a*b*(x**(b-1))+c
