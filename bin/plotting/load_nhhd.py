###########################
#
# AUTHOR: ALEKSI NUMMELIN
# YEAR:   2018-2020
#
# THIS SCRIPT WILL PREPROCESS
# THE DECOMPOSED VELOCITY DATA
# FOR MAKING THE FINAL FIGURES
# FOR NUMMELIN ET AL. 2020.
#
######################
# LOAD VELOCITY DATA #
######################
#
year='2003'
data=xr.open_mfdataset('../..data/processed/surface_currents_'+year+'_*.nc',parallel=True,concat_dim='time',combine='nested')
lon2,lat2=np.meshgrid(data.lon,data.lat)
r_earth=6371E3
nt,nd,ny,nx=data.ul_out.shape
#
SmallScale={}
LargeScale={}
years = ['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012']
ny=len(years)
c=0
for y,year in enumerate(years):
    print(y,year)
    dum1 = np.load('../data/processed/mean_currents_'+year+'_v4.npz')
    dum2 = np.load('../data/processed/LargeScaleMixing_'+year+'_v4.npz')
    if y==0:
        dgs = dum1['dgs']
        dgs0=dgs/0.25
        for var in dum1.keys():
            print(var)
            if var in ['u0','v0']:
                SmallScale[var]=np.percentile(dum1[var],[25,50,75],axis=0)
            else:
                SmallScale[var]=dum1[var]
        for var in dum2.keys():
            LargeScale[var]=dum2[var]
    else:
        for var in dum1.keys():
            print(y,var)
            if var in ['u0','v0']:
                SmallScale[var]=(np.percentile(dum1[var],[25,50,75],axis=0)+SmallScale[var]*y)/(y+1)
            else:
                SmallScale[var]=(dum1[var]+SmallScale[var]*y)/(y+1)
        for var in dum2.keys():
            print(y,var)
            LargeScale[var]=(dum2[var]+LargeScale[var]*y)/(y+1)
#
######################################
# LARGE SCALE SHEAR DRIVEN MIXING    #
# NOTE THAT IN THE PRE-PROCESSING    #
# WE LEFT THE LENGTH SCALE OPEN      # 
# HERE WE USE TWO DIFFERENT VERSIONS #
######################################
# ORIGINAL SUGGESTION BY LeSommer et al (2014):
#
ndg,nper,ny,nx = LargeScale['Kym'][:].shape
h0 = (np.tile(np.radians(dgs0*0.25)*r,(nx,ny,nper,1)).T**2)*np.cos(np.radians(lat2))
Kym  = LargeScale['Kym'][:]*h0
Kxm  = LargeScale['Kxm'][:]*h0
Kxym = LargeScale['Kxym'][:]*h0
#
if not frobenius:
    Km_tot = h0[:,1,:,:]*hutils.magnitude_as_square_of_diagonals(LargeScale['Kxm'][:,1,:,:],LargeScale['Kym'][:,1,:,:],LargeScale['Kxym'][:,1,:,:],num_cores=LargeScale['Kxm'].shape[0])
else:
    # Frobenius norm 
    Km_tot = np.sqrt(LargeScale['Kxm'][:]**2 + LargeScale['Kym'][:]**2 + 2*(LargeScale['Kxym'][:]**2))*((np.tile(np.radians(dgs0*0.25)*r,(nx,ny,nper,1)).T*np.cos(np.radians(lat2)))**2)
#
# MODIFIED SUGGESTION BY THIS PAPER:
# INTEGRATE OVER ALL THE LENGTH SCALES LARGER THAN THE GIVEN LENGTH SCALE (TAKES INTO ACCOUNT SHEAR AT LARGER SCALES, NOT JUST AT THE GRID SCALE)
Ldum, Kym5  = hutils.find_weighted_k0(dgs0*0.25,LargeScale['Kym'][:,1,:,:],lon2,lat2,r=r_earth,num_cores=15,extrap=False,smallscale=False)
Ldum, Kxm5  = hutils.find_weighted_k0(dgs0*0.25,LargeScale['Kxm'][:,1,:,:],lon2,lat2,r=r_earth,num_cores=15,extrap=False,smallscale=False)
Ldum, Kxym5 = hutils.find_weighted_k0(dgs0*0.25,LargeScale['Kxym'][:,1,:,:],lon2,lat2,r=r_earth,num_cores=15,extrap=False,smallscale=False)
#
if not frobenius:
    dumtot = hutils.magnitude_as_square_of_diagonals(LargeScale['Kxm'][:,1,:,:],LargeScale['Kym'][:,1,:,:],LargeScale['Kxym'][:,1,:,:],num_cores=LargeScale['Kxm'].shape[0])
else:
    # Frobenius norm
    dumtot = np.sqrt(LargeScale['Kxm'][:,1,:,:]**2+LargeScale['Kym'][:,1,:,:]**2 + 2*(LargeScale['Kxym'][:,1,:,:]**2))
#
Ldum, Km_tot5 = hutils.find_weighted_k0(dgs0*0.25,dumtot,lon2,lat2,r=r_earth,num_cores=15,extrap=False,smallscale=False)
# FIX THE MASK
mask2 = hutils.expand_mask(1-np.floor(LargeScale['mask']))
mask2 = hutils.expand_mask(mask2)
#
jinds2,iinds2=np.where(mask2)
Kym5[:,jinds2,iinds2] = np.nan
Kxm5[:,jinds2,iinds2] = np.nan
Kxym5[:,jinds2,iinds2] = np.nan
Km_tot5[:,jinds2,iinds2] = np.nan
Kxm[:,:,jinds2,iinds2] = np.nan
Kym[:,:,jinds2,iinds2] = np.nan
Km_tot[:,jinds2,iinds2] = np.nan
#####################################
# MIXING BY SUBGRIDSCALE VELOCITIES #
#####################################
#
# load the eddy binned eddy sizes
Leddy_data = np.load('../../data/processed/eddy_core_radius_binned_scale1.npz')
Leddy2 = 1E3*Leddy_data['var_grid'][:]
Leddy2[np.where(mask2)]=np.nan
Leddy2[:,-1]=np.nanmedian([Leddy2[:,-2],Leddy2[:,0]],axis=0) #for some reason the 180 bin is corrupted
inds2=np.where(abs(Leddy_data['grid_y'][:,0])<6.0)[0]
inds3=np.where(np.logical_and(abs(Leddy_data['grid_y'][:,0])>6.0,abs(Leddy_data['grid_y'][:,0])<=7.0))[0]
Leddy2[inds2,:]=np.nanmedian(Leddy2[inds3,:])
#
# length scale is the minimum of the observed
# eddy radius and the local filter size
k30 = np.ones(tuple([len(dgs0)])+lat2.shape)*np.nan
k30x=np.ones(tuple([len(dgs0)])+lat2.shape)*np.nan
k30y=np.ones(tuple([len(dgs0)])+lat2.shape)*np.nan
for j in range(len(dgs0)):
    L_h2 = np.radians(dgs0[j]*0.25)*r*np.sqrt(np.cos(np.radians(lat2)))
    k30[j,:,:]  = (SmallScale['KEs'][j,1,:,:]*np.nanmin([L_h2,Leddy2],axis=0))
    k30x[j,:,:] = (SmallScale['KEsx'][j,1,:,:]*np.nanmin([L_h2,Leddy2],axis=0))
    k30y[j,:,:] = (SmallScale['KEsy'][j,1,:,:]*np.nanmin([L_h2,Leddy2],axis=0))
#
k30[np.where(k30==0)]=np.nan
k30x[np.where(k30x==0)]=np.nan
k30y[np.where(k30y==0)]=np.nan
#
