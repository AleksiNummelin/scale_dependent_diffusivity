#################################################
# AUTHOR: ALEKSI NUMMELIN
# YEAR:   2018-2020
#
# This script will plot figures 2-9
# of Nummelin et al. (2020). Note that figure 9
# in the paper has gone through some additional
# hand editing. Figures 1 and 10 are hand made.
#
#################################################
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as mticker
#
def smart_wmean(din,win):
    '''
    DO NOT TAKE 0'S INTO ACCOUNT
    '''
    din[np.where(din<0)]=np.nan
    dout=np.nansum(win*din)/np.nansum(np.isfinite(din)*win)
    return dout
                    

def monte_carlo_r(y,x,n2=0.9,nn=100,pers=[5,50,95]):
    '''
    n2 = number of obs in each ensemble member (% of the whole length)
    nn = number of ensemble members 
    '''
    dum_slope   = np.zeros(nn)
    dum_interps = np.zeros(nn)
    inds=[]
    r=[]
    slope=[]
    intercept=[]
    num_models=len(x)
    num_subgroups=int(n2*num_models) #n2
    n=0
    c=0
    while n<nn:
        dum = np.sort(np.random.choice(np.arange(num_models),(num_subgroups),replace=False),axis=0)
        if n==0:
             inds.append(dum)
        else:
            if np.min(np.sum(abs(np.array(inds)-dum),axis=1))>0:
                inds.append(dum)
        n=len(inds)
        c=c+1
    #
    for i,ind in enumerate(inds):
            t=stats.linregress(x[ind],y[ind])
            r.append(t.rvalue)
            slope.append(t.slope)
            intercept.append(t.intercept)

    return np.percentile(slope,pers),np.percentile(intercept,pers), np.percentile(r,pers)

def spatial_filter(datain, std=2, mode='wrap'):
    #
    mask = np.isfinite(datain)
    dum = datain.copy()
    dum[np.where(1-mask)] = 0
    dum = gaussian_filter(dum,std,mode=mode)/gaussian_filter(mask.astype(np.float),std,mode=mode)
    dum[np.where(1-mask)] = np.nan
    dum[np.where(dum==0)] = np.nan
    #
    return dum

def map2_and_zonal_mean(projection1=ccrs.PlateCarree(200)):
    '''
    SETUP A FIGURE WITH 2 MAPS AND A ZONAL MEAN 
    '''
    fig  = plt.figure(figsize=(16,10))
    ax1  = plt.subplot2grid((2, 4), (0, 0), colspan=3, projection=projection1)
    ax2  = plt.subplot2grid((2, 4), (1, 0), colspan=3, projection=projection1)
    ax3  = plt.subplot2grid((2, 4), (0, 3), rowspan=2)
    #
    for ax in [ax1,ax2]:
        #
        ax.set_xticks([-180-20,-120-20,-60-20,0-20,60-20,120-20],crs=projection1)
        ax.set_xticklabels([-180,-120,-60,0,60,120], color='gray', fontdict={'fontsize':14})
        ax.set_yticks([-45,-30,-15,0,15,30,45], crs=projection1)
        ax.set_yticklabels([-45,-30,-15,0,15,30,45], color='gray', fontdict={'fontsize':14})
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    ax3.yaxis.tick_right()
    ax3.set_ylim(-60,60)
    #
    return fig, ax1, ax2, ax3


def map_and_zonal_mean(projection1=ccrs.PlateCarree(200)):
    '''
    SETUP A FIGURE WITH 3 MAPS AND A ZONAL MEAN
    '''
    fig  = plt.figure(figsize=(20,15))
    ax1  = plt.subplot2grid((3, 4), (0, 0), colspan=3, projection=projection1)
    ax2  = plt.subplot2grid((3, 4), (1, 0), colspan=3, projection=projection1)
    ax3  = plt.subplot2grid((3, 4), (2, 0), colspan=3, projection=projection1)
    ax4  = plt.subplot2grid((3, 4), (0, 3), rowspan=3)
    #
    for ax in [ax1,ax2,ax3]:
        #
        ax.set_xticks([-180-20,-120-20,-60-20,0-20,60-20,120-20],crs=projection1)
        ax.set_xticklabels([-180,-120,-60,0,60,120], color='gray', fontdict={'fontsize':14})
        ax.set_yticks([-45,-30,-15,0,15,30,45], crs=projection1)
        ax.set_yticklabels([-45,-30,-15,0,15,30,45], color='gray', fontdict={'fontsize':14})
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

    #
    ax4.yaxis.tick_right()
    ax4.set_ylim(-60,60)
    #
    return fig, ax1, ax2, ax3, ax4



mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
#
plotpath='figures/'
# DEFINE A LIMIT BELOW WHICH DATA IS CONSIDERED TOO NOISY
std_lim=2.5
# I.E IF THE SIGNAL NEEDS TO BE STD_LIM*NOISE TO BE SIGNIFICANT
# THIS IS EMPIRICAL JUDGEMENT BUT SEEMS TO WORK WELL IN SEPARATING
# OUT THE NOISY REGIONS.
#
# DEFINE COLORMAPS
#
# Figure 2 & 5
levelsX=np.array([0,100,200,300,500,750,1000,1250,1500,2000,2500,3000])
cmap0=mpl.cm.viridis
cmlist=[];
for cl in np.linspace(0,252,len(levelsX)): cmlist.append(int(cl))
cmapX, normX = from_levels_and_colors(levelsX,cmap0(cmlist),extend='max');

# Figure 4 and 8
levels3=np.array([0,.25,.5,.75,1,1.25,1.5,1.75,2]);
levels3_str=['0','0.25','0.5','0.75','1','1.25','1.5','1.75','2']
cmap0=mpl.cm.coolwarm
cmlist=[];
for cl in np.linspace(0,252,len(levels3)+1): cmlist.append(int(cl))
cmap3, norm3 = from_levels_and_colors(levels3,cmap0(cmlist),extend='both');

# Figure 4
levels4=levelsX/1000
cmap0=mpl.cm.viridis
cmlist=[];
for cl in np.linspace(0,252,len(levels4)+1): cmlist.append(int(cl))
cmap4, norm4 = from_levels_and_colors(levels4,cmap0(cmlist),extend='both');

# Figure 9 and S1
#levelsE=np.array([0,0.1,0.25,0.5,0.75,1,1.5,2,3,5,10,20])
levelsE=np.array([0,0.025,0.05,0.1,0.25,0.5,1,1.25,1.5,2,3,5,10])
cmap0=mpl.cm.viridis
cmlist=[];
for cl in np.linspace(0,252,len(levelsE)): cmlist.append(int(cl))
cmapE, normE = from_levels_and_colors(levelsE,cmap0(cmlist),extend='max')

###########################
# START THE ACTUAL PLOTTING
#
##################################
# FIGURE 2 - MAP AND ZONAL MEANS #
##################################
fig1, ax1, ax2, ax3= map2_and_zonal_mean()
axes=[ax1,ax2,ax3];
ttls=[]
titles=['$\kappa_{SSTA}$','$\kappa_{MITgcm}$','$\kappa_{SLA}$']
for j,ax in  enumerate([ax1,ax2]):
    ax.coastlines(resolution='50m',color='darkgray')
    ax.set_ylim(-60,60)
    ttl1=ax.set_title(titles[j],fontsize=25,fontdict={'verticalalignment': 'bottom'})
    ttls.append(ttl1)

dlon = data_all_out['sstds_out'].lon.values
dlon[np.where(dlon<0)] = dlon[np.where(dlon<0)]+360
njinds,ninds=np.where(mask3)
#
data0_all['sst_model']['Lon_vector']=data0_all['sst_model']['xx']
data0_all['sst_model']['Lat_vector']=data0_all['sst_model']['yy']
#
KK_zonal_means={}
KK_zonal_means0={}
KK_all = {}
for j,key in enumerate(['sst','sst_model']):
    print(j)
    if not frobenius:
        KKdum = np.sqrt(abs(data_ens[key+'_10']['Kx_25']*data_ens[key+'_10']['Ky_25']))
        KKdum = np.nanmin([np.sign(data_ens[key+'_10']['Kx_25']),np.sign(data_ens[key+'_10']['Ky_25'])],axis=0)*KKdum
    else:
        KKdum = np.sqrt(data_ens[key+'_10']['Kx_25']**2+data_ens[key+'_10']['Ky_25']**2)
    # 
    mask0 = (KKdum<=0)
    KKdum = spatial_filter(KKdum,2)
    KKdum[np.where(mask0)] = np.nan
    #
    KKdum25,KKdum50,KKdum75 = np.nanpercentile(KKdum,[25,50,75],axis=0)
    #
    KKmask = spatial_filter(KKdum50/abs(KKdum75-KKdum25),2)
    #
    dlon = data_ens[key+'_10']['lon25']
    dlat = data_ens[key+'_10']['lat25']
    if len(dlon.shape)>1:
        dlon=np.nanmean(dlon,0)
        dlat=np.nanmean(dlat,1)
    #
    jinds  = np.where(np.logical_and(dlat>-70, dlat<79.375))[0]
    # 
    KK_all[key] = KKdum[:,jinds,:]
    KK_all[key+'med']   = KKdum50[jinds,:]
    KK_all[key+'_mask'] = KKmask[jinds,:]
    KK_all[key+'_dlat'] = dlat[jinds]
    KK_zonal_means[key+'_dlat'] = dlat[jinds]
    KK_all[key+'_dlon'] = dlon

# combine the mask so that the zonal mean will be taken over the same regions for both
# model data and the observed SSTs. Note that this mask is a bit different than the
# std_mask set above, but we have tuned this so that the masked areas are similar.
#
combined_mask = ((KK_all['sst_mask']>1)+(KK_all['sst_model_mask']>1)).astype(np.float)
combined_mask[np.where(combined_mask<1)] = np.nan
#
for j,key in enumerate(['sst','sst_model']):
    dlat    = KK_all[key+'_dlat'][:]
    dlon    = KK_all[key+'_dlon'][:]
    KKdum   = KK_all[key+'med']
    # mask data
    KKdum[np.where(KK_all[key+'_mask'] < 1)] = np.nan
    #
    dlon[np.where(dlon<0)] = dlon[np.where(dlon<0)]+360
    dumX,dumXlon = add_cyclic_point(KKdum,coord=dlon)
    dumX[np.where(dumX<=0)] = np.nan
    cm0 = axes[j].pcolormesh(dumXlon, dlat, dumX,cmap=cmapX, norm=normX,rasterized=True, transform=ccrs.PlateCarree())
    #apply the mask and calculate zonal mean
    KK_all[key+'_masked'] = KK_all[key]*combined_mask
    KK_all[key+'_masked'][np.where(KK_all[key+'_masked']<=1)]=np.nan
    KK_zonal_means[key] = np.nanpercentile(np.nanmedian(KK_all[key+'_masked'],-1),[5,50,95],axis=0)
    KK_zonal_means[key+'_ens'] = np.nanpercentile(np.nanmedian(KK_all[key+'_masked'],0),[25,75],axis=-1)

######################
# ADD KINETIC ENERGY #
KE_tot_med = np.nanmedian(SmallScale['KEtot'][:,1,:,:],axis=0)
dumX = regridder_2deg(KE_tot_med)
dumX,dumXlon = add_cyclic_point(dumX,coord=lon_out)
for j, ax in enumerate([ax1,ax2]):
    ax.contour(dumXlon,lat_out, dumX,levels=np.array([0.25]),colors='r',rasterized=True, transform=ccrs.PlateCarree())

#######################################################
# CALCULATE THE ZONAL MEANS USING THE FULL TIMESERIES #
for j,key in enumerate(['data_sst_avhrr','sst_model']):
    if not frobenius:
        #
        KKdum = np.nanmin([np.sign(data0_all[key]['Kx']),np.sign(data0_all[key]['Ky'])],axis=0)*np.sqrt(abs(data0_all[key]['Kx']*data0_all[key]['Ky']))
        KKdum[np.where(KKdum<=0)]=np.nan
    else:
        KKdum = np.sqrt(np.nanmin([np.sign(data0_all[key]['Kx']),np.sign(data0_all[key]['Ky'])],axis=0)*(data0_all[key]['Kx']**2+data0_all[key]['Ky']**2))
    #
    yinds = np.where(np.logical_and(data0_all[key]['Lat_vector']>-70, data0_all[key]['Lat_vector']<79.375))[0]
    KK_zonal_means0[key] = np.nanmedian(KKdum[yinds,:]*combined_mask,-1)
    KK_zonal_means0[key+'_dlat'] = data0_all[key]['Lat_vector'][yinds]
#
ll=[]
sm_win = np.ones(4)/4
for v,var in enumerate(['data_sst_avhrr','sst_model']):
    l1,= ax3.semilogx(dum,KK_zonal_means[var+'_dlat'],color='C'+str(v),lw=2)
    ax3.semilogx(KK_zonal_means0[var],KK_zonal_means0[var+'_dlat'],color='C'+str(v),lw=2,ls='--')
    ll.append(l1)
    # smooth the percentiles before plotting
    dum = np.convolve(KK_zonal_means[var][1,], sm_win, mode='same')
    dum125,dum175= np.convolve(KK_zonal_means[var][0,], sm_win, mode='same'), np.convolve(KK_zonal_means[var][2,], sm_win, mode='same')
    dum225,dum275= np.convolve(KK_zonal_means[var+'_ens'][0,],sm_win, mode='same'),np.convolve(KK_zonal_means[var+'_ens'][1,],sm_win, mode='same')
    ax3.fill_betweenx(KK_zonal_means[var+'_dlat'],dum125,dum175,color='C'+str(v),alpha=0.3)
    ax3.fill_betweenx(KK_zonal_means[var+'_dlat'],dum225,dum275,color='C'+str(v),alpha=0.3)

fig1.subplots_adjust(wspace=0.1)
fig1.subplots_adjust(hspace=0.1)
cax  = fig1.add_axes([0.12,0.05,0.55,0.02])
cbar = plt.colorbar(mappable=cm0,cax=cax,orientation='horizontal')
clab = cbar.ax.set_xlabel('Diffusivity [m$^2$ s$^{-1}$]',fontsize=25,labelpad=15)
ylab = ax3.set_ylabel('Latitude [$\degree$]', fontsize=25)
xlab = ax3.set_xlabel('Diffusivity [m$^2$ s$^{-1}$]', fontsize=25)
ax3.yaxis.set_label_position('right')
ax3.legend(ll,titles,fontsize=22,handlelength=.75, handletextpad=0.5)
ax3.set_xlim(3E1,3E3)
extra_artists=[clab,xlab,ylab]; extra_artists.extend(ttls)
for j,ax in enumerate(axes):
    txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=20)
    extra_artists.append(txt1)

plt.savefig(plotpath+'Figure2_'+version+'.png',format='png',dpi=300,pad_inches=1,bbox_inches='tight',bbox_extra_artists=extra_artists)
#
###########################################################################
# FIG 4 - SLOPES AND ZONAL MEAN
#
fig3, ax1, ax2, ax3 = map2_and_zonal_mean()
axes=[ax1,ax2,ax3];
ttls=[]
titles=['$\kappa_{SSTA}$','$\kappa_{MITgcm}$','$\kappa_{SLA}$']
for j,ax in  enumerate([ax1,ax2]):
    ax.coastlines(resolution='50m',color='darkgray')
    ax.set_ylim(-60,60)
    ttl1=ax.set_title(titles[j],fontsize=25,fontdict={'verticalalignment': 'bottom'})
    ttls.append(ttl1)

dlon = data_all_out['sstds_out'].lon.values
dlon[np.where(dlon<0)] = dlon[np.where(dlon<0)]+360
#
Leddy_dum = spatial_filter(Leddy_data['var_grid'],4)
Leddy_dum[np.where(abs(Leddy_data['grid_y'])<5)]=np.nan
for j,var in enumerate(['sst','sst_model']):
    data_plot = np.nanmedian(data_ens[var+'_10']['S_out'],axis=0)
    jj = np.where(data_ens[var+'_10']['dgs']<=1.)[0][-1]
    for j2 in range(jj):
        if j2==0:
           std_mask = (data_ens[var+'_10_StoN'][j2,]<std_lim)
        else:
           std_mask = std_mask+(data_ens[var+'_10_StoN'][j2,]<std_lim)
    #
    if j==0:
        s_mask=std_mask.copy()
    else:
        s_mask=s_mask+std_mask
    #
    data_plot[np.where(std_mask)]=np.nan #mask regions that we don't trust
    dumX,dumXlon = add_cyclic_point(data_plot,coord=data_ens[var+'_10']['lon'][0,:])
    cm0=axes[j].pcolormesh(dumXlon, data_ens[var+'_10']['lat'][:,0], dumX, cmap=cmap3, norm=norm3, rasterized=True, transform=ccrs.PlateCarree())
    CS=axes[j].contour(Leddy_data['grid_x'],Leddy_data['grid_y'],Leddy_dum,colors='k',levels=np.array([25,50,75,100,150,200]),transform=ccrs.PlateCarree())
    axes[j].clabel(CS, inline=1, fontsize=10, fmt='%1.0f')
    
#
cax=fig3.add_axes([0.12,0.05,0.55,0.02])
cbar=plt.colorbar(mappable=cm0,cax=cax,orientation='horizontal')
clab=cbar.ax.set_xlabel('Slope $n$',fontsize=25,labelpad=15)
cbar.set_ticks(levels3)
cbar.set_ticklabels(levels3_str)
#
fig3.subplots_adjust(wspace=0.1)
#
#
jinds,iinds=np.where(s_mask)
ll=[]
for v,var in enumerate(['sst','sst_model']):
    #
    S_out2 = data_ens[var+'_10']['S_out'][:].copy()
    S_out2[:,jinds,iinds] = np.nan
    l1,   = ax3.plot(np.nanmedian(np.nanmedian(S_out2,-1),0),data_ens[var+'_10']['lat'][:,0],color='C'+str(v),lw=2)
    ax3.fill_betweenx(data_ens[var+'_10']['lat'][:,0],np.nanpercentile(np.nanmedian(S_out2,-1),5,axis=0),np.nanpercentile(np.nanmedian(S_out2,-1),95,axis=0), color='C'+str(v),alpha=0.3)
    ax3.fill_betweenx(data_ens[var+'_10']['lat'][:,0],np.nanpercentile(np.nanmedian(S_out2,0),25,axis=1),np.nanpercentile(np.nanmedian(S_out2,0),75,axis=-1), color='C'+str(v),alpha=0.3)
    ll.append(l1)

for n in [2/3,1,4/3]:
    ax3.axvline(x=n,color='gray',ls='--',lw=1.5)

ylab = ax3.set_ylabel('Latitude [$\degree$]', fontsize=25)
xlab = ax3.set_xlabel('Slope $n$', fontsize=25)
ax3.yaxis.set_label_position('right')
ax3.set_xlim(-0.25,2)
ax3.legend(ll,titles,fontsize=18,handlelength=.75, handletextpad=0.5,loc=6,borderaxespad=.1)
extra_artists=[clab,xlab,ylab]; extra_artists.extend(ttls)
for j,ax in enumerate(axes):
    txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=20)
    extra_artists.append(txt1)
    labs = ax.get_ymajorticklabels()
    extra_artists.extend(list(labs))

plt.savefig(plotpath+'Figure4_'+version+'.png',format='png',dpi=300,pad_inches=1,bbox_inches='tight',bbox_extra_artists=extra_artists)
#
##################################################
#  Figure 5  - parameterized maps and zonal mean #
##################################################
fig4, ax1, ax2, ax3, ax4 = map_and_zonal_mean()
axes=[ax1,ax2,ax3,ax4];
ttls=[]
titles=['$\kappa_{s}$','$\kappa_{l1}$','$\kappa_{l2}$']
for j,ax in  enumerate([ax1,ax2,ax3]):
    ax.coastlines(resolution='50m',color='darkgray')
    ax.set_ylim(-60,60)
    ttl1=ax.set_title(titles[j],fontsize=25,fontdict={'verticalalignment': 'bottom'})
    ttls.append(ttl1)

dlon = ds_out_2deg.lon.values
dlat = ds_out_2deg.lat.values
#
dlon[np.where(dlon<0)] = dlon[np.where(dlon<0)]+360
#shear
for j,data_plot in enumerate([data_out_ss[2,0,],data_out_ll[2,0,],data_out_ll2[2,0,]]):
    dumX,dumXlon = add_cyclic_point(data_plot,coord=dlon)
    cm0=axes[j].pcolormesh(dumXlon, dlat, [0.16,0.08,0.04][j]*dumX+[206,117,60][j], cmap=cmapX, norm=normX, rasterized=True, transform=ccrs.PlateCarree())

KE_tot_med = np.nanmedian(SmallScale['KEtot'][:,1,:,:],axis=0)
dumX = regridder_2deg(KE_tot_med)
dumX,dumXlon = add_cyclic_point(dumX,coord=lon_out)
for j, ax in enumerate(axes[:3]):
    ax.contour(dumXlon,lat_out, dumX,levels=np.array([0.25]),colors='r',rasterized=True, transform=ccrs.PlateCarree())
#
cax=fig4.add_axes([0.17,0.05,0.5,0.02])
cbar=plt.colorbar(mappable=cm0,cax=cax,orientation='horizontal') #,ticks=[0,3])
#
clab=cbar.ax.set_xlabel('Diffusivity [m$^2$ s$^{-1}$]',fontsize=25,labelpad=15)
fig4.subplots_adjust(wspace=-0.1)
#
ll=[]
for v,var in enumerate([data_out_ss,data_out_ll,data_out_ll2]):
    dum=[0.16,0.08,0.04][v]*var[2,0,:,:]+[206,117,60][v]
    dum[np.where(dum<0)]=np.nan
    dum = np.nanpercentile(dum,[25,50,75],axis=1)
    ninds = np.where(np.isfinite(dum[1,:]))[0]
    l1,= ax4.semilogx(dum[1,ninds],dlat[ninds],color='C'+str(v+3),lw=2)
    ll.append(l1)
    ax4.fill_betweenx(dlat[ninds],dum[0,ninds],dum[2,ninds],color='C'+str(v+3),alpha=0.3)

ylab = ax4.set_ylabel('Latitude [$\degree$]', fontsize=25)
xlab = ax4.set_xlabel('Diffusivity [m$^2$ s$^{-1}$]', fontsize=25)
ax4.set_xlim(3E1,4E3)
#
ax4.yaxis.set_label_position('right')
ax4.legend(ll,titles,fontsize=22,handlelength=.75, handletextpad=0.5)
extra_artists=[clab,xlab,ylab]; extra_artists.extend(ttls)
for j,ax in enumerate(axes):
    txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=20)
    extra_artists.append(txt1)

plt.savefig(plotpath+'Figure5_'+version+'.png',format='png',dpi=200,bbox_inches='tight',bbox_extra_artists=extra_artists)

#
#
##########################################
# FIGURE 8 - MAP OF PARAMETERIZED SLOPES #
##########################################
fig5, ax1, ax2, ax3, ax4 = map_and_zonal_mean()
axes=[ax1,ax2,ax3,ax4];
ttls=[]
titles=['$\kappa_{s}$','$\kappa_{l1}$','$\kappa_{l2}$']
for j,ax in  enumerate([ax1,ax2,ax3]):
    ax.coastlines(resolution='50m',color='darkgray')
    ax.set_ylim(-60,60)
    ttl1=ax.set_title(titles[j],fontsize=25,fontdict={'verticalalignment': 'bottom'})
    ttls.append(ttl1)

dlon = ds_out_2deg.lon.values
dlat = ds_out_2deg.lat.values
dlon[np.where(dlon<0)] = dlon[np.where(dlon<0)]+360
#
for j,data_plot in enumerate([slopes_nHHd['SmallScale'][2,1,:,:],slopes_nHHd['LargeScale'][2,1,:,:],slopes_nHHd['LargeScale2'][2,1,:,:]]):
    dumX,dumXlon = add_cyclic_point(data_plot,coord=dlon)
    cm0=axes[j].pcolormesh(dumXlon, dlat, dumX, cmap=cmap3, norm=norm3, rasterized=True, transform=ccrs.PlateCarree())

#
cax=fig5.add_axes([0.17,0.05,0.5,0.02])
cbar=plt.colorbar(mappable=cm0,cax=cax,orientation='horizontal')
clab=cbar.ax.set_xlabel('Slope $n$',fontsize=25,labelpad=15)
cbar.set_ticks(levels3)
cbar.set_ticklabels(levels3_str)
#
fig5.subplots_adjust(wspace=-0.1)
jinds,iinds=np.where(np.isnan(slopes_nHHd['SmallScale'][2,1,:,:]+slopes_nHHd['LargeScale'][2,1,:,:]+slopes_nHHd['LargeScale2'][2,1,:,:]))
ll=[]
for v,var in enumerate(['SmallScale','LargeScale','LargeScale2']):
    dum=slopes_nHHd[var][2,1,:,:].copy()
    dum[jinds,iinds]=np.nan
    dum[np.where(dum<0)]=np.nan
    dum = np.nanpercentile(dum,[25,50,75],axis=1)
    ninds = np.where(np.isfinite(dum[1,:]))[0]
    l1,= ax4.plot(dum[1,ninds],dlat[ninds],color='C'+str(v+3),lw=2)
    ll.append(l1)
    ax4.fill_betweenx(dlat[ninds],dum[0,ninds],dum[2,ninds],color='C'+str(v+3),alpha=0.3)

for n in [2/3,1,4/3]:
    ax4.axvline(x=n,color='gray',ls='--',lw=1.5)

ylab = ax4.set_ylabel('Latitude [$\degree$]', fontsize=25)
xlab = ax4.set_xlabel('Slope $n$', fontsize=25)
ax4.set_xlim(-0.25,2)
ax4.yaxis.set_label_position('right')
ax4.legend(ll,titles,fontsize=22,handlelength=.75, handletextpad=0.5)
extra_artists=[clab,xlab,ylab]; extra_artists.extend(ttls)
for j,ax in enumerate(axes):
    txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=20)
    extra_artists.append(txt1)

plt.savefig(plotpath+'Figure8_'+version+'.png',format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
#
##########################
# PLOT THE ERROR METRICS #
# FIGURE 9 AND S1        #
##########################
#
p_contr={}
t_err={}
r_err={}
slope_linregress={}
#
for jnd,jinds in enumerate([jinds_tropics,jinds_midlats]):
  for l,ll in enumerate([data_out_ll,data_out_ll2]):
    for j,jj in enumerate([0.25,1.25]):
        print(j)
        a_range=np.concatenate([np.arange(0,10),np.arange(10,100,10),np.arange(100,1000,100)],axis=0)/1E3 
        b_range=np.concatenate([np.arange(0,10),np.arange(10,100,10),np.arange(100,1000,100)],axis=0)/1E3 
        j2=np.where(jj*3==dgs0*0.25)[0][0]
        #
        for l2, key in enumerate(['sst_model_10','sst_10']):
            pcon=np.ones((len(a_range),len(b_range)))
            terr=np.ones((len(a_range),len(b_range)))
            rerr=np.ones((len(a_range),len(b_range)))
            slope_d=np.ones((len(a_range),len(b_range)))
            j1 = np.where(jj==np.array(data_ens[key]['dgs']))[0][0]
            k0 = np.nanmedian(data_ens[key]['K_out'][j1,],axis=0)
            k0[np.where(data_ens[key+'_StoN'][j1,]<std_lim)]=np.nan
            #
            k0=k0[jinds,:]
            k01=ll[2,j2,jinds,:]
            k02=data_out_ss[2,j2,jinds,:]
            #mask outliers
            njinds,niinds=np.where(np.isfinite(k0+k01+k02))
            #
            print('number of points for correlation', len(njinds))
            for a,aa in enumerate(a_range):
                for b,bb in enumerate(b_range):
                    k1=aa*k01
                    k2=bb*k02
                    pcon[a,b]=np.nanmedian(k1/(k1+k2))
                    slope_dum,intercept_dum,r,p,stder_dum=stats.linregress((k1+k2)[njinds,niinds],k0[njinds,niinds])
                    #
                    terr[a,b]=p 
                    rerr[a,b]=r 
                    slope_d[a,b]=slope_dum
            #
            rerr[np.where(terr>0.05)] = np.nan
            p_contr[str(jj)+'_'+str(l)+'_'+str(l2)+'_'+str(jnd)] = pcon
            t_err[str(jj)+'_'+str(l)+'_'+str(l2)+'_'+str(jnd)]   = terr
            r_err[str(jj)+'_'+str(l)+'_'+str(l2)+'_'+str(jnd)]   = rerr
            slope_linregress[str(jj)+'_'+str(l)+'_'+str(l2)+'_'+str(jnd)]=slope_d

for l2,key in enumerate(['sst_model_10','sst']):
    fig,axes=plt.subplots(nrows=2,ncols=4,sharex=True,sharey=True,figsize=(20,10))
    keys=['0.25','1.25','0.25','1.25']
    titles=['0.5','2.5','0.5','2.5']
    j=0 #resolution
    for c,ax in enumerate(axes.flatten()):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.001,0.8)
        ax.set_ylim(0.001,0.5)
        if j<2:
           j2=0 #tropics
        else:
           j2=1 #midlats
        if c<4:
            l=0 #stirring
            ax.set_title(titles[c]+'$\degree$', fontsize=20)
        else:
            l=1 #integrated stirring
        #
        cm1=ax.contourf(branges2[keys[j]+'_'+str(l)],aranges2[keys[j]+'_'+str(l)],err1[keys[j]+'_'+str(l)+'_'+str(l2)+'_'+str(j2)],levels=levelsE,cmap=cmapE,norm=normE,extend='max')
        CS=ax.contour(b_range,a_range, 1E2*p_contr[keys[j]+'_'+str(l)+'_'+str(l2)+'_'+str(j2)],levels=1E2*np.arange(0,1.2,0.2),colors='C3')
        plt.clabel(CS, inline=1,inline_spacing=1, fmt='%1.0f',fontsize=14)
        CS=ax.contour(b_range,a_range,(slope_linregress[keys[j]+'_'+str(l)+'_'+str(l2)+'_'+str(j2)]),levels=np.array([0.9,1,1.1]),colors='k')
        plt.clabel(CS, inline=1,inline_spacing=1, fmt='%1.1f',fontsize=14)
        # the following sets reasonable contour intervals automatically
        r_err_levels=np.arange(0,1,0.05)
        r_err_max=np.floor((np.nanmax(r_err[keys[j]+'_'+str(l)+'_'+str(l2)+'_'+str(j2)])**2)*100)/100
        r_err_min=np.round(np.nanmin(r_err[keys[j]+'_'+str(l)+'_'+str(l2)+'_'+str(j2)])**2,2)
        if len(np.where(np.logical_and(r_err_levels<r_err_max,r_err_levels>r_err_min))[0])<=4:
            r_err_levels=r_err_levels[np.where(r_err_levels<r_err_max)[0]]
            r_err_levels=np.concatenate([r_err_levels,np.arange(r_err_levels[-1]+0.01,r_err_max+0.01,0.01)],axis=0)
        #
        linestyles='solid'
        CS=ax.contour(b_range,a_range,r_err[keys[j]+'_'+str(l)+'_'+str(l2)+'_'+str(j2)]**2, levels=r_err_levels,colors='w')
        manual_locations = [] # we will force the labels to be roughly in the middle of the line
        for col in CS.collections:
            if len(col.get_paths())>0:
                dum = col.get_paths()[0].vertices
                if dum.shape[0]<=100:
                    iloc1,jloc1 = dum[dum.shape[0]//4,:]
                    manual_locations.append((iloc1,jloc1))
                elif dum.shape[0]>100:
                    iloc1,jloc1 = dum[dum.shape[0]//2,:]
                    iloc2,jloc2 = dum[dum.shape[0]//4,:]
                    manual_locations.append((iloc1,jloc1))
                    manual_locations.append((iloc2,jloc2))
        #
        plt.clabel(CS, inline=1,inline_spacing=1,manual=manual_locations, fmt='%1.2f',fontsize=14)
        j=j+1
        if j>3:
            j=0 
        #
    ylab1=axes.flatten()[0].set_ylabel('$||\mathbf{K}_{l1}||$', fontsize=25,labelpad=45)    
    ylab2=axes.flatten()[4].set_ylabel('$||\mathbf{K}_{l2}||$', fontsize=25,labelpad=45)
    #
    ttl1=fig.text(0.3,0.96, 'Tropics',ha='center',va='center', fontsize=25,color='C1')
    ttl2=fig.text(0.7,0.96, 'Midlatitudes',ha='center',va='center', fontsize=25,color='C0')    
    xlab=fig.text(0.5,0.02, 'Small scale stirring weight ($b_1$)',ha='center',va='center', fontsize=25)
    ylab=fig.text(0.07,0.5, 'Large scale shear weight ($b_2$)',ha='center',va='center',rotation='vertical', fontsize=25)
    cax=fig.add_axes([0.93,0.1,0.02,0.77])
    cbar=plt.colorbar(mappable=cm1,cax=cax)
    #
    clab=cbar.ax.set_ylabel('Mean Absolute Percentage Error',fontsize=25,labelpad=15)
    fig.subplots_adjust(wspace=0.1,hspace=0.1)
    extra_artists=[ttl1,ttl2,xlab,ylab,ylab1,ylab2,clab]
    for j,ax in enumerate(axes.flatten()):
        txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=20)
        extra_artists.append(txt1)
    
    plotname='Figure9_'+version+'_'+key+'_combined.png'
    plt.savefig(plotpath+plotname,format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)

#########################################
# FIGURE 3 - Slopes in selected regions #
#########################################
# 
variables=['sst','sst_model']
ttl_strings=['$\kappa_{SSTA}$','$\kappa_{MITgcm}$']
markers=['o','d','s','D','v','P','s','h','p']
#
projection1=ccrs.Mercator(central_longitude=200)
#
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
# spans two rows:
ax4 = fig.add_subplot(gs[0, :],projection=projection1)
#
axes = [ax1,ax2,ax4]
#
lonlims={}
latlims={}
for v,var in enumerate(variables):
    dgs=data_all_out[var+'_dgs'].copy()
    dlon=data_all_out[var+'ds_out'].lon.values
    dlat=data_all_out[var+'ds_out'].lat.values
    dlon[np.where(dlon>180)]=dlon[np.where(dlon>180)]-360
    rlat=np.radians(dlat)
    ax=axes[v]
    #
    kxs=np.zeros((len(dgs),7))
    xs=np.zeros((len(dgs),7))
    #
    data_out=data_all_out[var][2,]
    #
    pers = [5,25,50,75,95]
    data_ens_out5  = np.swapaxes(data_ens[var+'_5']['K_out'], 0,1)
    data_ens_out10 = np.swapaxes(data_ens[var+'_10']['K_out'], 0,1)
    data_ens_out5[np.where(data_ens_out5<0)]=np.nan
    data_ens_out10[np.where(data_ens_out10<0)]=np.nan
    ens = data_ens_out5.shape[0]
    kxs_5=np.ones((len(dgs),7,len(pers)))*np.nan
    kxs_10=np.ones((len(dgs),7,len(pers)))*np.nan
    kxs_all=np.ones((len(dgs),7,2*data_ens_out10.shape[0]))*np.nan
    #
    # ['1) TROPICAL PACIFIC', '2) NORTHERN PACIFIC', '3) NORTH ATLANTIC', '4) TROPICAL INDIAN OCEAN', '5) TROPICAL ATLANTIC', '6) SOUTHERN ATLANTIC', '7) SOUTHERN INDIAN OCEAN']
    lonlims['1']=[-160,-100]
    latlims['1']=[-10,10]
    latlims['2']=[35,45]
    lonlims['2']=[150,180]
    latlims['3']=[30,38]
    lonlims['3']=[-70,-50]
    latlims['4']=[-5,10]
    lonlims['4']=[40,60]
    latlims['5']=[-10,10]
    lonlims['5']=[-40,-10]
    latlims['6']=[-50,-35]
    lonlims['6']=[-50,-10]
    latlims['7']=[-40,-20]
    lonlims['7']=[40,100]
    #
    for j,dg in enumerate(dgs):
        if var in ['sst']:
            jinds,iinds=np.where(data_ens[key+'_StoN'][j,]<std_lim)
            data_out[j,jinds,iinds]=np.nan
        #
        kxs_dum = []
        kxs_5_dum = []
        kxs_10_dum = []
        lxs_dum = []
        Leddy_dum = []
        #
        for k, key in enumerate(lonlims.keys()):
            jind1 = np.where(np.logical_and(dlat>latlims[key][0],dlat<latlims[key][1]))[0]
            iind1 = np.where(np.logical_and(dlon>lonlims[key][0],dlon<lonlims[key][1]))[0]
            lx1   = np.tile(111E3*dg*np.cos(rlat[jind1]),(len(iind1),1)).T
            kx1   = smart_wmean(data_out[j,jind1][:,iind1], lx1)
            #
            jind2 = np.where(np.logical_and(Leddy_data['grid_y'][:,0]>latlims[key][0],Leddy_data['grid_y'][:,0]<latlims[key][1]))[0]
            iind2 = np.where(np.logical_and(Leddy_data['grid_x'][0,:]>lonlims[key][0],Leddy_data['grid_x'][0,:]<lonlims[key][1]))[0]
            Leddy_dum.append(1)
            #
            if dg in data_ens[var+'_5']['dgs']:
                jj = np.where(dg==data_ens[var+'_5']['dgs'])[0][0]
                kx1_5  = np.nansum(np.reshape(data_ens_out5[:,jj,jind1,][:,:,iind1]*lx1,(ens,-1)),-1)/np.nansum(np.reshape(np.isfinite(data_ens_out5[:,jj,jind1,][:,:,iind1]).astype(np.float)*lx1,(ens,-1)),-1)
                kx1_10 = np.nansum(np.reshape(data_ens_out10[:,jj,jind1,][:,:,iind1]*lx1,(ens,-1)),-1)/np.nansum(np.reshape(np.isfinite(data_ens_out10[:,jj,jind1,][:,:,iind1]).astype(np.float)*lx1,(ens,-1)),-1)
            #
            kxs_dum.append(kx1)
            kxs_5_dum.append(kx1_5)
            kxs_10_dum.append(kx1_10)
            lxs_dum.append(np.nanmean(lx1))
        #################################
        kxs[j,:]       = np.array(kxs_dum)
        xs[j,:]        = 2*np.array(lxs_dum)
        kxs_all[j,:,:] = np.concatenate([np.array(kxs_10_dum), np.array(kxs_5_dum)],axis=-1)
        if dg in data_ens[var+'_5']['dgs']:
            kxs_5[j,]  = np.nanpercentile(np.array(kxs_5_dum),pers,axis=-1).T
            kxs_10[j,] = np.nanpercentile(np.array(kxs_10_dum),pers,axis=-1).T
    #
    for j in range(7):
        dumx=kxs[:,j]
        dumx[np.where(dumx<10)]=np.nan
        dinx=np.where(np.isnan(dumx))[0]
        if len(dinx)>0:
            dumx[dinx[0]:]=np.nan
        #
        nonnaninds = np.where(np.isfinite(kxs_10[:,j,2]))[0]
        ld, = ax.loglog(xs[nonnaninds,j]/Leddy_dum[j],kxs_10[nonnaninds,j,2],"-",color='C'+str(j))
        for k in range(len(dgs)):
            ax.plot(np.array([xs[k,j],xs[k,j]])/Leddy_dum[j], np.array([kxs_10[k,j,1],kxs_10[k,j,-2]]),lw=2,color='C'+str(j))
            ax.plot(np.array([xs[k,j],xs[k,j]])/Leddy_dum[j], np.array([kxs_10[k,j,0],kxs_10[k,j,-1]]),lw=2,ls='--',color='C'+str(j))
            ax.plot(np.array([xs[k,j],xs[k,j]])/Leddy_dum[j], np.array([kxs_5[k,j,1],kxs_5[k,j,-2]]),lw=1,color='C'+str(j))
            ax.plot(np.array([xs[k,j],xs[k,j]])/Leddy_dum[j], np.array([kxs_5[k,j,0],kxs_5[k,j,-1]]),lw=1,ls='--',color='C'+str(j))
    #
    l8,   = ax.loglog(np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7])/Leddy_dum[j], (0.005*np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]))**(4/3), lw=1, color='k')
    l9,   = ax.loglog(np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7])/Leddy_dum[j], (0.05*np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]))**(1)   , lw=1, color='k', ls=':')
    l10,  = ax.loglog(np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7])/Leddy_dum[j], (0.5*np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]))**(2/3) , lw=1, color='k', ls='--')
    #
    labels = ['$\kappa_{Trop. Pa.}$','$\kappa_{Trop. Atl.}$','$\kappa_{Trop. Ind.}$','$\kappa_{N. Pa.}$','$\kappa_{N. Atl.}$','$\kappa_{S. Atl.}$','$\kappa_{S. Ind.}$','$l^{4/3}$','$l$','$l^{2/3}$']
    if v==0:
       xlab = fig.text(0.5,0.02,'Length Scale $l$ [km]', fontsize=25,ha='center', va='center')
       ylab = fig.text(0.01,0.25, 'Diffusivity [m$^2$ s$^{-1}$]', fontsize=25, rotation='vertical', ha='center', va='center')
       ax1.legend([l8,l9,l10],['n=4/3','n=1','n=2/3'])
    #
    ttl  = ax.set_title(ttl_strings[v],fontsize=20)
    xticks=np.array([4E4,6E4,1E5,2E5,6E5])
    xticlabs=[]
    for xt in xticks:
        xticlabs.append(format(xt/1E3,'1.0f'))
    #
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticlabs)
    ax.set_ylim(1E2,5E5)
    ax.set_xlim(3E4,14E5)
    labels2=['1) TROPICAL PACIFIC', '2) NORTHERN PACIFIC', '3) NORTH ATLANTIC', '4) TROPICAL INDIAN OCEAN', '5) TROPICAL ATLANTIC', '6) SOUTHERN ATLANTIC', '7) SOUTHERN INDIAN OCEAN']
    for j in range(7):
        ms_all=np.ones(kxs_all.shape[-1])*np.nan
        for jj in range(kxs_all.shape[-1]):
            nonnaninds = np.where(np.isfinite(kxs_all[:6,j,jj]))[0]; 
            ms_all[jj],mi,ls,hs=stats.theilslopes(np.log10(kxs_all[:6,j,jj][nonnaninds]),np.log10(xs[:6,j][nonnaninds]));
        ls,ms,hs = np.nanpercentile(ms_all,[5,50,95])
        print(labels2[j],np.round(ls,decimals=2),np.round(ms,decimals=2),np.round(hs,decimals=2))

# map
ax4.add_feature(cfeature.LAND)
ax4.gridlines()
ax4.set_ylim(-55,55)
for j,key in enumerate(lonlims.keys()):
    ax4.add_patch(mpl.patches.Rectangle((lonlims[key][0],latlims[key][0]),np.diff(lonlims[key]),np.diff(latlims[key]),fill=True,color='C'+str(j),transform=ccrs.PlateCarree(),zorder=0))

ax4.coastlines(resolution='50m',color='darkgray')
ax4.set_extent([-180, 180, -55, 55], ccrs.PlateCarree())
#
txt1=axes[2].text(0.0, 1.02, string.ascii_lowercase[0],transform=axes[2].transAxes, fontsize=20)
txt2=axes[0].text(0.0, 1.02, string.ascii_lowercase[1],transform=axes[0].transAxes, fontsize=20)
txt3=axes[1].text(0.0, 1.02, string.ascii_lowercase[2],transform=axes[1].transAxes, fontsize=20)
#
ax2.set_yticklabels([])
#
fig.subplots_adjust(wspace=0.1,hspace=0.1)
plotname='Figure3_'+version+'.png'
plt.savefig(plotpath+plotname,format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=[txt1,txt2,txt3,xlab,ylab,ttl])
plt.close('all')

print(plotname+' DONE!')

#######################################################
# FIG 7 - SAME AS FIG 3 BUT FOR THE PARAMETERIZATIONS #
#######################################################
projection1=ccrs.Mercator(central_longitude=200)
#
fig,axes = plt.subplots(nrows=3,ncols=1,sharex=True,sharey=True,figsize=(5,15))
#
titles=['$\kappa_{s}$','$\kappa_{l1}$','$\kappa_{l2}$']
#
for v,data_out in enumerate([k30,Km_tot,Km_tot5]):
    dgs=dgs0*0.25/3
    dlon=lon2.copy()
    dlat=lat2.copy()
    dlon[np.where(dlon>180)]=dlon[np.where(dlon>180)]-360
    rlat=np.radians(dlat)
    ax=axes[v]
    kxs=np.zeros((len(dgs),7))
    xs=np.zeros((len(dgs),7))
    #
    for j,dg in enumerate(dgs):
        kxs_dum = []
        lxs_dum = []
        #
        for k, key in enumerate(lonlims.keys()):
            jind1 = np.where(np.logical_and(dlat[:,0]>latlims[key][0],dlat[:,0]<latlims[key][1]))[0]
            iind1 = np.where(np.logical_and(dlon[0,:]>lonlims[key][0],dlon[0,:]<lonlims[key][1]))[0]
            lx1   = 111E3*dg*np.cos(rlat[jind1,:][:,iind1])
            kx1   = smart_wmean(data_out[j,jind1][:,iind1], lx1)
            #
            kxs_dum.append(kx1)
            lxs_dum.append(np.nanmean(lx1))
        #print(kxs_dum)
        #################################
        kxs[j,:]   = np.array(kxs_dum)
        xs[j,:]    = 2*np.array(lxs_dum)
    #
    for j in range(7):
        dumx=kxs[:,j]
        dumx[np.where(dumx<10)]=np.nan
        ld, = ax.loglog(xs[:,j],[0.16,0.08,0.04][v]*kxs[:,j]+[206,117,60][v],"-",color='C'+str(j))
    #
    l8,   = ax.loglog(np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]), (0.0025*np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]))**(4/3), lw=1, color='k')
    l9,   = ax.loglog(np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]), (0.025*np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]))**(1)   , lw=1, color='k', ls=':')
    l10,  = ax.loglog(np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]), (0.25*np.array([.5E2,1E2,1E3,1E4,1E5,1E6,1E7]))**(2/3) , lw=1, color='k', ls='--')
    #
    labels = ['$\kappa_{Trop. Pa.}$','$\kappa_{Trop. Atl.}$','$\kappa_{Trop. Ind.}$','$\kappa_{N. Pa.}$','$\kappa_{N. Atl.}$','$\kappa_{S. Atl.}$','$\kappa_{S. Ind.}$','$l^{4/3}$','$l$','$l^{2/3}$']
    if v==2:
       xlab = ax.set_xlabel('Length Scale $l$ [km]', fontsize=25)
    if v==0:
       ylab=fig.text(-0.1,0.5, 'Diffusivity [m$^2$ s$^{-1}$]', fontsize=25, rotation='vertical', ha='center', va='center')
       ax.legend([l8,l9,l10],['n=4/3','n=1','n=2/3'])
    #
    ttl  = ax.set_title(titles[v],fontsize=20)
    #
    xticks=np.array([4E4,6E4,1E5,2E5,6E5])
    xticlabs=[]
    for xt in xticks:
        xticlabs.append(format(xt/1E3,'1.0f'))
    #
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticlabs)
    ax.set_xlim(3E4,14E5)
    ax.set_ylim(1E2,5E5)

#
txt1=axes[0].text(0.0, 1.02, string.ascii_lowercase[0],transform=axes[0].transAxes, fontsize=20)
txt2=axes[1].text(0.0, 1.02, string.ascii_lowercase[1],transform=axes[1].transAxes, fontsize=20)
txt3=axes[2].text(0.0, 1.02, string.ascii_lowercase[2],transform=axes[2].transAxes, fontsize=20)
#
fig.subplots_adjust(wspace=0.1,hspace=0.15)
plotname='Figure7_'+version+'.png'
plt.savefig(plotpath+plotname,format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=[txt1,txt2,txt3,xlab,ylab,ttl])
plt.close('all')
#
print(plotname+' DONE!')
#
##########################
# FIGURE 6 - CORRELATION #
##########################
# 
plt.close('all')
#
levelsL=np.array([0,5,10,15,20,30,40,50,60])
cmap0=mpl.cm.viridis
cmlist=[];
for cl in np.linspace(0,252,len(levelsL)): cmlist.append(int(cl))
cmapL, normL = from_levels_and_colors(levelsL,cmap0(cmlist),extend='max');
#
fig,axes = plt.subplots(nrows=2,ncols=3,sharey=True,sharex=True,figsize=(15,10))
extra_artists=[]
titles1=['$\kappa_{SSTA}$','$\kappa_{MITgcm}$']
titles2=['$\kappa_{s}$','$\kappa_{l1}$','$\kappa_{l2}$']
#
for ax in axes.flatten():
    ax.plot(np.arange(0,6E3,1E3),np.arange(0,6E3,1E3),color='gray',lw=2,ls='--',zorder=0)

for v,varname in enumerate(['sst','sst_model']):
    dum0 = data_all_out[varname][2,0,].flatten()
    dum1 = data_out_ss[2,0,].flatten() #*50
    dum2 = data_out_ll[2,0,].flatten() #/4
    dum3 = data_out_ll2[2,0,].flatten() #/20
    #
    ninds = np.where(np.isfinite(dum0+dum1+dum2+dum3))[0]
    #
    v2=0
    for j,dum in enumerate([dum1,dum2,dum3]):
        print(j)
        ax=axes[v,j]
        ttl1=ax.set_title(titles1[v]+' vs '+titles2[v2],fontsize=20)
        medslope0,medintercept0,lo_slope,up_slope = stats.theilslopes(dum0[ninds],dum[ninds])
        print(medslope0,medintercept0)
        cm1  = ax.scatter(dum0[ninds],dum[ninds]*medslope0+medintercept0,s=10,c=abs(lat3.flatten()[ninds]),cmap=cmapL,norm=normL,marker='.')
        for l,lrange in enumerate([[0,90],[0,15],[15,30],[30,60]]):
            rinds=ninds[np.where(np.logical_and(abs(lat3.flatten()[ninds])<max(lrange),abs(lat3.flatten()[ninds])>=min(lrange)))]
            xmed = np.nanmedian(dum0[rinds])
            ymed = np.nanmedian(dum[rinds])
            Sl,Ic,Pr = monte_carlo_r(dum0[rinds],dum[rinds],n2=0.95,nn=500)
            ax.plot(np.arange(0,2E3,1E2),Ic[1]+Sl[1]*np.arange(0,2E3,1E2),color='C'+str(l))
            ax.text(0.7,0.05+0.06*l,'r$^2_{'+str(lrange[0])+'-'+str(lrange[1])+'}$ = '+str(np.round(Pr[1]**2,decimals=2)),fontdict={'color':'C'+str(l)},transform=ax.transAxes)
        #
        txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=20)
        v2=v2+1
        extra_artists.extend([txt1,ttl1])

ax.set_ylim(0,4E3)
ax.set_xlim(0,4E3)
xlab=fig.text(0.5,0.05,'MicroInverse Diffusivity [m$^2$ s$^{-1}$]',fontsize=20,ha='center',va='center')
ylab=fig.text(0.05,0.5,'Parameterized Diffusivity [m$^2$ s$^{-1}$]',rotation='vertical',fontsize=20,ha='center',va='center')
#
cax  = fig.add_axes([0.94,0.1,0.02,0.8])
cbar = plt.colorbar(mappable=cm1,cax=cax,orientation='vertical')
clab = cbar.ax.set_ylabel('|Latitude| [$\degree$]',fontsize=25,labelpad=15)
#
extra_artists.extend([xlab,ylab,xlab])
#
plt.savefig(plotpath+'Figure6_'+version+'.png',format='png',dpi=150,bbox_inches='tight',bbox_extra_artists=extra_artists)
#
