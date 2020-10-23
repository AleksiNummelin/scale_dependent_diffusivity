###########################
#
# AUTHOR: ALEKSI NUMMELIN
# YEAR:   2018-2020
#
# THIS SCRIPT WILL LOAD
# THE ENSEMBLE INVERSION DATA
# FOR MAKING THE FINAL FIGURES
# FOR NUMMELIN ET AL. 2020.
#
##################################################
# LOAD ENSEMBLE INVERSION 
#
fpath    = '..data/processed/'
data_ens = {}
if frobenius:
    data_ens['sst_model_5']  = np.load(fpath+'MITgcm_results_5_patch_v2.npz')
    data_ens['sst_model_10'] = np.load(fpath+'MITgcm_results_10_patch_v2.npz')
    data_ens['sst_5']        = np.load(fpath+'SST_results_5_patch_v2.npz')
    data_ens['sst_10']       = np.load(fpath+'SST_results_10_patch_v2.npz')
else:
    data_ens['sst_model_5']  = np.load(fpath+'MITgcm_results_5_patch.npz')
    data_ens['sst_model_10'] = np.load(fpath+'MITgcm_results_10_patch.npz')
    data_ens['sst_5']        = np.load(fpath+'SST_results_5_patch.npz')
    data_ens['sst_10']       = np.load(fpath+'SST_results_10_patch.npz')
#
# CALCULATE SIGNAL TO NOISE RATIO
for var in ['sst_model','sst']:
    data_ens[var+'_10_StoN'] = hutils.spatial_filter(np.nanmedian(data_ens[var+'_10']['K_out'],1)/np.nanstd(data_ens[var+'_10']['K_out'],1))
    data_ens[var+'_5_StoN']  = hutils.spatial_filter(np.nanmedian(data_ens[var+'_5']['K_out'],1)/np.nanstd(data_ens[var+'_5']['K_out'],1))


