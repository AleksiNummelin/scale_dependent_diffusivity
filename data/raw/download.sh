###########################
# Author: Aleksi Nummelin
# Year:   2020
#
###################
# DOWNLOAD OI-SST #
###################
wget ftp://ftp2.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.*.nc
#
################################
# GLOBCURRENT SURFACE CURRENTS #
################################
#
# This product requires the user to log in to the
# European Copernicus Marine Services website.
# The direct link to the data set is
# https://resources.marine.copernicus.eu/?option=com_csw&view=details&product_id=MULTIOBS_GLO_PHY_REP_015_004
# 1) Choose 'download product'
# 2) Log in and choose the daily frequency at depth 0 (surface)
# 3) Download using your preferred download method
#
#########################
# EDDY TRAJECTORY ATLAS #
#########################
#
# This product requires creating credentials to AVISO
# The product webpage (with very useful information) is at
#
# https://www.aviso.altimetry.fr/en/data/products/value-added-products/global-mesoscale-eddy-trajectory-product.html
#
# From there you can follow the link to the delayed time trajectory atlas - as of writing (2020) ftp download is available 
#
############################
# MITGCM TRACER SIMULATIONS
###########################
#
# Contact RYAN ABERNATHEY and/or JULIUS BUSECKE at the LDEO ocean transport group (https://ocean-transport.github.io/people.html)
# for access to a tracer simulation called run_sst_AMSRE_damped_1m.
#
# If you have access to the JHU computational infrastructure, the dataset can be accessed there as well.
#
# Once you have access to the data the postprocessing assumes that the data is at raw/run_sst_AMSRE_damped_1m/
