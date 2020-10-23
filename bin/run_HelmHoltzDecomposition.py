################################################
#
# AUTHOR: Aleksi Nummelin
# Year:   2016-2020
#
# THIS SCRIPT WILL PROCESS THE VELOCITY DATA
# AND PERFORM THE CALCULATIONS NEEDED FOR
# PARAMTERIZATIONS OF SUB-GRIDSCALE MIXING
# AND LAGE SCALE SHEAR DRIVEN SUB-GRIDSCALE
# STIRRING.
#
###############################################
#
# CALCULATE THE HELMHOLTZ-H0DGE DECOMPOSITION OF WEEKLY MEAN VELOCITIES
exec(open('HelmHoltzDecomposition.py').read())
#
# SMOOTH THE DECOMPOSED VELOCITY FIELD AND CALCULATE
# A NUMBER OF DIFFERENT QUANTITITIES GIVEN THE SMOOTHED
# FIELDS (SHEAR, STRAIN, STRESS, KE ETC.)
exec(open('HelmHoltzDecomposition_analysis.py').read())