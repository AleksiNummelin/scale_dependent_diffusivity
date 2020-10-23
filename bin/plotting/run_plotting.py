import xarray as xr
from xmitgcm import open_mdsdataset
import xesmf as xe
from dask.distributed import Client, LocalCluster
#
import numpy as np
from scipy import spatial,signal,stats,interpolate,integrate
from scipy.optimize import curve_fit
#
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import os
import sys
sys.path.append('/home/anummel1/Projects/MicroInv/MicroInverse/')
from MicroInverse import MicroInverse_utils as mutils
# no interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
#
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point
import string
#
sys.path.append('../')
import HelmHoltzDecomposition_analysis_utils as hutils
#
frobenius=False #whether to use a frobenius norm or not
version = 'v6'
#
cluster = LocalCluster(n_workers=4)
client = Client(cluster)
#
exec(open('load_nhhd.py').read())
#
exec(open('load_full_inversion.py').read())
#
exec(open('load_ensemble_inversion.py').read())
#
exec(open('preprocess_for_figures.py').read())
#
exec(open('figures.py').read())
#
# DONE!
