Scale_Dependent_Diffusivity
==============================
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)

Code for reproducing Nummelin et al. (2020) analysis.

Workflow:

1) Go to data/raw/ and follow the instructions in the download.sh in order to download the data.
2) Go to side_packages and use the fetch.sh MicroInverse and naturalHHD packages from GitHub.
3) Once the data is downloaded and the required packages are in place, run the following
   python bin/process_SST.py                # Produce low-pass filtered SST
   python bin/process_eddy_atlas.py         # Bin the Eddy trajectories to a grid
   python bin/run_inversion.py              # Run the inversion
   python bin/run_HelmHoltzDecomposition.py # Run the Helmholtz-Hodge decomposition
4) At this point all the major data processing is done.
   The final step is to do some postprocessing and plot the figures in Nummelin et al. 2020.
   python bin/run_plotting.py

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
