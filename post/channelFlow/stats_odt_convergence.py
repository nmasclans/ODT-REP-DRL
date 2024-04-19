# Description: 
# Compute ODT mean and rmsf velocity profiles and reynolds stresses
# Plot results for increasing averaging time, to observe profiles convergence

# Usage
# python3 stats_odt_convergence.py [case_name] [reynolds_number] [delta_time_stats]

# Arguments:
# case_name (str): Name of the case
# reynolds_number (int): reynolds number of the odt case, to get comparable dns result.
# delta_time_stats (int): delta time (seconds) between averaged profiles

# Example Usage:
# python3 stats_odt_convergence.py channel180 180 25

# Comments:
# Values are in wall units (y+, u+) for both ODT and DNS results,
# Scaling is done in the input file (not explicitly here).

## todo: change many things of this script and related functions to incorporate the new implementation of 
## statistics in a fine uniform grid, and saved in stat_dmp_*****.dat

import numpy as np
import glob as gb
import yaml
import sys
import os
from scipy.interpolate import interp1d

from ChannelVisualizer import ChannelVisualizer
from utils import *

eps   = 1e-8 # small number for using np.arange 

#--------------------------------------------------------------------------------------------

#------------ input parameters ---------------

try :
    caseN     = sys.argv[1]
    rlzN      = int(sys.argv[2])
    Retau     = int(sys.argv[3])
    dtAvg     = float(sys.argv[4])
    tBeginAvg = float(sys.argv[5])
    tEndAvg   = float(sys.argv[6])
    print(f"Script parameters: \n- Case name: {caseN} \n- Realization Number: {rlzN} \n- Retau: {Retau} \n- dt averaging: {dtAvg} \n- Time Begin Averaging: {tBeginAvg} \n- Time End Averaging: {tEndAvg} \n")
except :
    raise ValueError("Missing call arguments, should be: <case_name> <realization_number> <reynolds_number> <delta_time_stats> <time_begin_averaging> <time_end_averaging>")

# post-processing directory
postDir = f"../../data/{caseN}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)
# post-processing sub-directory for single realization
rlzStr = f"{rlzN:05d}"
postRlzDir = os.path.join(postDir, f"post_{rlzStr}")
if not os.path.exists(postRlzDir):
    os.mkdir(postRlzDir)

#------------ ODT data ---------------

# --- Get ODT input parameters ---

odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho   = yml["params"]["rho0"]
dxmin = yml["params"]["dxmin"]
nunif = yml["params"]["nunif"]
domainLength = yml["params"]["domainLength"] 
delta = domainLength * 0.5
utau  = 1.0
dTimeStart  = yml["dumpTimesGen"]["dTimeStart"]
dTimeEnd    = get_effective_dTimeEnd(caseN, rlzStr) # dTimeEnd = yml["dumpTimesGen"]["dTimeEnd"] can lead to errors if dTimeEnd > tEnd
dTimeStep   = yml["dumpTimesGen"]["dTimeStep"]
assert tEndAvg <= dTimeEnd, "Averaging end time for calculations and plots must be <= dTimeEnd and/or tEnd."
inputParams = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif":nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
               "caseN": caseN, "rlzStr": rlzStr,
               "dTimeStart": dTimeStart, "dTimeEnd": dTimeEnd, "dTimeStep": dTimeStep} 

# --- Chosen averaging times ---

if tBeginAvg >= dTimeStart:
    averaging_times = np.arange(tBeginAvg, tEndAvg+1e-4, dtAvg).round(4)
else:
    averaging_times = np.arange(dTimeStart, tEndAvg+1e-4, dtAvg).round(4)
averaging_times_plots = averaging_times - tBeginAvg

# --- Get ODT computational data ---

### (ydelta_post, yplus_post, 
###  um_post, urmsf_post, vm_post, vrmsf_post, wm_post, wrmsf_post, 
###  ufufm_post, vfvfm_post, wfwfm_post, ufvfm_post, ufwfm_post, vfwfm_post) \
###     = get_odt_statistics_post_at_chosen_averaging_times(inputParams, averaging_times)

#------------ ODT statistics-during-runtime data ---------------

(_, _, _, CI, ydelta_post_all, um_post_all, um_post_symmetric_all) \
    = get_odt_statistics_rt_at_chosen_averaging_times_um_symmetry(inputParams, averaging_times)
(ydelta_rt, yplus_rt, 
 um_rt, urmsf_rt, uFpert_rt, vm_rt, vrmsf_rt, vFpert_rt, wm_rt, wrmsf_rt, wFpert_rt,
 ufufm_rt, vfvfm_rt, wfwfm_rt, ufvfm_rt, ufwfm_rt, vfwfm_rt) \
    = get_odt_statistics_rt_at_chosen_averaging_times(inputParams, averaging_times)

#------------ ODT Reference data ---------------

# (ODT-Reference) calculated-at-runtime statistics 
(ydelta_ref, yplus_ref, 
 um_ref, urmsf_ref, uFpert_ref, vm_ref, vrmsf_ref, vFpert_ref, wm_ref, wrmsf_ref, wFpert_ref,
 ufufm_ref, vfvfm_ref, wfwfm_ref, ufvfm_ref, ufwfm_ref, vfwfm_ref,
 _, _, _) \
    = get_odt_statistics_reference(inputParams)

#--------------------------------------------------------------------------------------------

# Calculate ubulk
u_bulk = np.mean(um_rt, axis=0)
print("u_bulk(t) =", u_bulk)

# Build plots

visualizer = ChannelVisualizer(postRlzDir)
visualizer.build_u_mean_profile_odt_convergence(yplus_rt, yplus_ref, um_rt, um_ref, averaging_times_plots, "Reference")
visualizer.build_u_rmsf_profile_odt_convergence(yplus_rt, yplus_ref, urmsf_rt, vrmsf_rt, wrmsf_rt, urmsf_ref, vrmsf_ref, wrmsf_ref, averaging_times_plots, "Reference")
visualizer.build_reynolds_stress_diagonal_profile_odt_convergence(    yplus_rt, yplus_ref, ufufm_rt, vfvfm_rt, wfwfm_rt, ufufm_ref, vfvfm_ref, wfwfm_ref, averaging_times_plots, "Reference")
visualizer.build_reynolds_stress_not_diagonal_profile_odt_convergence(yplus_rt, yplus_ref, ufvfm_rt, ufwfm_rt, vfwfm_rt, ufvfm_ref, ufwfm_ref, vfwfm_ref, averaging_times_plots, "Reference")
visualizer.build_um_profile_symmetric_vs_nonsymmetric_odt_convergence(CI, ydelta_post_all, um_post_all, um_post_symmetric_all, averaging_times_plots)
