# Description: 
# Compute ODT mean and rmsf velocity profiles and reynolds stresses
# Plot results versus ODT-reference converged results from ODT_reference database.

# Usage
# python3 stats_odt_vs_odtReference.py #TODO: add input arguments

# Arguments:
# case_name (str): Name of the case
# TODO: add input arguments

# Example Usage:
# python3 stats_odt_vs_odtReference.py # TODO: Add arguments

# Comments:
# Values are in wall units (y+, u+) for ODT results,
# Scaling is done in the input file (not explicitly here).

import numpy as np
import yaml
import sys
import os
from ChannelVisualizer import ChannelVisualizer

from utils import *

#--------------------------------------------------------------------------------------------

try :
    caseN     = sys.argv[1]
    rlzN      = int(sys.argv[2])
    Retau     = int(sys.argv[3])
    tBeginAvg = float(sys.argv[4])
    tEndAvg   = float(sys.argv[5])
    print(f"Script parameters: \n- Case name: {caseN} \n- Realization Number: {rlzN} \n- Retau: {Retau} \n- Time Begin Averaging: {tBeginAvg} \n- Time End Averaging: {tEndAvg} \n")
except :
    raise ValueError("Missing call arguments, should be: <case_name> <realization_number> <reynolds_number> <time_begin_averaging> <time_end_averaging>")

# --- post-processing directory ---

# post-processing parent directory
postDir = f"../../data/{caseN}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)

# post-processing sub-directory for single realization
rlzStr = f"{rlzN:05d}"
postRlzDir = os.path.join(postDir, f"post_{rlzStr}")
if not os.path.exists(postRlzDir):
    os.mkdir(postRlzDir)

# --- Get ODT input parameters ---

odtInputDataFilepath  = f"../../data/{caseN}/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc        = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho          = yml["params"]["rho0"]
dxmin        = yml["params"]["dxmin"]
nunif        = yml["params"]["nunif"]
domainLength = yml["params"]["domainLength"] 
dTimeStart   = yml["dumpTimesGen"]["dTimeStart"]
dTimeEnd     = get_effective_dTimeEnd(caseN, rlzStr) # dTimeEnd = yml["dumpTimesGen"]["dTimeEnd"] can lead to errors if dTimeEnd > tEnd
dTimeStep    = yml["dumpTimesGen"]["dTimeStep"]
tBeginAvgRt  = yml["params"]["tBeginAvg"] 
delta        = domainLength * 0.5
utau         = 1.0

assert tBeginAvg == tBeginAvgRt, "Input argument 'tBeginAvg' = {tBeginAvg} must be equal to the input.yaml argument 'tBeginAvg' = {tBeginAvgRt} used for statistics calculation during runtime"
assert tEndAvg <= dTimeEnd, "Input argument 'tEndAvg' must be <= effective dTimeEnd, the time of the last stat_dmp_* file"

inputParams  = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
                "caseN": caseN, "rlzStr": rlzStr, 
                "dTimeStart": dTimeStart, "dTimeEnd": dTimeEnd, "dTimeStep": dTimeStep, 
                "tBeginAvg": tBeginAvg, "tEndAvg": tEndAvg} 


#------------ Get ODT (non-reference) data ---------------

odtStatisticsFilepath = os.path.join(postRlzDir, "ODTstat.dat")

# (ODT-non-reference) calculated at post-processing statistics (from instantaneous dmp_*.dat files) -> saved in 'odtStatisticsFilepath' data file  
compute_odt_statistics_post(odtStatisticsFilepath, inputParams, plot_reynolds_stress_terms=False)
(ydelta_post, yplus_post, um_post, vm_post, wm_post, urmsf_post, vrmsf_post, wrmsf_post, 
 ufufm_post, vfvfm_post, wfwfm_post, ufvfm_post, ufwfm_post, vfwfm_post, 
 viscous_stress_post, reynolds_stress_post, total_stress_post, _, _) \
    = get_odt_statistics_post(odtStatisticsFilepath)

# (ODT-non-reference) calculated-at-runtime statistics (from statistics stat_dmp_*.dat files)
(ydelta_rt, yplus_rt, um_rt, urmsf_rt, uFpert_rt, vm_rt, vrmsf_rt, vFpert_rt, wm_rt, wrmsf_rt, wFpert_rt,
 ufufm_rt, vfvfm_rt, wfwfm_rt, ufvfm_rt, ufwfm_rt, vfwfm_rt,
 viscous_stress_rt, reynolds_stress_rt, total_stress_rt) \
    = get_odt_statistics_rt(inputParams)

# check y/delta coordinates coincide for both statistics calculations 
assert (abs(ydelta_post - ydelta_rt) < 1e-6).all(), "yu/delta from get_odt_statistics_post != yu/delta from get_odt_statistics_rt"


#------------ Get ODT-Reference statistics ---------------

# (ODT-Reference) calculated-at-runtime statistics 
(ydelta_ref, yplus_ref, um_ref, urmsf_ref, uFpert_ref, vm_ref, vrmsf_ref, vFpert_ref, wm_ref, wrmsf_ref, wFpert_ref,
 ufufm_ref, vfvfm_ref, wfwfm_ref, ufvfm_ref, ufwfm_ref, vfwfm_ref,
 viscous_stress_ref, reynolds_stress_ref, total_stress_ref) \
    = get_odt_statistics_reference(inputParams)

#------------ Compute Convergence Indicator (CI) for ODT uavg ------------

(CI_tEnd, ydelta_all, um_all, um_symmetric_all) = compute_convergence_indicator_odt_tEndAvg(inputParams)
(time_list, CI_list) = compute_convergence_indicator_odt_along_avg_time(inputParams)

#--------------------------------------------------------------------------------------------

# Build plots

visualizer = ChannelVisualizer(postRlzDir)

visualizer.build_u_mean_profile(yplus_rt, yplus_ref, um_post, um_rt, um_ref, "Reference")
visualizer.build_u_rmsf_profile(yplus_rt, yplus_ref, urmsf_rt, vrmsf_rt, wrmsf_rt, urmsf_ref, vrmsf_ref, wrmsf_ref, "Reference")
visualizer.build_reynolds_stress_not_diagonal_profile(yplus_rt, yplus_ref, ufvfm_rt, ufwfm_rt, vfwfm_rt, ufvfm_ref, ufwfm_ref, vfwfm_ref, "Reference")
visualizer.build_reynolds_stress_diagonal_profile(yplus_rt, yplus_ref, ufufm_rt, vfvfm_rt, wfwfm_rt, ufufm_ref, vfvfm_ref, wfwfm_ref, "Reference")
visualizer.build_stress_decomposition(ydelta_rt, ydelta_ref, viscous_stress_rt, reynolds_stress_rt, total_stress_rt, viscous_stress_ref, reynolds_stress_ref, total_stress_ref, "Reference")
visualizer.build_um_profile_symmetric_vs_nonsymmetric(CI_tEnd, ydelta_all, um_all, um_symmetric_all)
visualizer.build_CI_evolution(time_list, CI_list)

# check runtime vs. post calculations -> should be equal
visualizer.build_runtime_vs_post_statistics(yplus_rt, um_post, vm_post, wm_post, urmsf_post, vrmsf_post, wrmsf_post, um_rt, vm_rt, wm_rt, urmsf_rt, vrmsf_rt, wrmsf_rt)
visualizer.build_runtime_vs_post_reynolds_stress(yplus_rt, ufufm_post, vfvfm_post, wfwfm_post, ufvfm_post, ufwfm_post, vfwfm_post, ufufm_rt, vfvfm_rt, wfwfm_rt, ufvfm_rt, ufwfm_rt, vfwfm_rt)
