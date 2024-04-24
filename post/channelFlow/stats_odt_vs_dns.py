# Description: 
# Compute ODT mean and rmsf velocity profiles and reynolds stresses
# Plot results versus DNS results from DNS_statistics database.

# Usage
# python3 stats_odt_vs_dns.py [case_name] [reynolds_number]

# Arguments:
# case_name (str): Name of the case
# reynolds_number (int): reynolds number of the odt case, to get comparable dns result.

# Example Usage:
# python3 stats_odt_vs_dns.py channel180 180

# Comments:
# Values are in wall units (y+, u+) for both ODT and DNS results,
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

# post-processing directory
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
kvisc           = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho             = yml["params"]["rho0"]
dxmin           = yml["params"]["dxmin"]
nunif           = yml["params"]["nunif"]
domainLength    = yml["params"]["domainLength"] 
tBeginAvgInput  = yml["params"]["tBeginAvg"]
dTimeStart      = yml["dumpTimesGen"]["dTimeStart"]
dTimeEnd        = get_effective_dTimeEnd(caseN, rlzStr) # dTimeEnd = yml["dumpTimesGen"]["dTimeEnd"] can lead to errors if dTimeEnd > tEnd
dTimeStep       = yml["dumpTimesGen"]["dTimeStep"]
delta           = domainLength * 0.5
utau            = 1.0

assert tBeginAvg == tBeginAvgInput, f"Input argument 'tBeginAvg' = {tBeginAvg} must be equal to the input.yaml argument 'tBeginAvg' = {tBeginAvgInput} used for runtime statistics calculation"
if dTimeEnd < tEndAvg:
    print(f"ATTENTION: simulation ending time = {dTimeEnd} < expected tEndAvg = {tEndAvg} -> simulation has been truncated/terminated early.\n")
    tEndAvg = dTimeEnd

inputParams  = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
                "caseN": caseN, "rlzStr": rlzStr, 
                "dTimeStart": dTimeStart, "dTimeEnd": dTimeEnd, "dTimeStep": dTimeStep,
                "tBeginAvg": tBeginAvg, "tEndAvg": tEndAvg} 

#------------ Get ODT data ---------------

odtStatisticsFilepath = os.path.join(postRlzDir, "ODTstat.dat")

# calculated at post-processing statistics (from instantaneous dmp_*.dat files) -> saved in 'odtStatisticsFilepath' data file  
compute_odt_statistics_post(odtStatisticsFilepath, inputParams)
(ydelta_odt_post, yplus_odt_post, 
 um_odt_post, vm_odt_post, wm_odt_post, urmsf_odt_post, vrmsf_odt_post, wrmsf_odt_post, 
 ufufm_odt_post, vfvfm_odt_post, wfwfm_odt_post, ufvfm_odt_post, ufwfm_odt_post, vfwfm_odt_post, 
 viscous_stress_odt_post, reynolds_stress_odt_post, total_stress_odt_post, vt_u_plus_odt_post, d_u_plus_odt_post) \
    = get_odt_statistics_post(odtStatisticsFilepath)

# calculated-at-runtime statistics (from statistics stat_dmp_*.dat files)
(ydelta_odt_rt, yplus_odt_rt, 
 um_odt_rt, urmsf_odt_rt, uFpert_odt_rt, vm_odt_rt, vrmsf_odt_rt, vFpert_odt_rt, wm_odt_rt, wrmsf_odt_rt, wFpert_odt_rt,
 ufufm_odt_rt, vfvfm_odt_rt, wfwfm_odt_rt, ufvfm_odt_rt, ufwfm_odt_rt, vfwfm_odt_rt, \
 viscous_stress_odt_rt, reynolds_stress_odt_rt, total_stress_odt_rt ) \
    = get_odt_statistics_rt(inputParams)

# check y/delta coordinates coincide for both statistics calculations 
assert (abs(ydelta_odt_post - ydelta_odt_rt) < 1e-6).all(), "yu/delta from get_odt_statistics_post != yu/delta from get_odt_statistics_rt"

#------------ Get DNS statistics ---------------

(ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns, viscous_stress_dns, reynolds_stress_dns, total_stress_dns, vt_u_plus_dns,               p_u_plus_dns) \
    = get_dns_statistics(Retau, inputParams)

#------------ Compute Convergence Indicator (CI) for ODT uavg ------------

(CI_tEnd, ydelta_all, um_all, um_symmetric_all) = compute_convergence_indicator_odt_tEndAvg(inputParams)
(time_list, CI_list) = compute_convergence_indicator_odt_along_avg_time(inputParams)

#--------------------------------------------------------------------------------------------

# Build plots

visualizer = ChannelVisualizer(postRlzDir)

visualizer.build_u_mean_profile(yplus_odt_rt, yplus_dns, um_odt_post, um_odt_rt, um_dns, "DNS Data")
visualizer.build_u_rmsf_profile(yplus_odt_rt, yplus_dns, urmsf_odt_rt, vrmsf_odt_rt, wrmsf_odt_rt, urmsf_dns, vrmsf_dns, wrmsf_dns, "DNS Data")
visualizer.build_reynolds_stress_not_diagonal_profile(yplus_odt_rt, yplus_dns, ufvfm_odt_rt, ufwfm_odt_rt, vfwfm_odt_rt, ufvfm_dns, ufwfm_dns, vfwfm_dns, "DNS Data")
visualizer.build_reynolds_stress_diagonal_profile(yplus_odt_rt, yplus_dns, ufufm_odt_rt, vfvfm_odt_rt, wfwfm_odt_rt, ufufm_dns, vfvfm_dns, wfwfm_dns, "DNS Data")
visualizer.build_stress_decomposition(ydelta_odt_rt, ydelta_dns, viscous_stress_odt_rt, reynolds_stress_odt_rt, total_stress_odt_rt, viscous_stress_dns, reynolds_stress_dns, total_stress_dns, "DNS Data")
visualizer.build_TKE_budgets(yplus_odt_post, yplus_dns, vt_u_plus_odt_post, d_u_plus_odt_post, vt_u_plus_dns, p_u_plus_dns, "DNS Data")
visualizer.build_um_profile_symmetric_vs_nonsymmetric(CI_tEnd, ydelta_all, um_all, um_symmetric_all)
visualizer.build_CI_evolution(time_list, CI_list)

# check runtime vs. post calculations -> should be equal
visualizer.build_runtime_vs_post_statistics(yplus_odt_post, um_odt_post, vm_odt_post, wm_odt_post, urmsf_odt_post, vrmsf_odt_post, wrmsf_odt_post, um_odt_rt, vm_odt_rt, wm_odt_rt, urmsf_odt_rt, vrmsf_odt_rt, wrmsf_odt_rt)
visualizer.build_runtime_vs_post_reynolds_stress(yplus_odt_post, ufufm_odt_post, vfvfm_odt_post, wfwfm_odt_post, ufvfm_odt_post, ufwfm_odt_post, vfwfm_odt_post, ufufm_odt_rt, vfvfm_odt_rt, wfwfm_odt_rt, ufvfm_odt_rt, ufwfm_odt_rt, vfwfm_odt_rt)
