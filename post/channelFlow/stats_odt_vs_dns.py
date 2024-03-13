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
    caseN    = sys.argv[1]
    rlzN     = int(sys.argv[2])
    Retau    = int(sys.argv[3])
    tEndAvg  = float(sys.argv[4])
    print(f"Script parameters: \n- Case name: {caseN} \n- Realization Number: {rlzN} \n- Retau: {Retau} \n- Time End Averaging: {tEndAvg} \n")
except :
    raise ValueError("Missing call arguments, should be: <case_name> <realization_number> <reynolds_number> <time_end_averaging>")

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

odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
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
delta        = domainLength * 0.5
utau         = 1.0
inputParams  = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
                "caseN": caseN, "rlzStr": rlzStr, 
                "dTimeStart": dTimeStart, "dTimeEnd": dTimeEnd, "dTimeStep": dTimeStep, "tEndAvg": tEndAvg} 

#------------ Get ODT data ---------------

odtStatisticsFilepath = os.path.join(postRlzDir, "ODTstat.dat")

# post-processed statistics
compute_odt_statistics(odtStatisticsFilepath, inputParams, plot_reynolds_stress_terms=False)
(ydelta_odt, yplus_odt, um_odt, vm_odt, wm_odt, urmsf_odt, vrmsf_odt, wrmsf_odt, ufufm_odt, vfvfm_odt, wfwfm_odt, ufvfm_odt, ufwfm_odt, vfwfm_odt, viscous_stress_odt, reynolds_stress_odt, total_stress_odt, vt_u_plus_odt, d_u_plus_odt) \
    = get_odt_statistics(odtStatisticsFilepath, inputParams)

# calculated-at-runtime statistics
(ydelta_odt_rt, yplus_odt_rt, um_odt_rt, urmsf_odt_rt, uFpert_odt_rt, vm_odt_rt, vrmsf_odt_rt, vFpert_odt_rt, wm_odt_rt, wrmsf_odt_rt, wFpert_odt_rt,
 ufufm_odt_rt, vfvfm_odt_rt, wfwfm_odt_rt, ufvfm_odt_rt, ufwfm_odt_rt, vfwfm_odt_rt, \
 # _, _, _, _, _), \
 ) = get_odt_statistics_rt(inputParams)

# check y/delta coordinates coincide for both statistics calculations 
assert (abs(ydelta_odt - ydelta_odt_rt) < 1e-6).all(), "yu/delta from get_odt_statistics != yu/delta from get_odt_statistics_rt"

#------------ Get DNS statistics ---------------

(ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns, viscous_stress_dns, reynolds_stress_dns, total_stress_dns, vt_u_plus_dns,               p_u_plus_dns) \
    = get_dns_statistics(Retau, inputParams)

#------------ Compute Convergence Indicator (CI) for ODT uavg ------------

(CI_tEnd, yuplus_all, um_all, um_symmetric_all) = compute_convergence_indicator_odt_tEnd(inputParams)
(time_list, CI_list) = compute_convergence_indicator_odt_along_time(inputParams)

#--------------------------------------------------------------------------------------------

# Build plots

visualizer = ChannelVisualizer(postRlzDir)

visualizer.build_u_mean_profile(yplus_odt, yplus_dns, um_odt, um_odt_rt, um_dns)
visualizer.build_u_rmsf_profile(yplus_odt, yplus_dns, urmsf_odt, vrmsf_odt, wrmsf_odt, urmsf_dns, vrmsf_dns, wrmsf_dns)
visualizer.build_reynolds_stress_not_diagonal_profile(yplus_odt, yplus_dns, ufvfm_odt, ufwfm_odt, vfwfm_odt, ufvfm_dns, ufwfm_dns, vfwfm_dns)
visualizer.build_reynolds_stress_diagonal_profile(yplus_odt, yplus_dns, ufufm_odt, vfvfm_odt, wfwfm_odt, ufufm_dns, vfvfm_dns, wfwfm_dns)
visualizer.build_stress_decomposition(ydelta_odt, ydelta_dns, viscous_stress_odt, reynolds_stress_odt, total_stress_odt, viscous_stress_dns, reynolds_stress_dns, total_stress_dns)
visualizer.build_TKE_budgets(yplus_odt, yplus_dns, vt_u_plus_odt, d_u_plus_odt, vt_u_plus_dns, p_u_plus_dns)
visualizer.build_um_profile_symmetric_vs_nonsymmetric(CI_tEnd, yuplus_all, um_all, um_symmetric_all)
visualizer.build_CI_evolution(time_list, CI_list)

# check runtime vs. post calculations -> should be equal
visualizer.build_runtime_vs_post_statistics(yplus_odt, um_odt, vm_odt, wm_odt, urmsf_odt, vrmsf_odt, wrmsf_odt, um_odt_rt, vm_odt_rt, wm_odt_rt, urmsf_odt_rt, vrmsf_odt_rt, wrmsf_odt_rt)
visualizer.build_runtime_vs_post_reynolds_stress(yplus_odt, ufufm_odt, vfvfm_odt, wfwfm_odt, ufvfm_odt, ufwfm_odt, vfwfm_odt, ufufm_odt_rt, vfvfm_odt_rt, wfwfm_odt_rt, ufvfm_odt_rt, ufwfm_odt_rt, vfwfm_odt_rt)
