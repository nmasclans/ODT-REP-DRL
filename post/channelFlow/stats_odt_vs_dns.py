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
import matplotlib
matplotlib.use('PDF')       
import matplotlib.pyplot as plt
import os
from ChannelVisualizer import ChannelVisualizer

from utils import *

#--------------------------------------------------------------------------------------------

try :
    caseN = sys.argv[1]
    Retau = int(sys.argv[2])
except :
    raise ValueError("Include the case name in the call")

if not os.path.exists("../../data/"+caseN+"/post") :
    os.mkdir("../../data/"+caseN+"/post")


# --- Get ODT input parameters ---

odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho   = yml["params"]["rho0"]
dxmin = yml["params"]["dxmin"]
domainLength = yml["params"]["domainLength"] 
delta = domainLength * 0.5
utau  = 1.0
inputParams = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "caseN": caseN, "utau": utau} 

#------------ Get ODT data ---------------

odtStatisticsFilepath = "../../data/" + caseN + "/post/ODTstat.dat"
compute_odt_statistics(odtStatisticsFilepath, inputParams, plot_reynolds_stress_terms=False)
(ydelta_odt, yplus_odt, um_odt, urmsf_odt, vrmsf_odt, wrmsf_odt, ufufm_odt, vfvfm_odt, wfwfm_odt, ufvfm_odt, ufwfm_odt, vfwfm_odt, viscous_stress_odt, reynolds_stress_odt, total_stress_odt, vt_u_plus_odt, d_u_plus_odt, um_rt_odt) \
    = get_odt_statistics(odtStatisticsFilepath, inputParams)

#------------ Get DNS statistics ---------------

(ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns, viscous_stress_dns, reynolds_stress_dns, total_stress_dns, vt_u_plus_dns,               p_u_plus_dns) \
    = get_dns_statistics(Retau, inputParams)

#------------ Compute Convergence Indicator (CI) for ODT uavg ------------

(CI_tEnd, yuplus_all, um_all, um_symmetric_all) = compute_convergence_indicator_odt_tEnd(inputParams)
(time_list, CI_list) = compute_convergence_indicator_odt_along_time(inputParams)

#--------------------------------------------------------------------------------------------

# Build plots

visualizer = ChannelVisualizer(caseN)

visualizer.build_u_mean_profile(yplus_odt, yplus_dns, um_odt, um_rt_odt, um_dns)
visualizer.build_u_rmsf_profile(yplus_odt, yplus_dns, urmsf_odt, vrmsf_odt, wrmsf_odt, urmsf_dns, vrmsf_dns, wrmsf_dns)
visualizer.build_reynolds_stress_not_diagonal_profile(yplus_odt, yplus_dns, ufvfm_odt, ufwfm_odt, vfwfm_odt, ufvfm_dns, ufwfm_dns, vfwfm_dns)
visualizer.build_reynolds_stress_diagonal_profile(yplus_odt, yplus_dns, ufufm_odt, vfvfm_odt, wfwfm_odt, ufufm_dns, vfvfm_dns, wfwfm_dns)
visualizer.build_stress_decomposition(ydelta_odt, ydelta_dns, viscous_stress_odt, reynolds_stress_odt, total_stress_odt, viscous_stress_dns, reynolds_stress_dns, total_stress_dns)
visualizer.build_TKE_budgets(yplus_odt, yplus_dns, vt_u_plus_odt, d_u_plus_odt, vt_u_plus_dns, p_u_plus_dns)
visualizer.build_um_profile_symmetric_vs_nonsymmetric(CI_tEnd, yuplus_all, um_all, um_symmetric_all)
visualizer.build_CI_evolution(time_list, CI_list)
