# Description: 
# 1. Compute ODT reynolds stresses, anisotropy tensor, TKE
# 2. Calculate eigenvectors and eigenvalues of anisotropy tensor
# 3. Project the anisotropy eigenvalues into a baricentric map by linear mapping
# 4. Check: is the anisotropy tensor / Reynolds stresses satisfying the realizability conditions?

# Usage
# python3 anisotropy_tensor_odt_vs_dns.py [case_name] [reynolds_number]

# Arguments:
# case_name (str): Name of the case
# reynolds_number (int): reynolds number of the odt case, to get comparable dns result.

# Example Usage:
# python3 anisotropy_tensor.py channel180 180

# Comments:
# Values are in wall units (y+, u+) for both ODT and DNS results,
# Scaling is done in the input file (not explicitly here).

import yaml
import sys
import os
import math

import matplotlib
import numpy as np
import pandas as pd

from utils import *
from ChannelVisualizer import ChannelVisualizer


plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 14 )
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb} \usepackage{color}")

#--------------------------------------------------------------------------------------------

verbose = False

# --- Define parameters ---
tensor_kk_tolerance   = 1.0e-8;	# [-]
eigenvalues_tolerance = 1.0e-8;	# [-]
simulation_list = ["odt"] #, "dns"]

# --- Location of Barycentric map corners ---
x1c = np.array( [ 1.0 , 0.0 ] )
x2c = np.array( [ 0.0 , 0.0 ] )
x3c = np.array( [ 0.5 , math.sqrt(3.0)/2.0 ] )

# --- Get CASE parameters ---

try :
    caseN = sys.argv[1]
    Retau = int(sys.argv[2])
except :
    raise ValueError("Include the case name in the call")

if not os.path.exists("../../data/"+caseN+"/post") :
    os.mkdir("../../data/"+caseN+"/post")

# --- Visualizer ---

visualizer  = ChannelVisualizer(caseN)
nbins       = 1000

# --- Get ODT input parameters ---

odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho   = yml["params"]["rho0"]
dxmin = yml["params"]["dxmin"]
nunif = yml["params"]["nunif"]
domainLength = yml["params"]["domainLength"] 
# tEnd = yml["params"]["tEnd"]
tEnd = get_provisional_tEnd(caseN)  # tEnd, valid even while running odt
delta = domainLength * 0.5
utau  = 1.0
inputParams = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "caseN": caseN, "utau": utau, "tEnd": tEnd} 

#------------ Get ODT data ---------------

# post-processed statistics
odtStatisticsFilepath = "../../data/" + caseN + "/post/ODTstat.dat"
compute_odt_statistics(odtStatisticsFilepath, inputParams)
(ydelta_odt, yplus_odt, um_odt, vm_odt, wm_odt, urmsf_odt, vrmsf_odt, wrmsf_odt, ufufm_odt, vfvfm_odt, wfwfm_odt, ufvfm_odt, ufwfm_odt, vfwfm_odt, viscous_stress_odt, reynolds_stress_odt, total_stress_odt, vt_u_plus_odt, d_u_plus_odt) \
    = get_odt_statistics(odtStatisticsFilepath, inputParams)
# calculated-at-runtime statistics
(_, _,_,_, _,_,_, _,_,_, Rxx_odt_rt, Ryy_odt_rt, Rzz_odt_rt, Rxy_odt_rt, Rxz_odt_rt, Ryz_odt_rt) = get_odt_statistics_rt(inputParams)
(Rkk_odt_rt, lambda1_odt_rt, lambda2_odt_rt, lambda3_odt_rt, xmap1_odt_rt, xmap2_odt_rt) = compute_reynolds_stress_dof(Rxx_odt_rt, Ryy_odt_rt, Rzz_odt_rt, Rxy_odt_rt, Rxz_odt_rt, Ryz_odt_rt)

#------------ Get DNS statistics ---------------

(ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns, viscous_stress_dns, reynolds_stress_dns, total_stress_dns, vt_u_plus_dns,               p_u_plus_dns) \
    = get_dns_statistics(Retau, inputParams)

for sim in simulation_list:

    # ----------- Get data of interest -----------------
    if sim == "odt":
        print("\n----------------- ODT simulation data -----------------")
        Rxx   = ufufm_odt
        Rxy   = ufvfm_odt
        Rxz   = ufwfm_odt
        Ryy   = vfvfm_odt
        Ryz   = vfwfm_odt
        Rzz   = wfwfm_odt
        urmsf = urmsf_odt 
        vrmsf = vrmsf_odt 
        wrmsf = wrmsf_odt 
        yplus = yplus_odt
        yplus_max = np.max(yplus)
    elif sim == "dns":
        print("\n----------------- DNS simulation data -----------------")
        Rxx   = ufufm_dns
        Rxy   = ufvfm_dns
        Rxz   = ufwfm_dns
        Ryy   = vfvfm_dns
        Ryz   = vfwfm_dns
        Rzz   = wfwfm_dns
        urmsf = urmsf_dns 
        vrmsf = vrmsf_dns 
        wrmsf = wrmsf_dns 
        yplus = yplus_dns
        yplus_max = np.max(yplus)
    else:
        raise ValueError("'simulation_list' can only take values 'odt' or 'dns', but element '{sim}' was chosen.")

    #-----------------------------------------------------------------------------------------
    #           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
    #-----------------------------------------------------------------------------------------
    
    (Rkk, lambda1, lambda2, lambda3, xmap1, xmap2) = compute_reynolds_stress_dof(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz)

    # ---------------------- Plot xmap 1st-coordinate vs. y (post-processing & runtime calculations) ---------------------- 
    
    # pdf plot of barycentric map 1st coordinate
    visualizer.plot_pdf(xmap1, [0.0, 1.0], "1st coord barycentric map", nbins, "pdf_barmapx_post")
    
    # plot of barycentric map 1st coordinate vs yplus
    visualizer.plot_line(yplus, xmap1,    [0.0, yplus.max()], [0.0, 1.0], r"$y^{+}$", "1st coord barycentric map", "anisotropy_tensor_yplus_vs_barmapx_vs_yplus_post")
    visualizer.plot_line(yplus, xmap1_odt_rt, [0.0, yplus.max()], [0.0, 1.0], r"$y^{+}$", "1st coord barycentric map", "anisotropy_tensor_yplus_vs_barmapx_vs_yplus_rt")


    # ---------------------- Plot Barycentric Map ---------------------- 
    
    # post-processing calculations
    visualizer.build_anisotropy_tensor_barycentric_map(xmap1, xmap2, yplus, tEnd, caseN, f"anisotropy_tensor_barycentric_map_{sim}_post")
    # runtime calculations
    visualizer.build_anisotropy_tensor_barycentric_map(xmap1_odt_rt, xmap2_odt_rt, yplus, tEnd, caseN, f"anisotropy_tensor_barycentric_map_odt_rt")