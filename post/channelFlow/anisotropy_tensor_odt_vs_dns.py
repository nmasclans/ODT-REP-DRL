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

import numpy as np

from utils import *
from ChannelVisualizer import ChannelVisualizer

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---

try :
    caseN     = sys.argv[1]
    rlzN      = int(sys.argv[2])
    Retau     = int(sys.argv[3])
    tBeginAvg = float(sys.argv[4])
    tEndAvg   = float(sys.argv[5])
    print(f"Script parameters: \n- Case name: {caseN} \n- Realization Number: {rlzN} \n- Retau: {Retau} \n- Time Begin Averaging: {tBeginAvg} \n- Time End Averaging: {tEndAvg} \n")
except :
    raise ValueError("Missing call arguments, should be: <case_name> <realization_number> <reynolds_number> <time_begin_averaging> <time_end_averaging>")

# --- Script parameters ---
verbose = False

# Define main parameters
tensor_kk_tolerance   = 1.0e-8;	# [-]
eigenvalues_tolerance = 1.0e-8;	# [-]

# Location of Barycentric map corners
x1c = np.array( [ 1.0 , 0.0 ] )
x2c = np.array( [ 0.0 , 0.0 ] )
x3c = np.array( [ 0.5 , math.sqrt(3.0)/2.0 ] )

# post-processing directory
postDir = f"../../data/{caseN}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)
# post-processing sub-directory for single realization
rlzStr = f"{rlzN:05d}"
postRlzDir = os.path.join(postDir, f"post_{rlzStr}")
if not os.path.exists(postRlzDir):
    os.mkdir(postRlzDir)

# --- Visualizer ---

visualizer = ChannelVisualizer(postRlzDir)
nbins      = 1000

# --- Get ODT input parameters ---

odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
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
assert tBeginAvg == tBeginAvgInput, f"Input argument 'tBeginAvg' = {tBeginAvg} must be equal to the input.yaml argument 'tBeginAvg' = {tBeginAvgRt} used for runtime statistics calculation"
inputParams     = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
                "caseN": caseN, "rlzStr":rlzStr,
                "dTimeStart": dTimeStart, "dTimeEnd": dTimeEnd, "dTimeStep": dTimeStep, 
                "tBeginAvg": tBeginAvg, "tEndAvg": tEndAvg} 

#------------ Get ODT data ---------------

### # post-processed statistics
### odtStatisticsFilepath = os.path.join(postRlzDir, "ODTstat.dat")
### compute_odt_statistics_post(odtStatisticsFilepath, inputParams)
### (ydelta_odt_post, yplus_odt_post, 
###  um_odt_post, vm_odt_post, wm_odt_post, urmsf_odt_post, vrmsf_odt_post, wrmsf_odt_post,
###  ufufm_odt_post, vfvfm_odt_post, wfwfm_odt_post, ufvfm_odt_post, ufwfm_odt_post, vfwfm_odt_post, 
###  viscous_stress_odt_post, reynolds_stress_odt_post, total_stress_odt_post, vt_u_plus_odt_post, d_u_plus_odt_post) \
###     = get_odt_statistics_post(odtStatisticsFilepath)
### (Rkk_odt_post, lambda1_odt_post, lambda2_odt_post, lambda3_odt_post, xmap1_odt_post, xmap2_odt_post) \
###     = compute_reynolds_stress_dof(ufufm_odt_post, vfvfm_odt_post, wfwfm_odt_post, ufvfm_odt_post, ufwfm_odt_post, vfwfm_odt_post)

# calculated-at-runtime statistics
(ydelta_odt_rt, yplus_odt_rt, 
 _,_,_, _,_,_, _,_,_, 
 ufufm_odt_rt, vfvfm_odt_rt, wfwfm_odt_rt, ufvfm_odt_rt, ufwfm_odt_rt, vfwfm_odt_rt,
 _,_,_) \
    = get_odt_statistics_rt(inputParams)
(Rkk_odt_rt, lambda1_odt_rt, lambda2_odt_rt, lambda3_odt_rt, xmap1_odt_rt, xmap2_odt_rt) \
    = compute_reynolds_stress_dof(ufufm_odt_rt, vfvfm_odt_rt, wfwfm_odt_rt, ufvfm_odt_rt, ufwfm_odt_rt, vfwfm_odt_rt)

#------------ Get DNS statistics ---------------

(ydelta_dns, yplus_dns, 
 _, _, _, _, 
 ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns, 
 _, _, _, _, _) \
    = get_dns_statistics(Retau, inputParams)
(Rkk_dns, lambda1_dns, lambda2_dns, lambda3_dns, xmap1_dns, xmap2_dns) \
    = compute_reynolds_stress_dof(ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns)


#-----------------------------------------------------------------------------------------
#           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
#-----------------------------------------------------------------------------------------

# ---------------------- Plot xmap 1st-coordinate vs. y (post-processing & runtime calculations) ---------------------- 

# pdf plot of barycentric map 1st coordinate
visualizer.plot_pdf(xmap1_odt_rt, [0.0, 1.0], "1st coord barycentric map", nbins, "anisotropy_tensor_pdf_barmapx_odt")
visualizer.plot_pdf(xmap1_dns,    [0.0, 1.0], "1st coord barycentric map", nbins, "anisotropy_tensor_pdf_barmapx_dns")

# plot of barycentric map 1st coordinate vs yplus
### # post-processing calculations
### visualizer.plot_line(yplus_odt_post, xmap1_odt_post, [0.0, yplus_odt_post.max()], [0.0, 1.0], r"$y^{+}$", "1st coord barycentric map", "anisotropy_tensor_yplus_vs_barmapx_post")
# runtime calculations
visualizer.plot_line(yplus_odt_rt, xmap1_odt_rt, [0.0, yplus_odt_rt.max()], [0.0, 1.0], r"$y^{+}$", "1st coord barycentric map", "anisotropy_tensor_yplus_vs_barmapx_odt")
# DNS data
visualizer.plot_line(yplus_dns,    xmap1_dns,    [0.0, yplus_dns.max()],    [0.0, 1.0], r"$y^{+}$", "1st coord barycentric map", "anisotropy_tensor_yplus_vs_barmapx_dns")

# ---------------------- Plot Barycentric Map ---------------------- 

### # post-processing calculations
### visualizer.build_anisotropy_tensor_barycentric_map(xmap1_odt_post, xmap2_odt_post, yplus_odt_post, tEndAvg, f"anisotropy_tensor_barycentric_map_post")
# runtime calculations
visualizer.build_anisotropy_tensor_barycentric_map(xmap1_odt_rt, xmap2_odt_rt, yplus_odt_rt, tEndAvg-tBeginAvg, f"anisotropy_tensor_barycentric_map_odt")
# DNS data
visualizer.build_anisotropy_tensor_barycentric_map(xmap1_dns,    xmap2_dns,    yplus_dns,    1000-tBeginAvg,    f"anisotropy_tensor_barycentric_map_dns")
