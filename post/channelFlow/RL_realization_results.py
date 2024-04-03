import yaml
import sys
import os
import numpy as np
import pandas as pd

from utils import *
from ChannelVisualizer import ChannelVisualizer

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---

# RL case is non-converged
# baseline non-RL has both non-converged and baseline statistics
try :
    i = 1
    Retau             = int(sys.argv[i]);   i+=1
    caseN_nonRL       = sys.argv[i];        i+=1
    rlzN_nonRL        = int(sys.argv[i]);   i+=1
    caseN_RL          = sys.argv[i];        i+=1
    rlzN_min_RL       = int(sys.argv[i]);   i+=1
    rlzN_max_RL       = int(sys.argv[i]);   i+=1
    rlzN_step_RL      = int(sys.argv[i]);   i+=1
    tBeginAvg_nonConv = float(sys.argv[i]); i+=1
    tEndAvg_nonConv   = float(sys.argv[i]); i+=1
    tEndAvg_conv      = float(sys.argv[i]); i+=1
    dtAvg             = float(sys.argv[i]); i+=1
    print(f"Script parameters: \n" \
          f"- Re_tau: {Retau} \n" \
          f"- Case name non-RL: {caseN_nonRL} \n" \
          f"- Realization Number non-RL: {rlzN_nonRL} \n" \
          f"- Case name RL: {caseN_RL} \n" \
          f"- Realization Number Min RL: {rlzN_min_RL} \n" \
          f"- Realization Number Max RL: {rlzN_max_RL} \n" \
          f"- Realization Number Step RL: {rlzN_step_RL} \n" \
          f"- Time Begin Averaging non-converged (both RL and non-RL): {tBeginAvg_nonConv} \n" \
          f"- Time End Averaging non-converged (both RL and non-RL): {tEndAvg_nonConv} \n" \
          f"- Time End Averaging converged (non-RL, baseline): {tEndAvg_conv} \n"
          f"- dt averaging: {dtAvg}"
    )
except :
    raise ValueError("Missing call arguments, should be: <1_Re_tau> <2_case_name_nonRL> <3_realization_number_nonRL> <4_case_name_RL> <5_realization_number_min_RL> <6_realization_number_max_RL> <7_realization_number_step_RL> <8_time_end_averaging_non_converged> <9_time_end_averaging_converged>")

# --- Get ODT input parameters ---

# first and last realizations
rlzN_first   = rlzN_min_RL
rlzStr_first = f"{rlzN_first:05d}"
rlzN_last    = rlzN_max_RL
rlzStr_last  = f"{rlzN_last:05d}"

# get input data from first realization
odtInputDataFilepath_RL_nonConv  = f"../../data/{caseN_RL}/input/input.yaml"
with open(odtInputDataFilepath_RL_nonConv) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc        = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho          = yml["params"]["rho0"]
dxmin        = yml["params"]["dxmin"]
nunif        = yml["params"]["nunif"]
domainLength = yml["params"]["domainLength"] 
dTimeStart   = yml["dumpTimesGen"]["dTimeStart"]
dTimeEnd     = get_effective_dTimeEnd(caseN_RL, rlzStr_first) # dTimeEnd = yml["dumpTimesGen"]["dTimeEnd"] can lead to errors if dTimeEnd > tEnd
dTimeStep    = yml["dumpTimesGen"]["dTimeStep"]
delta        = domainLength * 0.5
utau         = 1.0
inputParams_RL_nonConv = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
                          "caseN": caseN_RL, "rlzStr": rlzStr_first, 
                          "dTimeStart": dTimeStart, "dTimeEnd": dTimeEnd, "dTimeStep": dTimeStep, 
                          "tEndAvg": tEndAvg_nonConv} 
print(inputParams_RL_nonConv)

# --- post-processing directories to store results

# post-processing directory
postDir = f"../../data/{caseN_RL}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)

# post-processing sub-directory for multiples realizations comparison
postMultipleRlzDir = os.path.join(postDir, f"comparative_{rlzStr_first}_{rlzStr_last}")
if not os.path.exists(postMultipleRlzDir):
    os.mkdir(postMultipleRlzDir)

#------------ Get NON-CONVERGED runtime-calculated 'um' at t=tEndAvg for each ODT RL-realization ---------------

# realizations array
rlzN_Arr   = np.arange(rlzN_min_RL, rlzN_max_RL+1, rlzN_step_RL)
rlzStr_Arr = [f"{rlzN:05d}" for rlzN in rlzN_Arr]
nrlz     = len(rlzN_Arr)

# empty arrays
ydelta_RL_nonConv  = np.zeros(int(nunif/2))
yplus_RL_nonConv   = np.zeros(int(nunif/2))
um_RL_nonConv      = np.zeros([int(nunif/2), nrlz])
urmsf_RL_nonConv   = np.zeros([int(nunif/2), nrlz])
Rkk_RL_nonConv     = np.zeros([int(nunif/2), nrlz])
lambda1_RL_nonConv = np.zeros([int(nunif/2), nrlz])
lambda2_RL_nonConv = np.zeros([int(nunif/2), nrlz])
lambda3_RL_nonConv = np.zeros([int(nunif/2), nrlz])
xmap1_RL_nonConv   = np.zeros([int(nunif/2), nrlz])
xmap2_RL_nonConv   = np.zeros([int(nunif/2), nrlz])

for irlz in range(nrlz):

    # modify rlzN information in inputParams_RL:
    inputParams_RL_nonConv["rlzStr"] = rlzStr_Arr[irlz]
    print("\n\n--- RL: Realization #" + inputParams_RL_nonConv["rlzStr"] + ", Time " + str(inputParams_RL_nonConv["tEndAvg"]) +  " ---")

    # get u-statistics data
    ### (ydelta, yplus, um_data, urmsf_data) = get_odt_udata_rt(inputParams_RL_nonConv)
    (ydelta_RL_nonConv_, yplus_RL_nonConv_, 
     um_RL_nonConv_, urmsf_RL_nonConv_, _, _, _, _, _, _, _,
     ufufm_RL_nonConv_, vfvfm_RL_nonConv_, wfwfm_RL_nonConv_, ufvfm_RL_nonConv_, ufwfm_RL_nonConv_, vfwfm_RL_nonConv_, \
     _, _, _) \
        = get_odt_statistics_rt(inputParams_RL_nonConv)
    (Rkk_RL_nonConv_, lambda1_RL_nonConv_, lambda2_RL_nonConv_, lambda3_RL_nonConv_, xmap1_RL_nonConv_, xmap2_RL_nonConv_) \
        = compute_reynolds_stress_dof(ufufm_RL_nonConv_, vfvfm_RL_nonConv_, wfwfm_RL_nonConv_, ufvfm_RL_nonConv_, ufwfm_RL_nonConv_, vfwfm_RL_nonConv_)

    # store realization data
    if irlz == 0:
        ydelta_RL_nonConv = ydelta_RL_nonConv_
        yplus_RL_nonConv  = yplus_RL_nonConv_
    else:
        assert (ydelta_RL_nonConv == ydelta_RL_nonConv_).all()
        assert (yplus_RL_nonConv  == yplus_RL_nonConv_).all()
    um_RL_nonConv[:,irlz]      = um_RL_nonConv_
    urmsf_RL_nonConv[:,irlz]   = urmsf_RL_nonConv_
    Rkk_RL_nonConv[:,irlz]     = Rkk_RL_nonConv_
    lambda1_RL_nonConv[:,irlz] = lambda1_RL_nonConv_
    lambda2_RL_nonConv[:,irlz] = lambda2_RL_nonConv_
    lambda3_RL_nonConv[:,irlz] = lambda3_RL_nonConv_
    xmap1_RL_nonConv[:,irlz]   = xmap1_RL_nonConv_
    xmap2_RL_nonConv[:,irlz]   = xmap2_RL_nonConv_

#------------ Get NON-CONVERGED runtime-calculated 'um' at t=tEndAvg for single ODT non-RL-realization ---------------

# --- Get input parameters
odtInputDataFilepath_nonRL_nonConv  = f"../../data/{caseN_nonRL}/input/input.yaml"
rlzStr_nonRL = f"{rlzN_nonRL:05d}"
with open(odtInputDataFilepath_nonRL_nonConv) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc        = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho          = yml["params"]["rho0"]
dxmin        = yml["params"]["dxmin"]
nunif        = yml["params"]["nunif"]
domainLength = yml["params"]["domainLength"] 
dTimeStart   = yml["dumpTimesGen"]["dTimeStart"]
dTimeEnd     = get_effective_dTimeEnd(caseN_nonRL, rlzStr_nonRL) # dTimeEnd = yml["dumpTimesGen"]["dTimeEnd"] can lead to errors if dTimeEnd > tEnd
dTimeStep    = yml["dumpTimesGen"]["dTimeStep"]
delta        = domainLength * 0.5
utau         = 1.0
inputParams_nonRL_nonConv = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
                             "caseN": caseN_nonRL, "rlzStr": rlzStr_nonRL, 
                             "dTimeStart": dTimeStart, "dTimeEnd": dTimeEnd, "dTimeStep": dTimeStep, 
                             "tEndAvg": tEndAvg_nonConv} 
print(inputParams_nonRL_nonConv)

# --- Get data
print(f"\n--- Non-RL: Realization #" + inputParams_nonRL_nonConv["rlzStr"] + ", Time " + str(inputParams_nonRL_nonConv["tEndAvg"]) + " ---")
### (_, _, um_nonRL_nonConv, urmsf_nonRL_nonConv) = get_odt_udata_rt(inputParams_nonRL_nonConv)
(ydelta_nonRL_nonConv, yplus_nonRL_nonConv, 
 um_nonRL_nonConv, urmsf_nonRL_nonConv, _, _, _, _, _, _, _,
 ufufm_nonRL_nonConv, vfvfm_nonRL_nonConv, wfwfm_nonRL_nonConv, ufvfm_nonRL_nonConv, ufwfm_nonRL_nonConv, vfwfm_nonRL_nonConv, \
 _, _, _) \
    = get_odt_statistics_rt(inputParams_nonRL_nonConv)
(Rkk_nonRL_nonConv, lambda1_nonRL_nonConv, lambda2_nonRL_nonConv, lambda3_nonRL_nonConv, xmap1_nonRL_nonConv, xmap2_nonRL_nonConv) \
    = compute_reynolds_stress_dof(ufufm_nonRL_nonConv, vfvfm_nonRL_nonConv, wfwfm_nonRL_nonConv, ufvfm_nonRL_nonConv, ufwfm_nonRL_nonConv, vfwfm_nonRL_nonConv)

#------------ Get BASELINE runtime-calculated 'um' at t=dTimeEnd (>>tEndAvg used before) for single ODT non-RL-realization ---------------

# --- build input parameters, tEndAvg=dTimeEnd of the Baseline is >> tEndAvg used in previous NON-CONVERGED data
inputParams_nonRL_conv = inputParams_nonRL_nonConv.copy()
inputParams_nonRL_conv["tEndAvg"]  = tEndAvg_conv
print(inputParams_nonRL_conv)

# --- get data
print(f"\n--- Non-RL Baseline: Realization #" + inputParams_nonRL_conv["rlzStr"] + ", Time " + str(inputParams_nonRL_conv["tEndAvg"]) + " ---")
### (_, _, um_nonRL_conv, urmsf_nonRL_conv) = get_odt_udata_rt(inputParams_nonRL_conv)
(ydelta_nonRL_conv, yplus_nonRL_conv, 
 um_nonRL_conv, urmsf_nonRL_conv, _, _, _, _, _, _, _,
 ufufm_nonRL_conv, vfvfm_nonRL_conv, wfwfm_nonRL_conv, ufvfm_nonRL_conv, ufwfm_nonRL_conv, vfwfm_nonRL_conv, \
 _, _, _) \
    = get_odt_statistics_rt(inputParams_nonRL_conv)
(Rkk_nonRL_conv, lambda1_nonRL_conv, lambda2_nonRL_conv, lambda3_nonRL_conv, xmap1_nonRL_conv, xmap2_nonRL_conv) \
    = compute_reynolds_stress_dof(ufufm_nonRL_conv, vfvfm_nonRL_conv, wfwfm_nonRL_conv, ufvfm_nonRL_conv, ufwfm_nonRL_conv, vfwfm_nonRL_conv)


# for further use:
um_baseline      = um_nonRL_conv
urmsf_baseline   = urmsf_nonRL_conv 
Rkk_baseline     = Rkk_nonRL_conv
lambda1_baseline = lambda1_nonRL_conv
lambda2_baseline = lambda2_nonRL_conv
lambda3_baseline = lambda3_nonRL_conv
xmap1_baseline   = xmap1_nonRL_conv 
xmap2_baseline   = xmap2_nonRL_conv
tEndAvg_baseline = tEndAvg_conv

# ------------- Calculate errors non-converged results vs converged baseline -------------

# ---- Calculate NRMSE

num_points = len(um_nonRL_conv)
# Error of RL non-converged
NRMSE_RL = np.zeros(nrlz)
for irlz in range(nrlz):
    NRMSE_RL_num   = np.sqrt(np.sum((um_RL_nonConv[:,irlz] - um_baseline)**2))
    NRMSE_RL_denum = np.sqrt(np.sum((um_baseline)**2))
    NRMSE_RL[irlz] = NRMSE_RL_num / NRMSE_RL_denum 
# Error of non-RL non-converged
NRMSE_nonRL_num   = np.sqrt(np.sum((um_nonRL_nonConv - um_baseline)**2))
NRMSE_nonRL_denum = np.sqrt(np.sum((um_baseline)**2))
NRMSE_nonRL = NRMSE_nonRL_num / NRMSE_nonRL_denum

# ---- Calculate relative L2 Error, relL2Err

# Error of RL non-converged
relL2Err_RL = np.zeros(nrlz)
for irlz in range(nrlz):
    relL2Err_RL_num   = np.linalg.norm(um_RL_nonConv[:,irlz] - um_baseline, 2)
    relL2Err_RL_denum = np.linalg.norm(um_baseline, 2)
    relL2Err_RL[irlz] = relL2Err_RL_num / relL2Err_RL_denum
# Error of non-RL non-converged
relL2Err_nonRL_num    = np.linalg.norm(um_nonRL_nonConv - um_baseline, 2)
relL2Err_nonRL_denum  = np.linalg.norm(um_baseline, 2)
relL2Err_nonRL        = relL2Err_nonRL_num / relL2Err_nonRL_denum

# ------------------------ Build plots ------------------------

# ---- check all data have the same coordinates
assert (ydelta_nonRL_conv == ydelta_RL_nonConv).all()
assert (yplus_nonRL_conv  == yplus_RL_nonConv).all()
assert (ydelta_nonRL_nonConv == ydelta_RL_nonConv).all()
assert (yplus_nonRL_nonConv  == yplus_RL_nonConv).all()
ydelta = ydelta_RL_nonConv
yplus  = yplus_RL_nonConv

# ---- build plots
tEndAvg_nonConv_plots = tEndAvg_nonConv - tBeginAvg_nonConv
visualizer = ChannelVisualizer(postMultipleRlzDir)
visualizer.RL_u_mean_convergence(yplus[1:], rlzN_Arr, um_RL_nonConv[1:], urmsf_RL_nonConv[1:], um_nonRL_nonConv[1:], urmsf_nonRL_nonConv[1:], um_baseline[1:], urmsf_baseline[1:], tEndAvg_nonConv_plots, tEndAvg_baseline)
visualizer.RL_err_convergence(rlzN_Arr, NRMSE_RL, NRMSE_nonRL, tEndAvg_nonConv_plots, "NRMSE")
visualizer.RL_err_convergence(rlzN_Arr, relL2Err_RL, relL2Err_nonRL, tEndAvg_nonConv_plots, "RelL2Err")
visualizer.RL_Rij_convergence(ydelta, rlzN_Arr, 
                              Rkk_RL_nonConv,    lambda1_RL_nonConv,    lambda2_RL_nonConv,    lambda3_RL_nonConv,    xmap1_RL_nonConv,    xmap2_RL_nonConv,
                              Rkk_nonRL_nonConv, lambda1_nonRL_nonConv, lambda2_nonRL_nonConv, lambda3_nonRL_nonConv, xmap1_nonRL_nonConv, xmap2_nonRL_nonConv,
                              Rkk_baseline,      lambda1_baseline,      lambda2_baseline,      lambda3_baseline,      xmap1_baseline,      xmap2_baseline,
                              tEndAvg_nonConv_plots, tEndAvg_baseline)