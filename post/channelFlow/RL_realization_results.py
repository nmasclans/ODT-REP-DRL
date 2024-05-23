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
    caseN_RL          = sys.argv[i];        i+=1
    rlzN_min_RL       = int(sys.argv[i]);   i+=1
    rlzN_max_RL       = int(sys.argv[i]);   i+=1
    rlzN_step_RL      = int(sys.argv[i]);   i+=1
    tBeginAvg         = float(sys.argv[i]); i+=1
    tEndAvg_nonConv   = float(sys.argv[i]); i+=1
    tEndAvg_conv      = float(sys.argv[i]); i+=1
    dtAvg_nonRL       = float(sys.argv[i]); i+=1
    dtAvg_RL          = float(sys.argv[i]); i+=1
    print(f"Script parameters: \n" \
          f"- Re_tau: {Retau} \n" \
          f"- Case name RL: {caseN_RL} \n" \
          f"- Realization Number Min RL: {rlzN_min_RL} \n" \
          f"- Realization Number Max RL: {rlzN_max_RL} \n" \
          f"- Realization Number Step RL: {rlzN_step_RL} \n" \
          f"- Time Begin Averaging (both RL and non-RL): {tBeginAvg} \n" \
          f"- Time End Averaging non-converged (both RL and non-RL): {tEndAvg_nonConv} \n" \
          f"- Time End Averaging converged (non-RL, baseline): {tEndAvg_conv} \n"
          f"- dt averaging: {dtAvg_nonRL} \n" \
          f"- dt averaging: {dtAvg_RL} \n" \
    )
except :
    raise ValueError("Missing call arguments, should be: <1_Re_tau> <2_case_name_nonRL> <3_realization_number_nonRL> <4_case_name_RL> <5_realization_number_min_RL> <6_realization_number_max_RL> <7_realization_number_step_RL> <8_time_end_averaging_non_converged> <9_time_end_averaging_converged> <10_delta_time_stats_nonRL> <11_delta_time_stats_RL>")

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
kvisc          = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho            = yml["params"]["rho0"]
dxmin          = yml["params"]["dxmin"]
nunif          = yml["params"]["nunif"]
domainLength   = yml["params"]["domainLength"] 
tBeginAvgInput = yml["params"]["tBeginAvg"]
dTimeStart     = yml["dumpTimesGen"]["dTimeStart"]
dTimeStep      = yml["dumpTimesGen"]["dTimeStep"]
delta          = domainLength * 0.5
utau           = 1.0

assert tBeginAvg == tBeginAvgInput, f"Input argument 'tBeginAvg' = {tBeginAvg} must be equal to the input.yaml argument 'tBeginAvg' = {tBeginAvgInput} used for runtime statistics calculation"

inputParams_RL_nonConv = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
                          "caseN": caseN_RL, 
                          "dTimeStart": dTimeStart, "dTimeStep": dTimeStep, 
                          "tBeginAvg": tBeginAvg} 
print(f"\nInput params case RL + nonConv: {inputParams_RL_nonConv}")

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
ydelta_RL_nonConv           = np.zeros(int(nunif/2))         # half-channel
yplus_RL_nonConv            = np.zeros(int(nunif/2))         # half-channel
um_RL_nonConv_allChannel    = np.zeros([nunif, nrlz])        # all-channel > to calculate errors
urmsf_RL_nonConv_allChannel = np.zeros([nunif, nrlz])        # all-channel > to calculate errors
um_RL_nonConv               = np.zeros([int(nunif/2), nrlz]) # half-channel
urmsf_RL_nonConv            = np.zeros([int(nunif/2), nrlz]) # half-channel
Rkk_RL_nonConv              = np.zeros([int(nunif/2), nrlz]) # half-channel
lambda1_RL_nonConv          = np.zeros([int(nunif/2), nrlz]) # half-channel
lambda2_RL_nonConv          = np.zeros([int(nunif/2), nrlz]) # half-channel
lambda3_RL_nonConv          = np.zeros([int(nunif/2), nrlz]) # half-channel
xmap1_RL_nonConv            = np.zeros([int(nunif/2), nrlz]) # half-channel
xmap2_RL_nonConv            = np.zeros([int(nunif/2), nrlz]) # half-channel

for irlz in range(nrlz):

    print("\n--------------------------------------------------------------------------")
    # modify rlzN, dTimeEnd & tEndAvg information in inputParams_RL:
    dTimeEnd = get_effective_dTimeEnd(caseN_RL, rlzStr_Arr[irlz])
    if dTimeEnd < tEndAvg_nonConv:
        print(f"\nATTENTION: simulation ending time = {dTimeEnd} < expected tEndAvg = {tEndAvg_nonConv} -> simulation has been truncated/terminated early.")
        tEndAvg_ = dTimeEnd
    else:
        tEndAvg_ = tEndAvg_nonConv
    if irlz == nrlz - 1: # last realization
        tEndAvg_nonConv_RL = tEndAvg_
    inputParams_RL_nonConv["tEndAvg"]  = tEndAvg_
    inputParams_RL_nonConv["rlzStr"]   = rlzStr_Arr[irlz]
    inputParams_RL_nonConv["dTimeEnd"] = dTimeEnd
    print("\n--- RL: Realization #" + inputParams_RL_nonConv["rlzStr"] + ", Time: " + str(inputParams_RL_nonConv["tEndAvg"]) + " ---")
    print(f"\nInput parameters RL + non-conv: {inputParams_RL_nonConv}")

    # get u-statistics data
    # whole-channel data
    (_, _, um_RL_nonConv_allChannel_, urmsf_RL_nonConv_allChannel_) = get_odt_udata_rt(inputParams_RL_nonConv, half_channel_symmetry=False)
    # half-channel data, using half-channel symmetry
    (ydelta_RL_nonConv_, yplus_RL_nonConv_, 
     um_RL_nonConv_, urmsf_RL_nonConv_, _, _, _, _, _, _, _,
     ufufm_RL_nonConv_, vfvfm_RL_nonConv_, wfwfm_RL_nonConv_, ufvfm_RL_nonConv_, ufwfm_RL_nonConv_, vfwfm_RL_nonConv_, \
     _, _, _) \
        = get_odt_statistics_rt(inputParams_RL_nonConv)
    try:
        # calculate eigen-variables
        (Rkk_RL_nonConv_, lambda1_RL_nonConv_, lambda2_RL_nonConv_, lambda3_RL_nonConv_, xmap1_RL_nonConv_, xmap2_RL_nonConv_) \
            = compute_reynolds_stress_dof(ufufm_RL_nonConv_, vfvfm_RL_nonConv_, wfwfm_RL_nonConv_, ufvfm_RL_nonConv_, ufwfm_RL_nonConv_, vfwfm_RL_nonConv_)
        # store fluid variables for irlz
        um_RL_nonConv_allChannel[:,irlz]    = um_RL_nonConv_allChannel_
        urmsf_RL_nonConv_allChannel[:,irlz] = urmsf_RL_nonConv_allChannel_
        um_RL_nonConv[:,irlz]               = um_RL_nonConv_
        urmsf_RL_nonConv[:,irlz]            = urmsf_RL_nonConv_
        Rkk_RL_nonConv[:,irlz]              = Rkk_RL_nonConv_
        lambda1_RL_nonConv[:,irlz]          = lambda1_RL_nonConv_
        lambda2_RL_nonConv[:,irlz]          = lambda2_RL_nonConv_
        lambda3_RL_nonConv[:,irlz]          = lambda3_RL_nonConv_
        xmap1_RL_nonConv[:,irlz]            = xmap1_RL_nonConv_
        xmap2_RL_nonConv[:,irlz]            = xmap2_RL_nonConv_
    except np.linalg.LinAlgError as err:
        print(err)
        print(f"Fluid diverged - decomposition not possible (NaN values) - all fluid variables set to 0 everywhere for rlz #{irlz} by initialization")
    # store y coordinates
    if irlz == 0:
        ydelta_RL_nonConv = ydelta_RL_nonConv_
        yplus_RL_nonConv  = yplus_RL_nonConv_
    else:
        assert (ydelta_RL_nonConv == ydelta_RL_nonConv_).all()
        assert (yplus_RL_nonConv  == yplus_RL_nonConv_).all()

#------------ Get NON-CONVERGED runtime-calculated 'um' for chosen averaging times for each ODT RL-realization ---------------

# Attention: um_RL_nonConv_tk & rmsf_RL_nonConv_tk shape: [int(nunif/2), ntk_nonConv, nrlz]

if tBeginAvg >= dTimeStart:
    averaging_times_nonConv_RL = np.arange(tBeginAvg, tEndAvg_nonConv+1e-4, dtAvg_RL).round(4)
else:
    averaging_times_nonConv_RL = np.arange(dTimeStart, tEndAvg_nonConv+1e-4, dtAvg_RL).round(4)
averaging_times_nonConv_RL_plots = averaging_times_nonConv_RL - tBeginAvg
ntk_nonConv_RL = len(averaging_times_nonConv_RL)

um_RL_nonConv_allChannel_tk    = np.zeros([nunif, ntk_nonConv_RL, nrlz])   # all channel > to calculate errors
urmsf_RL_nonConv_allChannel_tk = np.zeros([nunif, ntk_nonConv_RL, nrlz])   # all channel > to calculate errors

for irlz in range(nrlz):

    print("\n--------------------------------------------------------------------------")
    # modify rlzN, dTimeEnd & tEndAvg information in inputParams_RL:
    dTimeEnd = get_effective_dTimeEnd(caseN_RL, rlzStr_Arr[irlz])
    if dTimeEnd < tEndAvg_nonConv:
        print(f"\nATTENTION: simulation ending time = {dTimeEnd} < expected tEndAvg = {tEndAvg_nonConv} -> simulation has been truncated/terminated early.")
        tEndAvg_ = dTimeEnd
    else:
        tEndAvg_ = tEndAvg_nonConv
    inputParams_RL_nonConv["tEndAvg"]  = tEndAvg_
    inputParams_RL_nonConv["rlzStr"]   = rlzStr_Arr[irlz]
    inputParams_RL_nonConv["dTimeEnd"] = dTimeEnd
    if tBeginAvg >= dTimeStart:
        averaging_times_nonConv_irlz = np.arange(tBeginAvg, tEndAvg_+1e-4, dtAvg_RL).round(4)
    else:
        averaging_times_nonConv_irlz = np.arange(dTimeStart, tEndAvg_+1e-4, dtAvg_RL).round(4)
    ntk_irlz = len(averaging_times_nonConv_irlz)
    print("\n--- Temporal convergence for RL: Realization #" + inputParams_RL_nonConv["rlzStr"] + " ---")
    print("--- Time: " + str(inputParams_RL_nonConv["tEndAvg"]) + " ---")
    print(f"\nInput parameters RL + non-conv: {inputParams_RL_nonConv}")

    # ODT statistics-during-runtime data

    (_, _, um_RL_nonConv_allChannel_tk_irlz, urmsf_RL_nonConv_allChannel_tk_irlz) \
        = get_odt_udata_rt_at_chosen_averaging_times(inputParams_RL_nonConv, averaging_times_nonConv_irlz, half_channel_symmetry=False)

    # store realization results
    um_RL_nonConv_allChannel_tk[:,:ntk_irlz,irlz]    = um_RL_nonConv_allChannel_tk_irlz
    urmsf_RL_nonConv_allChannel_tk[:,:ntk_irlz,irlz] = urmsf_RL_nonConv_allChannel_tk_irlz

#------------ Get NON-CONVERGED runtime-calculated 'um' at t=tEndAvg for averaged non-RL reference realizations ---------------

# --- Get input parameters
odtInputDataFilepath_nonRL_nonConv  = f"./ODT_reference/Re{Retau}/input_reference.yaml"
with open(odtInputDataFilepath_nonRL_nonConv) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
dTimeStart     = yml["dumpTimesGen"]["dTimeStart"]
dTimeEnd       = np.min([yml["dumpTimesGen"]["dTimeEnd"], yml["params"]["tEnd"]])
dTimeStep      = yml["dumpTimesGen"]["dTimeStep"]
tBeginAvgInput = yml["params"]["tBeginAvg"]
assert tBeginAvg == tBeginAvgInput, f"Input argument 'tBeginAvg' = {tBeginAvg} must be equal to the input.yaml argument 'tBeginAvg' = {tBeginAvgInput} used for runtime statistics calculation"

# --- Get data
print(f"\n--- Non-RL: Re{Retau}/data_rlz_avg,  Time: {tEndAvg_nonConv} ---")

dTimes        = np.round(np.arange(dTimeStart, dTimeEnd+1e-6, dTimeStep), 6)
tEndAvgDmpIdx = np.sum(tEndAvg_nonConv > dTimes) 
tEndAvgDmpStr = f"{tEndAvgDmpIdx:05d}"

### whole-channel data
fstat = f"./ODT_reference/Re{Retau}/data_rlz_avg/stat_dmp_{tEndAvgDmpStr}.dat"
print(f"\nGet statistics data from file: {fstat}")
data_stat = np.loadtxt(fstat)
# yu                           = data_stat[:,0]
um_nonRL_nonConv_allChannel    = data_stat[:,1] 
urmsf_nonRL_nonConv_allChannel = data_stat[:,2] 
ufufm_nonRL_nonConv_allChannel = data_stat[:,10]
vfvfm_nonRL_nonConv_allChannel = data_stat[:,11]
wfwfm_nonRL_nonConv_allChannel = data_stat[:,12]
ufvfm_nonRL_nonConv_allChannel = data_stat[:,13]
ufwfm_nonRL_nonConv_allChannel = data_stat[:,14]
vfwfm_nonRL_nonConv_allChannel = data_stat[:,15]

### half-channel data, using half-channel symmetry
nunif = len(um_nonRL_nonConv_allChannel); nunif2 = int(nunif/2)
nunifb, nunift = get_nunif2_walls(nunif, nunif2)
um_nonRL_nonConv    = 0.5 * ( um_nonRL_nonConv_allChannel[:nunifb]    + np.flipud(um_nonRL_nonConv_allChannel[nunift:]) )
urmsf_nonRL_nonConv = 0.5 * ( urmsf_nonRL_nonConv_allChannel[:nunifb] + np.flipud(urmsf_nonRL_nonConv_allChannel[nunift:]) )
ufufm_nonRL_nonConv = 0.5 * ( ufufm_nonRL_nonConv_allChannel[:nunifb] + np.flipud(ufufm_nonRL_nonConv_allChannel[nunift:]) )
vfvfm_nonRL_nonConv = 0.5 * ( vfvfm_nonRL_nonConv_allChannel[:nunifb] + np.flipud(vfvfm_nonRL_nonConv_allChannel[nunift:]) )
wfwfm_nonRL_nonConv = 0.5 * ( wfwfm_nonRL_nonConv_allChannel[:nunifb] + np.flipud(wfwfm_nonRL_nonConv_allChannel[nunift:]) )
ufvfm_nonRL_nonConv = 0.5 * ( ufvfm_nonRL_nonConv_allChannel[:nunifb] + np.flipud(ufvfm_nonRL_nonConv_allChannel[nunift:]) )
ufwfm_nonRL_nonConv = 0.5 * ( ufwfm_nonRL_nonConv_allChannel[:nunifb] + np.flipud(ufwfm_nonRL_nonConv_allChannel[nunift:]) )
vfwfm_nonRL_nonConv = 0.5 * ( vfwfm_nonRL_nonConv_allChannel[:nunifb] + np.flipud(vfwfm_nonRL_nonConv_allChannel[nunift:]) )
(Rkk_nonRL_nonConv, lambda1_nonRL_nonConv, lambda2_nonRL_nonConv, lambda3_nonRL_nonConv, xmap1_nonRL_nonConv, xmap2_nonRL_nonConv) \
    = compute_reynolds_stress_dof(ufufm_nonRL_nonConv, vfvfm_nonRL_nonConv, wfwfm_nonRL_nonConv, ufvfm_nonRL_nonConv, ufwfm_nonRL_nonConv, vfwfm_nonRL_nonConv)

#------------ Get BASELINE runtime-calculated 'um' at t=dTimeEnd (>>tEndAvg used before) for single ODT non-RL-realization ---------------

# --- get data
print(f"\n--- Non-RL Baseline: Re{Retau}/statistics_reference.dat, Time: {tEndAvg_conv} ---")

fstat = f"./ODT_reference/Re{Retau}/statistics_reference.dat"
tEndAvg_conv_inputData = get_time(fstat)
assert tEndAvg_conv_inputData == tEndAvg_conv, f"tEndAvg_conv_inputData = {tEndAvg_conv_inputData} but tEndAvg_conv = {tEndAvg_conv}"

### whole-channel data
print(f"\nGet statistics data from file: {fstat}")
data_stat = np.loadtxt(fstat)
# yu                           = data_stat[:,0]
um_nonRL_conv_allChannel    = data_stat[:,1] 
urmsf_nonRL_conv_allChannel = data_stat[:,2] 
# uvel_rhsfRatio
vm_nonRL_conv_allChannel    = data_stat[:,4]
vrmsf_nonRL_conv_allChannel = data_stat[:,5]
# vvel_rhsfRatio
vm_nonRL_conv_allChannel    = data_stat[:,7]
vrmsf_nonRL_conv_allChannel = data_stat[:,8]
# wvel_rhsfRatio
ufufm_nonRL_conv_allChannel = data_stat[:,10]
vfvfm_nonRL_conv_allChannel = data_stat[:,11]
wfwfm_nonRL_conv_allChannel = data_stat[:,12]
ufvfm_nonRL_conv_allChannel = data_stat[:,13]
ufwfm_nonRL_conv_allChannel = data_stat[:,14]
vfwfm_nonRL_conv_allChannel = data_stat[:,15]

### half-channel data, using half-channel symmetry
nunif = len(um_nonRL_nonConv_allChannel); nunif2 = int(nunif/2)
nunifb, nunift = get_nunif2_walls(nunif, nunif2)
um_nonRL_conv    = 0.5 * ( um_nonRL_conv_allChannel[:nunifb]    + np.flipud(um_nonRL_conv_allChannel[nunift:]) )
urmsf_nonRL_conv = 0.5 * ( urmsf_nonRL_conv_allChannel[:nunifb] + np.flipud(urmsf_nonRL_conv_allChannel[nunift:]) )
ufufm_nonRL_conv = 0.5 * ( ufufm_nonRL_conv_allChannel[:nunifb] + np.flipud(ufufm_nonRL_conv_allChannel[nunift:]) )
vfvfm_nonRL_conv = 0.5 * ( vfvfm_nonRL_conv_allChannel[:nunifb] + np.flipud(vfvfm_nonRL_conv_allChannel[nunift:]) )
wfwfm_nonRL_conv = 0.5 * ( wfwfm_nonRL_conv_allChannel[:nunifb] + np.flipud(wfwfm_nonRL_conv_allChannel[nunift:]) )
ufvfm_nonRL_conv = 0.5 * ( ufvfm_nonRL_conv_allChannel[:nunifb] + np.flipud(ufvfm_nonRL_conv_allChannel[nunift:]) )
ufwfm_nonRL_conv = 0.5 * ( ufwfm_nonRL_conv_allChannel[:nunifb] + np.flipud(ufwfm_nonRL_conv_allChannel[nunift:]) )
vfwfm_nonRL_conv = 0.5 * ( vfwfm_nonRL_conv_allChannel[:nunifb] + np.flipud(vfwfm_nonRL_conv_allChannel[nunift:]) )
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

#------------ Get non-RL runtime-calculated 'um' at chosen averaging times for multiple realizations ---------------

if tBeginAvg >= dTimeStart:
    averaging_times_conv = np.arange(tBeginAvg, tEndAvg_conv+1e-4, dtAvg_nonRL).round(4)
else:
    averaging_times_conv = np.arange(dTimeStart, tEndAvg_conv+1e-4, dtAvg_nonRL).round(4)
averaging_times_conv_plots = averaging_times_conv - tBeginAvg
ntk_conv            = len(averaging_times_conv)
idx_tEndAvg_nonConv = np.where(averaging_times_conv == tEndAvg_nonConv)[0][0]

dTimeVec = np.arange(dTimeStart, dTimeEnd+1e-4, dTimeStep).round(4)
averaging_times_idx = []
for t_idx in range(ntk_conv):
    averaging_times_idx.append(np.where(dTimeVec==averaging_times_conv[t_idx])[0][0])
averaging_times_str = [str(idx).zfill(5) for idx in averaging_times_idx]
if (len(averaging_times_str) != ntk_conv):
    raise ValueError("Not all averaging_times where found!")
nRef = 10
um_nonRL_conv_allRlz_allChannel_tk    = np.zeros([nRef, nunif, ntk_conv])    
urmsf_nonRL_conv_allRlz_allChannel_tk = np.zeros([nRef, nunif, ntk_conv])    

for iRef in range(nRef):
    for tk in range(ntk_conv):
        fstat = f"./ODT_reference/Re{Retau}/data_{iRef:05d}/stat_dmp_{tk:05d}.dat"
        ### whole-channel data
        data_stat = np.loadtxt(fstat)
        # yu = data_stat[:,0]
        um_nonRL_conv_allRlz_allChannel_tk[iRef, :, tk] = data_stat[:,1]
        urmsf_nonRL_conv_allRlz_allChannel_tk[iRef, :, tk] = data_stat[:,2]

### # ------------- Calculate errors (RL + non-RL) non-converged results vs converged baseline at t=tEndAvg -------------
### 
### # ---- Calculate NRMSE
### 
### num_points = len(um_nonRL_conv)
### # Error of RL non-converged
### NRMSE_RL = np.zeros(nrlz)
### for irlz in range(nrlz):
###     NRMSE_RL_num   = np.sqrt(np.sum((um_RL_nonConv_allChannel[:,irlz] - um_nonRL_conv_allChannel)**2))
###     NRMSE_RL_denum = np.sqrt(np.sum((um_nonRL_conv_allChannel)**2))
###     NRMSE_RL[irlz] = NRMSE_RL_num / NRMSE_RL_denum 
### # Error of non-RL non-converged
### NRMSE_nonRL_num   = np.sqrt(np.sum((um_nonRL_nonConv_allChannel - um_nonRL_conv_allChannel)**2))
### NRMSE_nonRL_denum = np.sqrt(np.sum((um_nonRL_conv_allChannel)**2))
### NRMSE_nonRL = NRMSE_nonRL_num / NRMSE_nonRL_denum
### 
### print("\n---------------------------------------------------------------------------------------------")
### print(f"\nNon-RL non-conv NRMSE errors at t={tEndAvg_nonConv} from 10 averaged realizations are:", NRMSE_nonRL)
### print(f"\nRL     non-conv NRMSE errors at t={tEndAvg_nonConv} for {nrlz} realizations are:", NRMSE_RL)
### print("---------------------------------------------------------------------------------------------\n")

# ------------- Calculate errors (RL + non-RL) non-converged results vs converged baseline at chosed averaging times -------------

um_NRMSE_denum    = np.linalg.norm(um_nonRL_conv_allChannel, 2)
urmsf_NRMSE_denum = np.linalg.norm(urmsf_nonRL_conv_allChannel, 2)

### RL non-converged:
# at tk time instants:
ntk_nonConv_RL = len(averaging_times_nonConv_RL)
um_NRMSE_RL_nonConv_tk    = np.zeros((ntk_nonConv_RL, nrlz))
urmsf_NRMSE_RL_nonConv_tk = np.zeros((ntk_nonConv_RL, nrlz))
for irlz in range(nrlz):
    for itk in range(ntk_nonConv_RL):
        um_NRMSE_RL_nonConv_tk[itk, irlz]    = np.linalg.norm(um_RL_nonConv_allChannel_tk[:,itk,irlz]    - um_nonRL_conv_allChannel, 2)    / um_NRMSE_denum
        urmsf_NRMSE_RL_nonConv_tk[itk, irlz] = np.linalg.norm(urmsf_RL_nonConv_allChannel_tk[:,itk,irlz] - urmsf_nonRL_conv_allChannel, 2) / urmsf_NRMSE_denum
# at tEndAvg_nonConv instant:
um_NRMSE_RL_nonConv_tEndAvg    = um_NRMSE_RL_nonConv_tk[idx_tEndAvg_nonConv,:]
urmsf_NRMSE_RL_nonConv_tEndAvg = urmsf_NRMSE_RL_nonConv_tk[idx_tEndAvg_nonConv,:]

# non-RL converged
# at tk time instants:
um_NRMSE_nonRL_conv_tk    = np.zeros((nRef, ntk_conv))
urmsf_NRMSE_nonRL_conv_tk = np.zeros((nRef, ntk_conv))
for iref in range(nRef):
    for itk in range(ntk_conv):
        um_NRMSE_nonRL_conv_tk[iref,itk]    = np.linalg.norm(um_nonRL_conv_allRlz_allChannel_tk[iref,:,itk]    - um_nonRL_conv_allChannel, 2)    / um_NRMSE_denum
        urmsf_NRMSE_nonRL_conv_tk[iref,itk] = np.linalg.norm(urmsf_nonRL_conv_allRlz_allChannel_tk[iref,:,itk] - urmsf_nonRL_conv_allChannel, 2) / urmsf_NRMSE_denum
# at tEndAvg_nonConv instant, and averaged along all non-RL realizations
um_NRMSE_nonRL_nonConv_tEndAvg = np.mean(um_NRMSE_nonRL_conv_tk[:,idx_tEndAvg_nonConv])
urmsf_NRMSE_nonRL_nonConv_tEndAvg = np.mean(urmsf_NRMSE_nonRL_conv_tk[:,idx_tEndAvg_nonConv])

print("\n---------------------------------------------------------------------------------------------")
print(f"\nNon-RL non-conv NRMSE errors at t={tEndAvg_nonConv} averaged from {nRef} realizations are:", um_NRMSE_nonRL_nonConv_tEndAvg)
print(f"\nRL     non-conv NRMSE errors at t={tEndAvg_nonConv} for {nrlz} realizations are:", um_NRMSE_RL_nonConv_tEndAvg)
print("\n---------------------------------------------------------------------------------------------\n")

# ------------------------ Build plots ------------------------

ydelta = ydelta_RL_nonConv
yplus  = yplus_RL_nonConv

tEndAvg_nonConv_nonRL_plots = tEndAvg_nonConv - tBeginAvg
tEndAvg_nonConv_RL_plots    = tEndAvg_nonConv_RL - tBeginAvg

# ---- build plots
tEndAvg_conv_plots    = tEndAvg_conv - tBeginAvg
visualizer = ChannelVisualizer(postMultipleRlzDir)
visualizer.RL_u_mean_convergence(yplus[1:], rlzN_Arr, 
                                 um_RL_nonConv[1:], urmsf_RL_nonConv[1:], um_nonRL_nonConv[1:], urmsf_nonRL_nonConv[1:], um_baseline[1:], urmsf_baseline[1:], 
                                 um_NRMSE_RL_nonConv_tEndAvg, urmsf_NRMSE_RL_nonConv_tEndAvg, um_NRMSE_nonRL_nonConv_tEndAvg, urmsf_NRMSE_nonRL_nonConv_tEndAvg,
                                 tEndAvg_nonConv_RL_plots, tEndAvg_nonConv_nonRL_plots, tEndAvg_conv_plots)
visualizer.RL_err_convergence(rlzN_Arr, um_NRMSE_RL_nonConv_tEndAvg, um_NRMSE_nonRL_nonConv_tEndAvg, tEndAvg_nonConv_nonRL_plots, "NRMSE(um+)")
visualizer.RL_err_convergence_along_time(rlzN_Arr, um_NRMSE_RL_nonConv_tk,    um_NRMSE_nonRL_conv_tk[:,:-1],    averaging_times_nonConv_RL_plots, averaging_times_conv_plots[:-1], {"title": "NRMSE_umean", "ylabel": r"$\textrm{NRMSE}(\overline{u}^{+})$"})
visualizer.RL_err_convergence_along_time(rlzN_Arr, urmsf_NRMSE_RL_nonConv_tk, urmsf_NRMSE_nonRL_conv_tk[:,:-1], averaging_times_nonConv_RL_plots, averaging_times_conv_plots[:-1], {"title": "NRMSE_urmsf", "ylabel": r"$\textrm{NRMSE}(u^{+}_{\textrm{rms}})$"})
visualizer.RL_Rij_convergence(ydelta[1:-1], rlzN_Arr, 
                              Rkk_RL_nonConv[1:-1],    lambda1_RL_nonConv[1:-1],    lambda2_RL_nonConv[1:-1],    lambda3_RL_nonConv[1:-1],    xmap1_RL_nonConv[1:-1],    xmap2_RL_nonConv[1:-1],
                              Rkk_nonRL_nonConv[1:-1], lambda1_nonRL_nonConv[1:-1], lambda2_nonRL_nonConv[1:-1], lambda3_nonRL_nonConv[1:-1], xmap1_nonRL_nonConv[1:-1], xmap2_nonRL_nonConv[1:-1],
                              Rkk_baseline[1:-1],      lambda1_baseline[1:-1],      lambda2_baseline[1:-1],      lambda3_baseline[1:-1],      xmap1_baseline[1:-1],      xmap2_baseline[1:-1],
                              tEndAvg_nonConv_nonRL_plots, tEndAvg_baseline)