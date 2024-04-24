import yaml
import sys
import os
import glob
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
    caseN        = sys.argv[i];        i+=1
    rlzN_min_RL  = int(sys.argv[i]);   i+=1
    rlzN_max_RL  = int(sys.argv[i]);   i+=1
    rlzN_step_RL = int(sys.argv[i]);   i+=1
    tBeginAvg    = float(sys.argv[i]); i+=1
    tEndAvg      = float(sys.argv[i]); i+=1
    dtActions    = 0.1
    print(f"Script parameters: \n" \
          f"- RL Case name: {caseN} \n" \
          f"- Realization Number Min RL: {rlzN_min_RL} \n" \
          f"- Realization Number Max RL: {rlzN_max_RL} \n" \
          f"- Realization Number Step RL: {rlzN_step_RL} \n" \
          f"- Time Begin Averaging non-converged (both RL and non-RL): {tBeginAvg} \n" \
          f"- Time End Averaging non-converged (both RL and non-RL): {tEndAvg} \n" \
          f"- dTime between actions: {dtActions}\n" \
    )
except :
    raise ValueError("Missing call arguments, should be: <1_case_name_RL>")

# --- post-processing directories to store results

# first and last realizations
rlzN_first   = rlzN_min_RL
rlzStr_first = f"{rlzN_first:05d}"
rlzN_last    = rlzN_max_RL
rlzStr_last  = f"{rlzN_last:05d}"

# post-processing directory
postDir = f"../../data/{caseN}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)

# post-processing sub-directory for multiples realizations comparison
postMultipleRlzDir = os.path.join(postDir, f"comparative_{rlzStr_first}_{rlzStr_last}")
if not os.path.exists(postMultipleRlzDir):
    os.mkdir(postMultipleRlzDir)

# data run directory
runDir    = os.path.join(f"../../data/{caseN}/run/")

# --- Get ODT input parameters ---

# for the first realization
odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
tBeginAvgInput = yml["params"]["tBeginAvg"]
dTimeStart  = yml["dumpTimesGen"]["dTimeStart"]
dTimeEnd    = get_effective_dTimeEnd(caseN, rlzStr_first) # dTimeEnd = yml["dumpTimesGen"]["dTimeEnd"] can lead to errors if dTimeEnd > tEnd
dTimeStep   = yml["dumpTimesGen"]["dTimeStep"]

assert tBeginAvg == tBeginAvgInput, f"Input argument 'tBeginAvg' = {tBeginAvg} must be equal to the input.yaml argument 'tBeginAvg' = {tBeginAvgInput} used for runtime statistics calculation"
if dTimeEnd < tEndAvg:
    print(f"ATTENTION: simulation ending time = {dTimeEnd} < expected tEndAvg = {tEndAvg} -> simulation has been truncated/terminated early.\n")
    tEndAvg = dTimeEnd

# --- Chosen averaging times ---

if tBeginAvg >= dTimeStart:
    averaging_times_plots = np.arange(tBeginAvg + dtActions, tEndAvg + 1e-8, dtActions) - tBeginAvg
else:
    averaging_times_plots = np.arange(dTimeStart + dtActions, tEndAvg + 1e-8, dtActions) - tBeginAvg
# There is no RL action for the first 10 dtimes:
timeRL = averaging_times_plots[10:]

# --------------- Get Actions and Rewards for RL training along realizations ---------------

rlzArr          = np.arange(rlzN_min_RL, rlzN_max_RL+1, rlzN_step_RL)
nrlz            = len(rlzArr) 
logFilenameList = [f"log_{irlz}.npz" for irlz in rlzArr]
logFilepathList = [os.path.join(runDir, logFilename) for logFilename in logFilenameList] 

# get data shape (from the first realization data)
data_0 = np.load(logFilepathList[0])
(nActSteps, nActDof) = data_0['act'].shape

# initialize rewards and actions arrays for all realizations
actions           = np.zeros([nrlz, nActSteps, nActDof])    # each rlz actions have shape [nActSteps, nActDof]
rewards_total     = np.zeros([nrlz, nActSteps,])            # each rlz rewards have shape [nActSteps,]
## rewards_bc     = np.zeros([nrlz, nActSteps,])            # each rlz rewards have shape [nActSteps,]
rewards_err_umean = np.zeros([nrlz, nActSteps,])            # each rlz rewards have shape [nActSteps,]
rewards_err_urmsf = np.zeros([nrlz, nActSteps,])            # each rlz rewards have shape [nActSteps,]
rewards_rhsfRatio = np.zeros([nrlz, nActSteps,])            # each rlz rewards have shape [nActSteps,]

# get rewards and actions data
for irlz in range(nrlz):
    
    data = np.load(logFilepathList[irlz])
    actions[irlz,:,:]         = data['act']                 # shape [nActSteps, nActDof]
    rewards_total[irlz,:]     = data['rew']                 # shape [nActSteps,]
    ## rewards_bc[irlz,:]     = data['rew_bc']              # shape [nActSteps,]
    rewards_rhsfRatio[irlz,:] = data['rew_rhsfRatio']       # shape [nActSteps,]
    try: # relL2Err reward penalties: only u-mean penalty term ('rew_err')
        rewards_err_umean[irlz,:] = data['rew_err']         # shape [nActSteps,]
        isUrmsfPenaltyTerm = False
    except KeyError:
        rewards_err_umean[irlz,:] = data['rew_err_umean']
        rewards_err_urmsf[irlz,:] = data['rew_err_urmsf']
        isUrmsfPenaltyTerm = True    

# --- Get reward from non-RL realization
# TODO
    
# --------------- Plot rewards and actions ---------------

visualizer = ChannelVisualizer(postMultipleRlzDir)
if not(isUrmsfPenaltyTerm):
    visualizer.build_RL_rewards_convergence(rlzArr, timeRL, rewards_total, rewards_err_umean, rewards_rhsfRatio)
else:
    visualizer.build_RL_rewards_convergence_v2(rlzArr, timeRL, rewards_total, rewards_err_umean, rewards_err_urmsf, rewards_rhsfRatio)
visualizer.build_RL_actions_convergence(rlzArr, timeRL, actions)

    

