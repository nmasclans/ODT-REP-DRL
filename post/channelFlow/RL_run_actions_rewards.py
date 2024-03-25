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

# --------------- Get Actions and Rewards for RL training along realizations ---------------

rlzArr          = np.arange(rlzN_min_RL, rlzN_max_RL+1, rlzN_step_RL)
nrlz            = len(rlzArr) 
logFilenameList = [f"log_{irlz}.npz" for irlz in rlzArr]
logFilepathList = [os.path.join(runDir, logFilename) for logFilename in logFilenameList] 

# get data shape (from the first realization data)
data_0 = np.load(logFilepathList[0])
(nActSteps, nActDof) = data_0['act'].shape

# initialize rewards and actions arrays for all realizations
actions = np.zeros([nrlz, nActSteps, nActDof])  # each rlz actions have shape [nActSteps, nActDof]
rewards_total = np.zeros([nrlz, nActSteps,])    # each rlz rewards have shape [nActSteps,]
rewards_bc    = np.zeros([nrlz, nActSteps,])    # each rlz rewards have shape [nActSteps,]
rewards_err   = np.zeros([nrlz, nActSteps,])    # each rlz rewards have shape [nActSteps,]

# get rewards and actions data
for irlz in range(nrlz):
    
    data = np.load(logFilepathList[irlz])
    actions[irlz,:,:]     = data['act']         # shape [nActSteps, nActDof]
    rewards_total[irlz,:] = data['rew']         # shape [nActSteps,]
    rewards_bc[irlz,:]    = data['rew_bc']      # shape [nActSteps,]
    rewards_err[irlz,:]   = data['rew_err']     # shape [nActSteps,]

# --- Get reward from non-RL realization
# TODO
    
# --------------- Plot rewards and actions ---------------

timeRL = np.arange(tBeginAvg + dtActions, tEndAvg + 1e-8, dtActions) - tBeginAvg

visualizer = ChannelVisualizer(postMultipleRlzDir)
visualizer.build_RL_rewards_convergence(rlzArr, timeRL, rewards_total, rewards_bc, rewards_err)
visualizer.build_RL_actions_convergence(rlzArr, timeRL, actions)

    

