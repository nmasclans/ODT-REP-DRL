import sys
import os
import numpy as np
import pandas as pd

from utils import *
from ChannelVisualizer import ChannelVisualizer

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---

try :
    i = 1
    caseN_RL          = sys.argv[i];        i+=1
    nohup_filename    = sys.argv[i];        i+=1
    actions_avg_freq  = int(sys.argv[i]);   i+=1
    print(f"Script parameters: \n" \
          f"- Case name RL: {caseN_RL} \n" \
          f"- Nohup filename for RL training run: {nohup_filename} \n" \
          f"- Actions averaging frequency, number of simulation steps for averaged actions kde: {actions_avg_freq} \n" \
    )
except :
    raise ValueError("Missing call arguments, should be: <1_Re_tau> <2_case_name_RL> <3_nohup_filename> <4_actions_avg_freq>")

# --- nohup and inputRL source files ---

### post-process on eckert:
#nohup_filepath   = os.path.join(f"../../data/{caseN_RL}/nohup", nohup_filename)
### post-process on triton
rl_path          = os.environ.get("RL_PATH")
nohup_filepath   = os.path.join(rl_path, "nohup", nohup_filename)
inputRL_filepath = os.path.join(f"../../data/{caseN_RL}/input", "inputRL.i")

# --- post-processing directories to store results

# post-processing directory
postDir = f"../../data/{caseN_RL}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)

# post-processing sub-directory for multiples realizations comparison
postNohupDir = os.path.join(postDir, f"post_{nohup_filename}")
if not os.path.exists(postNohupDir):
    os.mkdir(postNohupDir)

# --------------- Get Actions and Rewards for RL training along realizations ---------------

"""
Assumption: nohup lines regarding RL training process are structured as:

[parallel_env/step] *** STEP #2548 ***
[parallel_env/action] Actions: {'ctrl_y0': array([ 0.15302497, -0.5235988 ,  0.5235988 , -0.5235988 , -0.0148077 ,
        0.        ], dtype=float32)}
[parallel_env/evolve_simulation] Execute C++ ODT application /workspace/ODT/run/odt.x / Case channel180_RL1 / Realization #0
[parallel_env/evolve_simulation] MPI communication finished and disconnected
[parallel_env/evolve_simulation] numerical utau: 0.9906122530844218
[parallel_env/evolve_simulation] numerical bc: 7.128120421e-21
[parallel_env/evolve_simulation] Actuation #2548
[parallel_env/evolve_simulation] Rewards: {'ctrl_y0': 0.025562294553694742}
[parallel_env/evolve_simulation] 1st term: Relative L2 Error = 1.0296301e-05
[parallel_env/evolve_simulation] 2nd term: BCs Error = 7.128120421e-21
[parallel_env/evolve_simulation] 3rd term: rhsfRatio Error = 0.96414140414852
.... or ....
[parallel_env/evolve_simulation] 1st term: Relative L2 Error u-mean = 0.02556821003881191
[parallel_env/evolve_simulation] 2nd term: Relative L2 Error u-rmsf = 0.4296638197039632
[parallel_env/evolve_simulation] 3rd term: rhsfRatio Error = 0.6740214782443593
"""

with open(nohup_filepath, "r") as file:
    lines = file.readlines()

utau = []
bc   = []
rewards_total = []
rewards_relL2Err_umean = []
rewards_relL2Err_urmsf = []
rewards_rhsfRatio = []
actions = []
for iline in range(len(lines)):
    line = lines[iline]
    try:
        if "numerical utau: " in line:
            start_idx = line.find("numerical utau: ") + len("numerical utau: ")
            utau.append(float(line[start_idx:]))
        elif "numerical bc: " in line:
            start_idx = line.find("numerical bc: ") + len("numerical bc: ")
            bc.append(float(line[start_idx:]))
        elif "Rewards: {'ctrl_y0': " in line:
            start_idx = line.find("'ctrl_y0': ") + len("'ctrl_y0': ")
            end_idx   = line.find("}", start_idx)
            rewards_total.append(float(line[start_idx:end_idx]))
        elif "Relative L2 Error = " in line:
            start_idx = line.find("Relative L2 Error = ") + len("Relative L2 Error = ")
            rewards_relL2Err_umean.append(float(line[start_idx:]))
        elif "Relative L2 Error u-mean = " in line:
            start_idx = line.find("Relative L2 Error u-mean = ") + len("Relative L2 Error u-mean = ")
            rewards_relL2Err_umean.append(float(line[start_idx:]))
        elif "Relative L2 Error u-rmsf = " in line:
            start_idx = line.find("Relative L2 Error u-rmsf = ") + len("Relative L2 Error u-rmsf = ")
            rewards_relL2Err_urmsf.append(float(line[start_idx:]))
        elif "rhsfRatio Error" in line:
            start_idx = line.find("rhsfRatio Error = ") + len("rhsfRatio Error = ")
            rewards_rhsfRatio.append(float(line[start_idx:]))
        elif "Actions: {'ctrl_y0': array(" in line:
            start_idx = line.find("Actions: {'ctrl_y0': array(") + len("Actions: {'ctrl_y0': array([")
            if line.find("]", start_idx) == -1: # not found, actions span for 2 lines:
                str_1stline = line[start_idx:-len("\n")]
                line2nd     = lines[iline+1]
                str_2ndline = line2nd[:line2nd.find("]")]
                str_actions = str_1stline + str_2ndline
            else:
                end_idx   = line.find("]", start_idx)
                str_actions = line[start_idx:end_idx]
            list_str_actions   = str_actions.split(",")
            list_float_actions = [float(value.strip()) for value in list_str_actions]
            if len(list_float_actions) == 6:
                actions.append(list_float_actions)
        else:
            pass
    except ValueError:
        pass

# convert lists to np.arrays
utau                    = np.array(utau)
bc                      = np.array(bc)
rewards_total           = np.array(rewards_total)
rewards_relL2Err_umean  = np.array(rewards_relL2Err_umean)
rewards_relL2Err_urmsf  = np.array(rewards_relL2Err_urmsf)
rewards_rhsfRatio       = np.array(rewards_rhsfRatio)
actions                 = np.array(actions)

# --------------- Plot rewards and actions ---------------

visualizer = ChannelVisualizer(postNohupDir)
if len(rewards_relL2Err_urmsf) == 0: # no u-rmsf penalty term in the reward
    visualizer.build_RL_rewards_convergence_nohup(rewards_total, rewards_relL2Err_umean, rewards_rhsfRatio, inputRL_filepath)
else: # there is a u-rmsf penalty term in the reward
    visualizer.build_RL_rewards_convergence_nohup_v2(rewards_total, rewards_relL2Err_umean, rewards_relL2Err_urmsf, rewards_rhsfRatio, inputRL_filepath)
visualizer.build_RL_actions_convergence_nohup(actions, actions_avg_freq)