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
    caseN    = sys.argv[i];        i+=1
    runId    = int(sys.argv[i]);   i+=1
    envIdStr = "000"
    print(f"Script parameters: \n" \
          f"- RL Case name: {caseN} \n" \
          f"- RL Run Id: {runId} \n" \
          f"- RL Env Id String: {envIdStr} \n"
    )
except :
    raise ValueError("Missing call arguments, should be: <1_case_name_RL> <2_RL_run_id>")

# --- post-processing directories to store results

# post-processing directory
postDir = f"../../data/{caseN}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)

# post-processing sub-directory for run analysis
postRunDir = os.path.join(postDir, f"run_{runId}")
if not os.path.exists(postRunDir):
    os.mkdir(postRunDir)

# data run directory
runDir    = os.path.join("../../data/{caseN}/run/{runId}/")
runEnvDir = os.path.join(runDir, f"env_{envIdStr}")

# --- Get Actions and Rewards

logFilepathList = glob.glob(os.path.join(runEnvDir, "log_*.npz")) 
nlog = len(logFilepathList) 

for ilog in range(nlog):
    
    if ilog == 0:
        # TODO: continue code


