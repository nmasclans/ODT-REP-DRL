import yaml
import sys
import os
import numpy as np
import pandas as pd

from utils import *
from ChannelVisualizer import ChannelVisualizer
import matplotlib.pyplot as plt

plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 14 )
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb} \usepackage{color}")

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---

try :
    i = 1
    caseN_nonRL  = sys.argv[i];        i+=1
    rlzN_nonRL   = int(sys.argv[i]);   i+=1
    caseN_RL     = sys.argv[i];        i+=1
    rlzN_min_RL  = int(sys.argv[i]);   i+=1
    rlzN_max_RL  = int(sys.argv[i]);   i+=1
    rlzN_step_RL = int(sys.argv[i]);   i+=1
    Retau        = int(sys.argv[i]);   i+=1
    tEndAvg      = float(sys.argv[i]); i+=1
    print(f"Script parameters: \n" \
          f"- Case name non-RL: {caseN_nonRL} \n" \
          f"- Realization Number non-RL: {rlzN_nonRL} \n" \
          f"- Case name RL: {caseN_RL} \n" \
          f"- Realization Number Min RL: {rlzN_min_RL} \n" \
          f"- Realization Number Max RL: {rlzN_max_RL} \n" \
          f"- Realization Number Step RL: {rlzN_step_RL} \n" \
          f"- Re_tau: {Retau} \n" \
          f"- Time End Averaging: {tEndAvg} \n")
except :
    raise ValueError("Missing call arguments, should be: <case_name_nonRL> <case_name_RL> <realization_number_min_RL> <realization_number_max_RL> <reynolds_number> <time_end_averaging>")

# --- Get ODT input parameters ---

# first and last realizations
rlzN_first   = rlzN_min_RL
rlzStr_first = f"{rlzN_first:05d}"
rlzN_last    = rlzN_max_RL
rlzStr_last  = f"{rlzN_last:05d}"

# get input data from first realization
odtInputDataFilepath_RL  = f"../../data/{caseN_RL}/input/input.yaml"
with open(odtInputDataFilepath_RL) as ifile :
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
inputParams_RL = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
                  "caseN": caseN_RL, "rlzStr": rlzStr_first, 
                  "dTimeStart": dTimeStart, "dTimeEnd": dTimeEnd, "dTimeStep": dTimeStep, 
                  "tEndAvg": tEndAvg} 

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
um_RL    = np.zeros([int(nunif/2), nrlz])
urmsf_RL = np.zeros([int(nunif/2), nrlz])

for irlz in range(nrlz):

    # modify rlzN information in inputParams_RL:
    inputParams_RL["rlzStr"] = rlzStr_Arr[irlz]
    print(f"\n\n--- RL: Realization #{rlzN_Arr[irlz]}, Time " + str(inputParams_RL["tEndAvg"]) +  " ---")

    # get u-statistics data
    if irlz == 0:
        (ydelta, yplus, um_data, urmsf_data) = get_odt_udata_rt(inputParams_RL)
    else:
        (_, _, um_data, urmsf_data) = get_odt_udata_rt(inputParams_RL)
    
    # store realization data
    um_RL[:,irlz]    = um_data
    urmsf_RL[:,irlz] = urmsf_data

#------------ Get NON-CONVERGED runtime-calculated 'um' at t=tEndAvg for single ODT non-RL-realization ---------------

# --- Build inputParams for non-RL case and realization, other params are idem.
inputParams_nonRL = inputParams_RL.copy()
inputParams_nonRL['caseN'] = caseN_nonRL
inputParams_nonRL['rlzStr']    = f"{rlzN_nonRL:05d}"

# --- Get data
print(f"\n--- Non-RL: Realization #{rlzN_nonRL}, Time " + str(inputParams_nonRL["tEndAvg"]) + " ---")
(_, _, um_nonRL, urmsf_nonRL) = get_odt_udata_rt(inputParams_nonRL)

#------------ Get BASELINE runtime-calculated 'um' at t=dTimeEnd (>>tEndAvg used before) for single ODT non-RL-realization ---------------

# --- build input parameters, tEndAvg=dTimeEnd of the Baseline is >> tEndAvg used in previous NON-CONVERGED data
inputParams_baseline = inputParams_nonRL.copy()
inputParams_baseline["tEndAvg"]  = 500.0 # TODO: this value should be a variable, it is the dTimeEnd of the input yaml of nonRL case
inputParams_baseline["dTimeEnd"] = 500.0 # TODO: this value should be a variable, it is the dTimeEnd of the input yaml of nonRL case
# --- get data
print(f"\n--- Non-RL Baseline: Realization #{rlzN_nonRL}, Time " + str(inputParams_baseline["tEndAvg"]) + " ---")
(_, _, um_baseline, urmsf_baseline) = get_odt_udata_rt(inputParams_baseline)

# ------------------------ Build plots ------------------------
visualizer = ChannelVisualizer(postMultipleRlzDir)
visualizer.build_u_mean_tEndAvg_nRlz_RL_vs_nonRL_vs_baseline(yplus, rlzN_Arr, um_RL, urmsf_RL, um_nonRL, urmsf_nonRL, um_baseline, urmsf_baseline, inputParams_RL["tEndAvg"], inputParams_baseline["tEndAvg"])

