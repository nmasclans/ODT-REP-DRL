import yaml
import sys
import math
import os
import numpy as np
import pandas as pd

from utils import *
from ChannelVisualizer import ChannelVisualizer


# --- Define parameters

odt_path = os.environ.get("ODT_PATH")
eps      = 1e-5

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
    dt_gif_RL         = float(sys.argv[i]); i+=1
    dt_gif_nonRL      = 1.0
    print(f"Script parameters: \n" \
          f"- Re_tau: {Retau} \n" \
          f"- Case name RL: {caseN_RL} \n" \
          f"- Realization Number Min RL: {rlzN_min_RL} \n" \
          f"- Realization Number Max RL: {rlzN_max_RL} \n" \
          f"- Realization Number Step RL: {rlzN_step_RL} \n" \
          f"- Time Begin Averaging (both RL and non-RL): {tBeginAvg} \n" \
          f"- dt gif: {dt_gif_RL} \n" \
          f"- dt gif: {dt_gif_nonRL} \n" \
    )
except :
    raise ValueError("Missing call arguments, should be: <1_Re_tau> <2_case_name_RL> <5_realization_number_min_RL> <6_realization_number_max_RL> <7_realization_number_step_RL> <8_delta_time_gif_RL> <9_delta_time_gif_nonRL>")

# post-processing directory
postDir = f"{odt_path}/data/{caseN_RL}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)

# post-processing sub-directory for multiples realizations comparison
postMultipleRlzDir = os.path.join(postDir, f"comparative_{rlzN_min_RL:05d}_{rlzN_max_RL:05d}")
if not os.path.exists(postMultipleRlzDir):
    os.mkdir(postMultipleRlzDir)

# -------------------------------------------------------------------------
# ------------------- Get u-mean of all RL realizations -------------------
# -------------------------------------------------------------------------

# --- Get ODT input parameters ---

odtInputDataFilepath = f"{odt_path}/data/{caseN_RL}/input/input.yaml"
with open(odtInputDataFilepath) as ifile:
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
utau         = 1.0
kvisc        = yml["params"]["kvisc0"]
nunif        = yml["params"]["nunif"]
delta        = yml["params"]["domainLength"] / 2.0
dTimeStart   = yml["dumpTimesGen"]["dTimeStart"]
dTimeStep    = yml["dumpTimesGen"]["dTimeStep"]
dTimeEnd     = yml["dumpTimesGen"]["dTimeEnd"]
dTimeVec     = np.arange(dTimeStart, dTimeEnd+eps, dTimeStep)

# --- Realizations ---

# realizations identification
rlzN_Arr   = np.arange(rlzN_min_RL, rlzN_max_RL+1, rlzN_step_RL)
rlzStr_Arr = [f"{rlzN:05d}" for rlzN in rlzN_Arr]
nrlz       = len(rlzN_Arr)

# initialize rlz data
yu_RL        = np.zeros(nunif)
yplus_RL     = np.zeros(nunif)
ntk_RL       = []
tEnd_RL      = []
um_RL_dict   = {}
time_RL_dict = {}


# --- Get ODT RL rlz data ---

for irlz in range(nrlz):

    # realization data filenames
    irlzN            = rlzStr_Arr[irlz]
    tEnd_irlz        = get_effective_dTimeEnd(caseN_RL, irlzN) # dTimeEnd = yml["dumpTimesGen"]["dTimeEnd"] can lead to errors if dTimeEnd > tEnd
    dTimeVec_irlz    = np.arange(tBeginAvg, tEnd_irlz + eps, dt_gif_RL)
    ntk_irlz         = len(dTimeVec_irlz)
    dTimeVecIdx_irlz = []; dTimeVecStr_irlz = []
    for itk in range(ntk_irlz):
        ### tIdx = np.where(dTimeVec==dTimeVec_irlz[itk])[0][0] -> causes floating point errors, do 'np.isclose' with a certain tolerance instead
        tIdx = np.where(np.isclose(dTimeVec, dTimeVec_irlz[itk], atol=eps))[0][0]
        dTimeVecIdx_irlz.append(tIdx)
        dTimeVecStr_irlz.append(str(tIdx).zfill(5))
    if len(dTimeVecIdx_irlz) != ntk_irlz:
        raise ValueError(f"Not all averaging_times where found! \n{ntk_irlz}\n{dTimeVecIdx_irlz}")
    flist_irlz = [f"{odt_path}/data/{caseN_RL}/data/data_{irlzN}/statistics/stat_dmp_{s}.dat" for s in dTimeVecStr_irlz]
    print(f"\n--- Temporal convergence for RL: Realization #{irlz} - " + irlzN + " ---")
    print(f"--- End Time: {tEnd_irlz} ---")

    # initialize variables
    yu_irlz = np.zeros([nunif])
    um_irlz = np.zeros([nunif, ntk_irlz])
    
    # get data
    for i in range(ntk_irlz):
        data_stat = np.loadtxt(flist_irlz[i])
        # y-coord
        if i == 0:
            yu_irlz = data_stat[:,0]
        else:
            yu_aux  = data_stat[:,0]
            assert (yu_irlz == yu_aux).all()
        # u-mean
        um_irlz[:,i] = data_stat[:,1]

    # store realization data
    # y-coord
    if irlz == 0:
        yu_RL    = yu_irlz
        yplus_RL = yu_irlz * utau / kvisc
    else:
        assert (yu_RL == yu_irlz).all()
    # u-mean
    um_RL_dict[irlz]   = um_irlz
    # time
    time_RL_dict[irlz] = dTimeVec_irlz 
    ntk_RL.append(ntk_irlz)
    tEnd_RL.append(dTimeVec_irlz[-1])

# unify um and time data from each realization, 
# neglecting the initial tBeginAvg of irlz > 0, i.e. (- (len(ntk_RL)-1))
ntk_RL_total = np.sum(ntk_RL) - (len(ntk_RL)-1)
um_RL    = np.zeros([nunif, ntk_RL_total])
time_RL  = np.zeros(ntk_RL_total)
for irlz in range(nrlz):
    # store data
    if irlz == 0:
        itkIdx_initial  = 0
        itkIdx_final    = ntk_RL[irlz]
        um_RL[:, itkIdx_initial:itkIdx_final] = um_RL_dict[irlz][:,:] # at all time instants
        time_RL[itkIdx_initial:itkIdx_final]  = time_RL_dict[irlz][:] # at all time instants
    else:
        itkIdx_initial  = itkIdx_final
        itkIdx_final    = itkIdx_initial + (ntk_RL[irlz] - 1)             # "-1" to neglect initial tBeginAvg for irlz > 0
        um_RL[:, itkIdx_initial:itkIdx_final] = um_RL_dict[irlz][:,1:]    # "1:" to neglect initial tBegin for irlz > 0
        time_RL[itkIdx_initial:itkIdx_final]  = time_RL_dict[irlz][1:]    # "1:" to neglect initial tBegin for irlz > 0

# --- Mirror data ---

nunif2 = int(nunif/2)
nunifb, nunift = get_nunif2_walls(nunif, nunif2)
yu_half_RL     = yu_RL[:nunifb] + delta
yplus_half_RL  = yu_half_RL * utau / kvisc
um_half_RL     = np.zeros([nunifb, ntk_RL_total])
for itk in range(ntk_RL_total):
    um_half_RL[:,itk] = 0.5 * ( um_RL[:nunifb,itk] + np.flipud(um_RL[nunift:,itk]) )

# At this point, yu_RL, yplus_RL, um_RL, yu_half_RL, yplus_half_RL, um_half_RL, time_RL and tEnd_RL have been defined :)

# ---------------------------------------------------------------------------------------
# ------------------- Get u-mean of non-converged non-RL rlz-averaged -------------------
# --------------------- take data from tBeginAvg to sum(RL-rlz-time) --------------------
# ---------------------------------------------------------------------------------------

# --- Get ODT input parameters ---

odtInputDataFilepath = f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/input_reference.yaml"
with open(odtInputDataFilepath) as ifile:
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
utau         = 1.0
kvisc        = yml["params"]["kvisc0"]
nunif        = yml["params"]["nunif"]
delta        = yml["params"]["domainLength"] / 2.0
dTimeStart   = yml["dumpTimesGen"]["dTimeStart"]
dTimeStep    = yml["dumpTimesGen"]["dTimeStep"]
dTimeEnd     = yml["dumpTimesGen"]["dTimeEnd"]
dTimeVec     = np.arange(dTimeStart, dTimeEnd+eps, dTimeStep)

# --- Time instants, filenames
# tEnd for non-RL non-Conv, must be equivalent to the time computed along all RL realizations
assert dt_gif_nonRL == 1.0, "tEnd_nonRL is calculated assuming dt_gif_nonRL == 1"
tEnd_nonRL  = math.trunc(np.sum(tEnd_RL)) - tBeginAvg * (nrlz-1)
time_nonRL  = np.arange(tBeginAvg, tEnd_nonRL + eps, dt_gif_nonRL)
ntk         = len(time_nonRL)
dTimeVecIdx = []; dTimeVecStr = []
for itk in range(ntk):
    ### tIdx = np.where(dTimeVec==time_nonRL[itk])[0][0] -> causes floating point errors, do 'np.isclose' with a certain tolerance instead
    tIdx = np.where(np.isclose(dTimeVec, time_nonRL[itk], atol=eps))[0][0]
    dTimeVecIdx.append(tIdx)
    dTimeVecStr.append(str(tIdx).zfill(5))
if len(dTimeVecIdx) != ntk:
    raise ValueError(f"Not all averaging_times where found! \n{ntk}\n{dTimeVecIdx}")
flist = [f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/data_00003/stat_dmp_{s}.dat" for s in dTimeVecStr]
print(f"\n--- Temporal convergence for non-RL ---")
print(f"--- End Time: {tEnd_nonRL} ---")


# --- Get ODT non-RL rlz-avg data for certain (non-conv) time instants ---

# initialize variables
yu_nonRL = np.zeros([nunif])
um_nonRL = np.zeros([nunif, ntk])
    
# get data for each time instant 
for itk in range(ntk):

    data_stat = np.loadtxt(flist[itk])
    # y-coord
    if itk == 0:
        yu_nonRL    = data_stat[:,0]
        yplus_nonRL = yu_nonRL * utau / kvisc
    else:
        yu_aux  = data_stat[:,0]
        assert (yu_nonRL == yu_aux).all()
    # u-mean
    um_nonRL[:,itk] = data_stat[:,1]
    # time
    time = get_time(flist[itk])
    assert time == time_nonRL[itk]

# --- Mirror data ---

nunif2 = int(nunif/2)
nunifb, nunift   = get_nunif2_walls(nunif, nunif2)
yu_half_nonRL    = yu_nonRL[:nunifb] + delta
yplus_half_nonRL = yu_half_nonRL * utau / kvisc
um_half_nonRL    = np.zeros([nunifb, ntk])
for itk in range(ntk):
    um_half_nonRL[:,itk] = 0.5 * ( um_nonRL[:nunifb,itk] + np.flipud(um_nonRL[nunift:,itk]) )

# At this point, yu_nonRL, yplus_nonRL, um_nonRL, yu_half_nonRL, yplus_half_nonRL, um_half_nonRL, time_nonRL, tEnd_nonRL have been defined

# -------------------------------------------------------------------------------------------
# ------------------- Get u-mean Reference, from converged, non-RL, Rlz-Avg at tf = 1000 -------------------
# -------------------------------------------------------------------------------------------

# --- Get ODT input parameters ---

odtInputDataFilepath = f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/input_reference.yaml"
with open(odtInputDataFilepath) as ifile:
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
utau  = 1.0
kvisc = yml["params"]["kvisc0"]
nunif = yml["params"]["nunif"]
delta = yml["params"]["domainLength"] / 2.0

# --- Get ODT Reference data (converged, non-RL, rlz-averaged) ---

fstat     = f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/statistics_reference.dat"
data_stat = np.loadtxt(fstat)
yu_ref    = data_stat[:,0]
yplus_ref = yu_ref * utau / kvisc
um_ref    = data_stat[:,1]
tEnd_ref  = get_time(fstat)

# --- Mirror data ---
nunif2 = int(nunif/2)
nunifb, nunift = get_nunif2_walls(nunif, nunif2)
yu_half_ref    = yu_ref[:nunifb] + delta
yplus_half_ref = yu_half_ref * utau / kvisc
um_half_ref = 0.5 * ( um_ref[:nunifb] + np.flipud(um_ref[nunift:]) )

# At this point, yu_ref, yplus_ref, um_ref, yu_half_ref, yplus_half_ref, um_half_ref, tEnd_ref have been defined :)

# ---------------------------------------------------------------------------------------
# -------------------------------------- Build gif --------------------------------------
# ---------------------------------------------------------------------------------------

# velocity data
um_max = np.max([np.max(um_half_RL), np.max(um_half_nonRL)])

# time data
# Attention, if dt_gif_RL != dt_fig_nonRL, then ntk_RL != ntk_nonRL
ntk_RL    = len(time_RL)
ntk_nonRL = len(time_nonRL)
if ntk_RL == ntk_nonRL:
    # meaning dt_fig_RL == dt_fig_nonRL
    ntk = ntk_RL
elif ntk_RL > ntk_nonRL:
    dt_ratio = int(dt_gif_nonRL / dt_gif_RL) # e.g.  dt_ratio = 10 if dt_gif_nonRL = 1.0 and dt_gif_RL = 0.1
    assert dt_ratio > 1, "Assumed dt_gif_RL is a fraction of dt_gif_nonRL, i.e. ntk_RL > ntk_nonRL"
    time_nonRL_ext    = np.zeros(ntk_RL)
    um_nonRL_ext      = np.zeros([nunif,  ntk_RL])
    um_half_nonRL_ext = np.zeros([nunifb, ntk_RL])
    for itk in range(ntk_RL):
        time_nonRL_ext[itk]      = time_nonRL[int(itk/dt_ratio)]
        um_nonRL_ext[:,itk]      = um_nonRL[:,int(itk/dt_ratio)]
        um_half_nonRL_ext[:,itk] = um_half_nonRL[:,int(itk/dt_ratio)]
    time_nonRL = np.copy(time_nonRL_ext)
    um_nonRL = np.copy(um_nonRL_ext)
    um_half_nonRL = np.copy(um_half_nonRL_ext)
    ntk = ntk_RL
else:
    raise NotImplementedError()

# calculate averaging time
avg_time_RL     = time_RL    - tBeginAvg       
avg_time_nonRL  = time_nonRL - tBeginAvg           
avg_tEnd_ref    = tEnd_ref   - tBeginAvg       

# build gif frames for each averaging time
print("\nBuilding gif frames...")
visualizer = ChannelVisualizer(postMultipleRlzDir)
frames_um  = []
for itk in range(ntk):
    
    # print for-loop progress
    if (itk+1) % int(ntk/10) == 0:
        print(f"{itk/ntk*100:.0f}%")

    # build frames
    frames_um = visualizer.build_um_frame_RL_nonRL_ref(
        frames_um, 
        yplus_half_RL, yplus_half_nonRL, yplus_half_ref, 
        um_half_RL[:,itk], um_half_nonRL[:,itk], um_half_ref,
        avg_time_RL[itk], avg_time_nonRL[itk], avg_tEnd_ref, 
        ylim=[0.0, um_max]
    )

# Set duration for each frame
# 100ms for each frame, 10000ms (10s) for the last frame
durations = [100] * (len(frames_um) - 1) + [10000]

# save gif
filename = os.path.join(postMultipleRlzDir, "u_mean_convergence_RL_nonRL_Ref.gif")
print(f"\nMAKING GIF U-MEAN for RL vs Non-RL vs Ref along AVG. TIME in {filename}" )
frames_um[0].save(filename, save_all=True, append_images=frames_um[1:], duration=durations, loop=0)    
