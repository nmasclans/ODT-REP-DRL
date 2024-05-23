# TODO: add description
#
# Usage
# python3 anisotropy_tensor_odt_convergence.py [case_name] [reynolds_number] [delta_time_stats]

import os
import sys
import tqdm
import yaml

import numpy as np
import pandas as pd

from utils import *
from ChannelVisualizer import ChannelVisualizer

plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 14 )
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb} \usepackage{color}")

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---

try :
    caseN     = sys.argv[1]
    rlzN      = int(sys.argv[2])
    Retau     = int(sys.argv[3])
    dtAvg     = float(sys.argv[4])
    tBeginAvg = float(sys.argv[5])
    tEndAvg   = float(sys.argv[6])
    print(f"Script parameters: \n- Case name: {caseN} \n- Realization Number: {rlzN} \n- Retau: {Retau} \n- dt statistics anisotropy gifs: {dtAvg} \n- Time begin averaging: {tBeginAvg} \n- Time End Averaging: {tEndAvg} \n")
except :
    raise ValueError("Missing call arguments, should be: <case_name> <realization_number> <reynolds_number> <delta_time_stats_anisotropy_gifs> <time_begin_averaging> <time_end_averaging>")


# --- Define parameters ---

# main parameters
verbose = True
tensor_kk_tolerance   = 1.0e-8;	# [-]
eigenvalues_tolerance = 1.0e-8;	# [-]
nbins = 50;			            # [-]

# post-processing directory
postDir = f"../../data/{caseN}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)

# post-processing sub-directory for single realization
rlzStr = f"{rlzN:05d}"
postRlzDir = os.path.join(postDir, f"post_{rlzStr}")
if not os.path.exists(postRlzDir):
    os.mkdir(postRlzDir)

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
delta = domainLength * 0.5
utau  = 1.0
dTimeStart  = yml["dumpTimesGen"]["dTimeStart"]
dTimeEnd    = get_effective_dTimeEnd(caseN, rlzStr) # dTimeEnd = yml["dumpTimesGen"]["dTimeEnd"] can lead to errors if dTimeEnd > tEnd
dTimeStep   = yml["dumpTimesGen"]["dTimeStep"]

assert tBeginAvg == tBeginAvgInput, f"Input argument 'tBeginAvg' = {tBeginAvg} must be equal to the input.yaml argument 'tBeginAvg' = {tBeginAvgRt} used for runtime statistics calculation"
if dTimeEnd < tEndAvg:
    print(f"ATTENTION: simulation ending time = {dTimeEnd} < expected tEndAvg = {tEndAvg} -> simulation has been truncated/terminated early.\n")
    tEndAvg = dTimeEnd

inputParams = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "utau": utau,
               "caseN": caseN, "rlzStr": rlzStr, 
               'dTimeStart': dTimeStart, 'dTimeEnd': dTimeEnd, 'dTimeStep': dTimeStep, 
               'tBeginAvg': tBeginAvg, 'tEndAvg': tEndAvg} 

# --- Chosen averaging times ---

if tBeginAvg >= dTimeStart:
    averaging_times = np.arange(tBeginAvg, tEndAvg+1e-4, dtAvg).round(4)
else:
    averaging_times = np.arange(dTimeStart, tEndAvg+1e-4, dtAvg).round(4)
averaging_times_plots = averaging_times - tBeginAvg

# remove first time, as only 1 file is used to calculate the statistics, therefore 
# they are just instantaneous, and make the reynolds stress tensor not satisfy realizability conditions
### averaging_times = averaging_times[1:] 


# -------------------------------------------------------------------------
# ----------------------- ODT-Reference statistics ----------------------- 
# -------------------------------------------------------------------------

### (ODT-Reference) CONVERGED calculated-at-runtime statistics 
(ydelta_ref, yplus_ref, 
 um_ref, urmsf_ref,_, _,_,_, _,_,_,
 ufufm_ref, vfvfm_ref, wfwfm_ref, ufvfm_ref, ufwfm_ref, vfwfm_ref,
 _,_,_) \
    = get_odt_statistics_reference(inputParams)
(Rkk_ref, lambda1_ref, lambda2_ref, lambda3_ref, xmap1_ref, xmap2_ref) \
    = compute_reynolds_stress_dof(ufufm_ref, vfvfm_ref, wfwfm_ref, ufvfm_ref, ufwfm_ref, vfwfm_ref)
eigenvalues_ref = np.array([lambda1_ref, lambda2_ref, lambda3_ref]).transpose()

# -------------------------------------------------------------------------
# --------------------------- Runtime statistics --------------------------
# -------------------------------------------------------------------------

print("\n------ Calculate Rij dof from statistics calculated at runtime by ODT ------")

(ydelta_rt, yplus_rt, um_rt, urmsf_rt,_, _,_,_, _,_,_, Rxx_rt, Ryy_rt, Rzz_rt, Rxy_rt, Rxz_rt, Ryz_rt) = get_odt_statistics_rt_at_chosen_averaging_times(inputParams, averaging_times)

# --- Animation frames (gif) ---
visualizer      = ChannelVisualizer(postRlzDir)
frames_um_rt  = []; frames_urmsf_rt = []
frames_rkk_rt = []; frames_eig_rt = []; frames_xmap_coord_rt = []; frames_xmap_triang_rt = []
um_rt_max     = int(np.max(um_rt)+1)
urmsf_rt_max  = int(np.max(urmsf_rt)+1)

print("\nBuilding gif frames...")
ntk = len(averaging_times)
for i in range(ntk):
    
    # print for-loop progress
    if i % int(ntk/100) == 0:
        print(f"{i/ntk*100:.0f}%")

    # get runtime-calculated dof of Rij
    (Rkk_rt, lambda1_rt, lambda2_rt, lambda3_rt, xmap1_rt, xmap2_rt) = compute_reynolds_stress_dof(Rxx_rt[:,i], Ryy_rt[:,i], Rzz_rt[:,i], Rxy_rt[:,i], Rxz_rt[:,i], Ryz_rt[:,i])
    eigenvalues_rt = np.array([lambda1_rt, lambda2_rt, lambda3_rt]).transpose()
    
    # build frames
    frames_um_rt          = visualizer.build_um_frame(frames_um_rt, ydelta_rt[1:], ydelta_ref[1:], um_rt[1:,i], um_ref[1:], averaging_times_plots[i], ylim=[0.0, um_rt_max])
    frames_urmsf_rt       = visualizer.build_urmsf_frame(frames_urmsf_rt, ydelta_rt[1:], ydelta_ref[1:], urmsf_rt[1:,i], urmsf_ref[1:], averaging_times_plots[i], ylim=[0.0, urmsf_rt_max])
    frames_rkk_rt         = visualizer.build_reynolds_stress_tensor_trace_frame(frames_rkk_rt, ydelta_rt[1:-1], ydelta_ref[1:-1], Rkk_rt[1:-1], Rkk_ref[1:-1], averaging_times_plots[i])
    frames_eig_rt         = visualizer.build_anisotropy_tensor_eigenvalues_frame(frames_eig_rt, ydelta_rt[1:-1], ydelta_ref[1:-1], eigenvalues_rt[1:-1], eigenvalues_ref[1:-1], averaging_times_plots[i])
    frames_xmap_coord_rt  = visualizer.build_anisotropy_tensor_barycentric_xmap_coord_frame(frames_xmap_coord_rt, ydelta_rt[1:-1], ydelta_ref[1:-1], xmap1_rt[1:-1], xmap2_rt[1:-1], xmap1_ref[1:-1], xmap2_ref[1:-1], averaging_times_plots[i])
    frames_xmap_triang_rt = visualizer.build_anisotropy_tensor_barycentric_xmap_triang_frame(frames_xmap_triang_rt, ydelta_rt[1:-1], xmap1_rt[1:-1], xmap2_rt[1:-1], averaging_times_plots[i])

# -------------------------------------------------------------------------
# ------------------ Create the animation from the frames -----------------
# -------------------------------------------------------------------------

### # post-processed statistics
### filename = os.path.join(postRlzDir, "post/anisotropy_tensor_eigenvalues_odt_convergence_post.gif")
### print(f"\nMAKING GIF EIGENVALUES OF ANISOTROPY TENSOR for POST-PROCESSING calculations ALONG AVG. TIME in {filename}" )
### frames_eig_post[0].save(filename, save_all=True, append_images=frames_eig_post[1:], duration=100, loop=0)
### filename = os.path.join(postRlzDir, "anisotropy_tensor_barycentric_map_odt_convergence_post.gif")
### print(f"\nMAKING GIF OF BARYCENTRIC MAP OF ANISOTROPY TENSOR for POST-PROCESSING calculations ALONG AVG. TIME in {filename}" )
### frames_bar_post[0].save(filename, save_all=True, append_images=frames_bar_post[1:], duration=100, loop=0)

# runtime statistics
filename = os.path.join(postRlzDir, "u_mean_convergence.gif")
print(f"\nMAKING GIF U-MEAN for RUNTIME calculations along AVG. TIME in {filename}" )
frames_um_rt[0].save(filename, save_all=True, append_images=frames_um_rt[1:], duration=100, loop=0)    

filename = os.path.join(postRlzDir, "u_rmsf_convergence.gif")
print(f"\nMAKING GIF U-RMSF for RUNTIME calculations along AVG. TIME in {filename}" )
frames_urmsf_rt[0].save(filename, save_all=True, append_images=frames_urmsf_rt[1:], duration=100, loop=0)    

filename = os.path.join(postRlzDir, "reynolds_stress_tensor_trace_convergence.gif")
print(f"\nMAKING GIF TRACE/MAGNITUDE OF REYNOLDS STRESS TENSOR for RUNTIME calculations along AVG. TIME in {filename}" )
frames_rkk_rt[0].save(filename, save_all=True, append_images=frames_rkk_rt[1:], duration=100, loop=0)    

filename = os.path.join(postRlzDir, "anisotropy_tensor_eigenvalues_convergence.gif")
print(f"\nMAKING GIF EIGENVALUES OF ANISOTROPY TENSOR for RUNTIME calculations along AVG. TIME in {filename}" )
frames_eig_rt[0].save(filename, save_all=True, append_images=frames_eig_rt[1:], duration=100, loop=0)

print(f"\nMAKING GIF OF BARYCENTRIC MAP COORDINATES OF ANISOTROPY TENSOR for RUNTIME calculations along AVG. TIME in {filename}" )
filename = os.path.join(postRlzDir, "anisotropy_tensor_barycentric_map_coord_convergence.gif")
frames_xmap_coord_rt[0].save(filename, save_all=True, append_images=frames_xmap_coord_rt[1:], duration=100, loop=0)

print(f"\nMAKING GIF OF BARYCENTRIC MAP REALIZABLE TRIANGLE OF ANISOTROPY TENSOR for RUNTIME calculations along AVG. TIME in {filename}" )
filename = os.path.join(postRlzDir, "anisotropy_tensor_barycentric_map_triang_convergence.gif")
frames_xmap_triang_rt[0].save(filename, save_all=True, append_images=frames_xmap_triang_rt[1:], duration=100, loop=0)

