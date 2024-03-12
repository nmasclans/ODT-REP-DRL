# Description: 
# Compute ODT mean and rmsf velocity profiles and reynolds stresses
# Plot results for increasing averaging time, to observe profiles convergence

# Usage
# python3 stats_odt_convergence.py [case_name] [reynolds_number] [delta_time_stats]

# Arguments:
# case_name (str): Name of the case
# reynolds_number (int): reynolds number of the odt case, to get comparable dns result.
# delta_time_stats (int): delta time (seconds) between averaged profiles

# Example Usage:
# python3 stats_odt_convergence.py channel180 180 25

# Comments:
# Values are in wall units (y+, u+) for both ODT and DNS results,
# Scaling is done in the input file (not explicitly here).

## todo: change many things of this script and related functions to incorporate the new implementation of 
## statistics in a fine uniform grid, and saved in stat_dmp_*****.dat

import numpy as np
import glob as gb
import yaml
import sys
import os
from scipy.interpolate import interp1d

from ChannelVisualizer import ChannelVisualizer
from utils import *

eps   = 1e-8 # small number for using np.arange 

#--------------------------------------------------------------------------------------------

#------------ input parameters ---------------

try :
    caseN = sys.argv[1]
    rlzN   = int(sys.argv[2])
    Retau = int(sys.argv[3])
    delta_aver_time = int(sys.argv[4])
except :
    raise ValueError("Missing call arguments, should be: <case_name> <realization_number> <reynolds_number> <delta_time_stats_stats>")

# post-processing directory
postDir = f"../../data/{caseN}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)
# post-processing sub-directory for single realization
rlzStr = f"{rlzN:05d}"
postRlzDir = os.path.join(postDir, f"post_{rlzStr}")
if not os.path.exists(postRlzDir):
    os.mkdir(postRlzDir)

#------------ ODT data ---------------

# --- Get ODT input parameters ---

odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho   = yml["params"]["rho0"]
dxmin = yml["params"]["dxmin"]
nunif = yml["params"]["nunif"]
domainLength = yml["params"]["domainLength"] 
delta = domainLength * 0.5
utau  = 1.0
dTimeStart  = yml["dumpTimesGen"]["dTimeStart"]
#dTimeEnd   = yml["dumpTimesGen"]["dTimeEnd"]
dTimeEnd    = get_provisional_tEnd(caseN)  # tEnd, valid even while running odt, instead of yml["params"]["tEnd"]
dTimeStep   = yml["dumpTimesGen"]["dTimeStep"]
inputParams = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif":nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "caseN": caseN, "utau": utau, 'dTimeStart':dTimeStart, 'dTimeEnd':dTimeEnd, 'dTimeStep':dTimeStep} 
# --- Get ODT computational data ---

flist = sorted(gb.glob('../../data/'+caseN+'/data/data_00000/dmp_*.dat'))

# Num points uniform grid
nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)
nunifb, nunift = get_nunif2_walls(nunif, nunif2)

# Averaging times
averaging_times = np.arange(dTimeStart, dTimeEnd+eps, delta_aver_time)
num_aver_times  = len(averaging_times)

yu  = np.linspace(-delta,delta,nunif) # uniform grid in y-axis
# empty vectors of time-averaged quantities
um_aux  = np.zeros(nunif)   
vm_aux  = np.zeros(nunif)   
wm_aux  = np.zeros(nunif)   
u2m_aux = np.zeros(nunif)   
v2m_aux = np.zeros(nunif)   
w2m_aux = np.zeros(nunif)   
uvm_aux = np.zeros(nunif)   
uwm_aux = np.zeros(nunif)   
vwm_aux = np.zeros(nunif)   
um      = np.zeros([nunif, num_aver_times])   # mean velocity
vm      = np.zeros([nunif, num_aver_times])
wm      = np.zeros([nunif, num_aver_times])
u2m     = np.zeros([nunif, num_aver_times])   # mean square velocity (for rmsf and reynolds stresses)
v2m     = np.zeros([nunif, num_aver_times])
w2m     = np.zeros([nunif, num_aver_times])
uvm     = np.zeros([nunif, num_aver_times])   # mean velocity correlations (for reynolds stresses)
uwm     = np.zeros([nunif, num_aver_times])
vwm     = np.zeros([nunif, num_aver_times])

nfiles = 0
for ifile in flist :
    nfiles += 1
    data = np.loadtxt(ifile)
    y = data[:,0] # not normalized
    u = data[:,2] # normalized by u_tau, u is in fact u+
    v = data[:,3] # normalized by u_tau, v is in fact v+
    w = data[:,4] # normalized by u_tau, w is in fact w+

    # interpolate to uniform grid
    uu = interp1d(y, u, fill_value='extrapolate')(yu)  
    vv = interp1d(y, v, fill_value='extrapolate')(yu)
    ww = interp1d(y, w, fill_value='extrapolate')(yu)

    # update mean profiles
    um_aux  += uu
    vm_aux  += vv
    wm_aux  += ww
    u2m_aux += uu*uu
    v2m_aux += vv*vv
    w2m_aux += ww*ww
    uvm_aux += uu*vv
    uwm_aux += uu*ww
    vwm_aux += vv*ww

    # Averaging time
    averaging_time = get_time(ifile)

    idx_averaging_time = np.where(averaging_time == averaging_times)[0]
    if len(idx_averaging_time)>0:
        idx = idx_averaging_time[0]
        um[:,idx]  = um_aux/nfiles   
        vm[:,idx]  = vm_aux/nfiles   
        wm[:,idx]  = wm_aux/nfiles   
        u2m[:,idx] = u2m_aux/nfiles    
        v2m[:,idx] = v2m_aux/nfiles    
        w2m[:,idx] = w2m_aux/nfiles    
        uvm[:,idx] = uvm_aux/nfiles    
        uwm[:,idx] = uwm_aux/nfiles    
        vwm[:,idx] = vwm_aux/nfiles    

# mirror data (symmetric channel in y-axis)
um  = 0.5 * (um[:nunifb,:]  + np.flipud(um[nunift:,:]))  # mirror data (symmetric)
vm  = 0.5 * (vm[:nunifb,:]  + np.flipud(vm[nunift:,:]))
wm  = 0.5 * (wm[:nunifb,:]  + np.flipud(wm[nunift:,:]))
u2m = 0.5 * (u2m[:nunifb,:] + np.flipud(u2m[nunift:,:]))
v2m = 0.5 * (v2m[:nunifb,:] + np.flipud(v2m[nunift:,:]))
w2m = 0.5 * (w2m[:nunifb,:] + np.flipud(w2m[nunift:,:]))
uvm = 0.5 * (uvm[:nunifb,:] + np.flipud(uvm[nunift:,:]))
uwm = 0.5 * (uwm[:nunifb,:] + np.flipud(uwm[nunift:,:]))
vwm = 0.5 * (vwm[:nunifb,:] + np.flipud(vwm[nunift:,:]))

# Reynolds stresses
ufufm = u2m - um*um # = <uf·uf>
vfvfm = v2m - vm*vm # = <vf·vf>
wfwfm = w2m - wm*wm # = <wf·wf>
ufvfm = uvm - um*vm # = <uf·vf>
ufwfm = uwm - um*wm # = <uf·wf>
vfwfm = vwm - vm*wm # = <vf·wf>

# root-mean-squared fluctuations (rmsf)
urmsf = np.sqrt(ufufm) 
vrmsf = np.sqrt(vfvfm) 
wrmsf = np.sqrt(wfwfm) 

# ------------ scale y to y+ ------------

# y-coordinates
yu += delta         # domain center is at 0; shift so left side is zero
yu = yu[:nunifb]    # plotting to domain center

# Re_tau of ODT data
dudy = (um[1,-1]-um[0,-1])/(yu[1]-yu[0])
utau = np.sqrt(kvisc * np.abs(dudy))
RetauOdt = utau * delta / kvisc

# scale y --> y+ (note: utau should be unity)
yuplus = yu * utau/kvisc    

print("\n(ODT) Nominal Retau: ", Retau)
print("(ODT) Actual  Retau (at simulation end):", RetauOdt)


#------------ ODT statistics-during-runtime data ---------------

(ydelta_rt, yplus_rt, um_rt, CI, yuplus_all, um_all, um_symmetric_all) = get_odt_statistics_during_runtime(inputParams, averaging_times)


#------------ DNS data ---------------

(ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns, viscous_stress_dns, reynolds_stress_dns, total_stress_dns, vt_u_plus_dns,               p_u_plus_dns) \
    = get_dns_statistics(Retau, inputParams)

#--------------------------------------------------------------------------------------------

# Build plots

visualizer = ChannelVisualizer(postRlzDir)
visualizer.build_u_mean_profile_odt_convergence(yuplus, yplus_dns, um, um_dns, averaging_times, y_odt_rt = yplus_rt, u_odt_rt = um_rt)
visualizer.build_u_rmsf_profile_odt_convergence(yuplus, yplus_dns, urmsf, vrmsf, wrmsf, urmsf_dns, vrmsf_dns, wrmsf_dns, averaging_times)
visualizer.build_reynolds_stress_diagonal_profile_odt_convergence(    yuplus, yplus_dns, ufufm, vfvfm, wfwfm, ufufm_dns, vfvfm_dns, wfwfm_dns, averaging_times)
visualizer.build_reynolds_stress_not_diagonal_profile_odt_convergence(yuplus, yplus_dns, ufvfm, ufwfm, vfwfm, ufvfm_dns, ufwfm_dns, vfwfm_dns, averaging_times)
visualizer.build_um_profile_symmetric_vs_nonsymmetric_odt_convergence(CI, yuplus_all, um_all, um_symmetric_all, averaging_times)
