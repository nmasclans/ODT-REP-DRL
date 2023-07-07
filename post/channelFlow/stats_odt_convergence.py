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

import numpy as np
import glob as gb
import yaml
import sys
import os
from scipy.interpolate import interp1d

from ChannelVisualizer import ChannelVisualizer
from utils import get_dns_data, get_time


#--------------------------------------------------------------------------------------------

#------------ input parameters ---------------

try :
    caseN          = sys.argv[1]
    reynolds_number = int(sys.argv[2])
    delta_aver_time = int(sys.argv[3])
except :
    raise ValueError("Include the case name in the call")

if not os.path.exists("../../data/"+caseN+"/post") :
    os.mkdir("../../data/"+caseN+"/post")

#------------ ODT data ---------------

# --- Get ODT input parameters ---

with open("../../data/"+caseN+"/input/input.yaml") as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc  = yml["params"]["kvisc0"]
dxmin  = yml["params"]["dxmin"]
delta  = yml["params"]["domainLength"] * 0.5
Retau  = 1.0/kvisc
tEnd   = yml["params"]["tEnd"]
tStart = yml["dumpTimesGen"]["dTimeStart"]

# --- Get ODT computational data ---

flist = sorted(gb.glob('../../data/'+caseN+'/data/data_00000/dmp_*.dat'))

# Num points uniform grid
nunif  = int(1/dxmin)        # num. points uniform grid (using smallest grid size)   
nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)

# Averaging times
averaging_times = np.arange(tStart, tEnd+0.1, delta_aver_time)
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
um  = 0.5 * (um[:nunif2,:]  + np.flipud(um[nunif2:,:]))  # mirror data (symmetric)
vm  = 0.5 * (vm[:nunif2,:]  + np.flipud(vm[nunif2:,:]))
wm  = 0.5 * (wm[:nunif2,:]  + np.flipud(wm[nunif2:,:]))
u2m = 0.5 * (u2m[:nunif2,:] + np.flipud(u2m[nunif2:,:]))
v2m = 0.5 * (v2m[:nunif2,:] + np.flipud(v2m[nunif2:,:]))
w2m = 0.5 * (w2m[:nunif2,:] + np.flipud(w2m[nunif2:,:]))
uvm = 0.5 * (uvm[:nunif2,:] + np.flipud(uvm[nunif2:,:]))
uwm = 0.5 * (uwm[:nunif2,:] + np.flipud(uwm[nunif2:,:]))
vwm = 0.5 * (vwm[:nunif2,:] + np.flipud(vwm[nunif2:,:]))

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
yu = yu[:nunif2]    # plotting to domain center

# Re_tau of ODT data
dudy = (um[1,-1]-um[0,-1])/(yu[1]-yu[0])
utau = np.sqrt(kvisc * np.abs(dudy))
RetauOdt = utau * delta / kvisc

# scale y --> y+ (note: utau should be unity)
yuplus = yu * utau/kvisc    

print("Nominal Retau: ", Retau)
print("Actual  Retau (at simulation end):", RetauOdt)

#------------ DNS data ---------------

(ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns) \
    = get_dns_data(reynolds_number)

#--------------------------------------------------------------------------------------------

# Build plots

visualizer = ChannelVisualizer(caseN)
visualizer.build_u_mean_profile_odt_convergence(yuplus, yplus_dns, um, um_dns, averaging_times)
visualizer.build_u_rmsf_profile_odt_convergence(yuplus, yplus_dns, urmsf, vrmsf, wrmsf, urmsf_dns, vrmsf_dns, wrmsf_dns, averaging_times)
visualizer.build_reynolds_stress_diagonal_profile_odt_convergence(    yuplus, yplus_dns, ufufm, vfvfm, wfwfm, ufufm_dns, vfvfm_dns, wfwfm_dns, averaging_times)
visualizer.build_reynolds_stress_not_diagonal_profile_odt_convergence(yuplus, yplus_dns, ufvfm, ufwfm, vfwfm, ufvfm_dns, ufwfm_dns, vfwfm_dns, averaging_times)

