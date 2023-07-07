# Description: 
# Compute ODT mean and rmsf velocity profiles and reynolds stresses
# Plot results versus DNS results from DNS_statistics database.

# Usage
# python3 stats_odt_vs_dns.py [case_name] [reynolds_number]

# Arguments:
# case_name (str): Name of the case
# reynolds_number (int): reynolds number of the odt case, to get comparable dns result.

# Example Usage:
# python3 stats_odt_vs_dns.py channel180 180

# Comments:
# Values are in wall units (y+, u+) for both ODT and DNS results,
# Scaling is done in the input file (not explicitly here).

import numpy as np
import glob as gb
import yaml
import sys
import matplotlib
matplotlib.use('PDF')       
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from ChannelVisualizer import ChannelVisualizer

from utils import get_dns_data

#--------------------------------------------------------------------------------------------

try :
    caseN           = sys.argv[1]
    reynolds_number = int(sys.argv[2])
except :
    raise ValueError("Include the case name in the call")

if not os.path.exists("../../data/"+caseN+"/post") :
    os.mkdir("../../data/"+caseN+"/post")

#------------ ODT data ---------------

# --- Get ODT input parameters ---

with open("../../data/"+caseN+"/input/input.yaml") as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"]
dxmin = yml["params"]["dxmin"]
delta = yml["params"]["domainLength"] * 0.5
rho   = yml["params"]["rho0"]
Retau = 1.0/kvisc

# --- Get ODT computational data ---

flist = sorted(gb.glob('../../data/'+caseN+'/data/data_00000/dmp_*.dat'))

nunif  = int(1/dxmin)        # num. points uniform grid (using smallest grid size)   
nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)

nfiles = len(flist)          # num. files of instantaneous data, i.e. num. discrete time instants
yu  = np.linspace(-delta,delta,nunif) # uniform grid in y-axis
# empty vectors of time-averaged quantities
um  = np.zeros(nunif)        # mean velocity
vm  = np.zeros(nunif)
wm  = np.zeros(nunif)
u2m = np.zeros(nunif)        # mean square velocity (for rmsf and reynolds stresses)
v2m = np.zeros(nunif)
w2m = np.zeros(nunif)
uvm = np.zeros(nunif)        # mean velocity correlations (for reynolds stresses)
uwm = np.zeros(nunif)
vwm = np.zeros(nunif)

for ifile in flist :

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
    um  += uu                 
    vm  += vv
    wm  += ww
    u2m += uu*uu
    v2m += vv*vv
    w2m += ww*ww
    uvm += uu*vv
    uwm += uu*ww
    vwm += vv*ww

# means
um /= nfiles
vm /= nfiles
wm /= nfiles
um = 0.5*(um[:nunif2] + np.flipud(um[nunif2:]))  # mirror data (symmetric)
vm = 0.5*(vm[:nunif2] + np.flipud(vm[nunif2:]))
wm = 0.5*(wm[:nunif2] + np.flipud(wm[nunif2:]))

# squared means
u2m /= nfiles
v2m /= nfiles
w2m /= nfiles
u2m = 0.5*(u2m[:nunif2] + np.flipud(u2m[nunif2:]))
v2m = 0.5*(v2m[:nunif2] + np.flipud(v2m[nunif2:]))
w2m = 0.5*(w2m[:nunif2] + np.flipud(w2m[nunif2:]))

# velocity correlations
uvm /= nfiles
uwm /= nfiles
vwm /= nfiles
uvm = 0.5*(uvm[:nunif2] + np.flipud(uvm[nunif2:]))
uwm = 0.5*(uwm[:nunif2] + np.flipud(uwm[nunif2:]))
vwm = 0.5*(vwm[:nunif2] + np.flipud(vwm[nunif2:]))

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

# y-coordinate, y+
yu += delta         # domain center is at 0; shift so left side is zero
yu = yu[:nunif2]    # plotting to domain center
dudy = (um[1]-um[0])/(yu[1]-yu[0])
utau = np.sqrt(kvisc * np.abs(dudy))
RetauOdt = utau * delta / kvisc
yuplus = yu * utau/kvisc    # scale y --> y+ (note: utau should be unity)

odt_data = np.vstack([yu/delta,yuplus,um,vm,wm,urmsf,vrmsf,wrmsf,ufufm,vfvfm,wfwfm,ufvfm,ufwfm,vfwfm]).T
fname = "../../data/"+caseN+"/post/ODTstat.dat"
np.savetxt(fname, odt_data, 
           header="y/delta,   y+,         u+_mean,     v+_mean,     w+_mean,     u+_rmsf,     v+_rmsf,     w+_rmsf      "\
                  "<u'u'>+,   <v'v'>+,    <w'w'>+,     <u'v'>+,     <u'w'>+,     <v'w'>+ ",
           fmt='%12.5E')

print("Nominal Retau: ", Retau)
print("Actual  Retau: ", RetauOdt)

#------------ Get ODT statistics ---------------

filename_odt = "../../data/"+caseN+"/post/ODTstat.dat"
print(f"Getting ODT data from {filename_odt}")
odt = np.loadtxt(filename_odt)
ydelta_odt = odt[:,0]   # y/delta
yplus_odt  = odt[:,1]   # y+
um_odt     = odt[:,2]  # u+_mean

urmsf_odt  = odt[:,5]  # u+_rmsf
vrmsf_odt  = odt[:,6]  # v+_rmsf
wrmsf_odt  = odt[:,7]  # w+_rmsf

ufufm_odt  = odt[:,8]  # R_xx+
vfvfm_odt  = odt[:,9]  # R_yy+
wfwfm_odt  = odt[:,10] # R_zz+
ufvfm_odt  = odt[:,11] # R_xy+
ufwfm_odt  = odt[:,12] # R_xz+
vfwfm_odt  = odt[:,13] # R_yz+

# Stress decomposition: Viscous, Reynolds and Total stress
dumdy_odt = (um_odt[1:] - um_odt[:-1])/(ydelta_odt[1:] - ydelta_odt[:-1])
viscous_stress_odt  = kvisc * rho * dumdy_odt
reynolds_stress_odt = - rho * ufvfm_odt[:-1]
total_stress_odt    = viscous_stress_odt + reynolds_stress_odt

#------------ Get DNS statistics ---------------

(ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns) \
    = get_dns_data(reynolds_number)

# Stress decomposition: Viscous, Reynolds and Total stress
dumdy_dns = (um_dns[1:] - um_dns[:-1])/(ydelta_dns[1:] - ydelta_dns[:-1])
viscous_stress_dns  = kvisc * rho * dumdy_dns
reynolds_stress_dns = - rho * ufvfm_dns[:-1]
total_stress_dns    = viscous_stress_dns + reynolds_stress_dns

#--------------------------------------------------------------------------------------------

# Build plots

visualizer = ChannelVisualizer(caseN)
visualizer.build_u_mean_profile(yplus_odt, yplus_dns, um_odt, um_dns)
visualizer.build_u_rmsf_profile(yplus_odt, yplus_dns, urmsf_odt, vrmsf_odt, wrmsf_odt, urmsf_dns, vrmsf_dns, wrmsf_dns)
visualizer.build_reynolds_stress_not_diagonal_profile(yplus_odt, yplus_dns, ufvfm_odt, ufwfm_odt, vfwfm_odt, ufvfm_dns, ufwfm_dns, vfwfm_dns)
visualizer.build_reynolds_stress_diagonal_profile(yplus_odt, yplus_dns, ufufm_odt, vfvfm_odt, wfwfm_odt, ufufm_dns, vfvfm_dns, wfwfm_dns)
visualizer.build_stress_decomposition(ydelta_odt, ydelta_dns, viscous_stress_odt, reynolds_stress_odt, total_stress_odt, viscous_stress_dns, reynolds_stress_dns, total_stress_dns)
