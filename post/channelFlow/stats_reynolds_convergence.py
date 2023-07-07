# Description: 
# Analysis of uncertainties and convergence of turbulence statistical quantities
# Based on Oliver et al. (2014), Thompson et al. (2016), Andrade et al. (2018)

# Usage
# python3 stats_reynolds_convergence.py [case_name] [reynolds_number] [delta_time_stats]

# Arguments:
# case_name (str): Name of the case
# reynolds_number (int): reynolds number of the odt case, to get comparable dns result.
# delta_time_stats (int): delta time (seconds) between averaged profiles

# Example Usage:
# python3 stats_reynolds_convergence.py channel180 180 25

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
tStart = 50.0

# --- Get ODT computational data ---

flist = sorted(gb.glob('../../data/'+caseN+'/data/data_00000/dmp_*.dat'))

# Num points uniform grid
nunif  = int(1/dxmin)        # num. points uniform grid (using smallest grid size)   
nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)

# Averaging times
averaging_times = np.arange(tStart, tEnd+0.1, delta_aver_time)
num_aver_times  = len(averaging_times)

yu  = np.linspace(-delta,delta,nunif) # uniform grid in y-axis
um_aux  = np.zeros(nunif)   
vm_aux  = np.zeros(nunif)   
uvm_aux = np.zeros(nunif)   
um      = np.zeros([nunif, num_aver_times])   # mean velocity
vm      = np.zeros([nunif, num_aver_times])
uvm     = np.zeros([nunif, num_aver_times])   # mean velocity correlations (for reynolds stresses)

nfiles = 0
for ifile in flist :
    nfiles += 1

    data = np.loadtxt(ifile)
    y = data[:,0] # not normalized
    u = data[:,2] # normalized by u_tau=1, u is in fact u+
    v = data[:,3] # normalized by u_tau=1, v is in fact v+

    # interpolate to uniform grid
    uu = interp1d(y, u, fill_value='extrapolate')(yu)  
    vv = interp1d(y, v, fill_value='extrapolate')(yu)

    # update mean profiles
    um_aux  += uu
    vm_aux  += vv
    uvm_aux += uu*vv

    # Averaging time
    averaging_time = get_time(ifile)

    idx_averaging_time = np.where(averaging_time == averaging_times)[0]
    if len(idx_averaging_time)>0:
        idx = idx_averaging_time[0]
        um[:,idx]  = um_aux/nfiles   
        vm[:,idx]  = vm_aux/nfiles   
        uvm[:,idx] = uvm_aux/nfiles    

# mirror data (symmetric channel in y-axis)
um  = 0.5 * (um[:nunif2,:]  + np.flipud(um[nunif2:,:]))
vm  = 0.5 * (vm[:nunif2,:]  + np.flipud(vm[nunif2:,:]))
uvm = 0.5 * (uvm[:nunif2,:] + np.flipud(uvm[nunif2:,:]))

# Reynolds stress
R_xy  = uvm - um*vm

# y-coordinates
yu += delta         # domain center is at 0; shift so left side is zero
yu = yu[:nunif2]    # plotting to domain center

# ------------ scale y to y+ ------------

# Re_tau of ODT data
dudy = (um[1,-1]-um[0,-1])/(yu[1]-yu[0])
utau = np.sqrt(kvisc * np.abs(dudy))
RetauOdt = utau * delta / kvisc

# scale y --> y+ (note: utau should be unity)
yu *= utau/kvisc  # y+

print("Nominal Retau: ", Retau)
print("Actual  Retau (at simulation end):", RetauOdt)

# ----------- calculate dU+/dy+ -----------

# dU+/dy+ for each averaging time considered
# calculated using central-difference 
dumdy = np.zeros([nunif-2, num_aver_times])
print(dumdy.shape, um.shape, yu.shape)
for j in range(nunif-2):
    jj = j+1
    print(j,jj)
    print(dumdy[j,0],um[jj+1,0])
    dumdy[j,:] = (um[jj+1,:]-um[jj-1,:])/(yu[jj+1]-yu[jj-1])
print('shape:',dumdy.shape)


#------------ DNS data ---------------

(y_dns, u_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, Rxx_dns, Ryy_dns, Rzz_dns, Rxy_dns, Rxz_dns, Ryz_dns) \
    = get_dns_data(reynolds_number)


