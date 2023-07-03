# Compute ODT mean and rms velocity profiles. Plot results versus DNS results from DNS_statistics database.
# Run as: python3 stats_large_database.py case_name reynolds_number
# Values are in wall units (y+, u+).
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

#--------------------------------------------------------------------------------------------

try :
    caseN          = sys.argv[1]
    reynoldsNumber = int(sys.argv[2])
except :
    raise ValueError("Include the case name in the call")

if not os.path.exists("../../data/"+caseN+"/post") :
    os.mkdir("../../data/"+caseN+"/post")

deltaTimeStats = 10 # seconds
tStart         = 50.0

#------------ ODT data ---------------

# --- Get ODT input parameters ---

with open("../../data/"+caseN+"/input/input.yaml") as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"]
dxmin = yml["params"]["dxmin"]
delta = yml["params"]["domainLength"] * 0.5
Retau = 1.0/kvisc
tEnd  = yml["params"]["tEnd"]

# --- Get ODT computational data ---

flist = sorted(gb.glob('../../data/'+caseN+'/data/data_00000/dmp_*.dat'))

# Num points uniform grid
nunif  = int(1/dxmin)        # num. points uniform grid (using smallest grid size)   
nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)

# Averaging times
timeStats = np.arange(tStart, tEnd+0.1, deltaTimeStats)
numTimeStats = len(timeStats)

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
um      = np.zeros(nunif, numTimeStats)   # mean velocity
vm      = np.zeros(nunif, numTimeStats)
wm      = np.zeros(nunif, numTimeStats)
u2m     = np.zeros(nunif, numTimeStats)   # mean square velocity (for rmsf and reynolds stresses)
v2m     = np.zeros(nunif, numTimeStats)
w2m     = np.zeros(nunif, numTimeStats)
uvm     = np.zeros(nunif, numTimeStats)   # mean velocity correlations (for reynolds stresses)
uwm     = np.zeros(nunif, numTimeStats)
vwm     = np.zeros(nunif, numTimeStats)

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
    # -> averaging time is stored as a comment in the first line of the .dat file, e.g.
    # "# time = 50.0"
    with open(ifile,"r") as file:
        firstLine = file.readline().strip()
    if firstLine.startswith("# time = "):
        averagingTime = float(firstLine.split(" = ")[1])
    else:
        print(f"No valid format found in the first line of {ifile}.")

    indexAveragingTime = np.where()


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
R_xx  = u2m - um*um
R_yy  = v2m - vm*vm
R_zz  = w2m - wm*wm
R_xy  = uvm - um*vm
R_xz  = uwm - um*wm
R_yz  = vwm - vm*wm

# root-mean-squared fluctuations (rmsf)
urmsf = np.sqrt(R_xx) 
vrmsf = np.sqrt(R_yy) 
wrmsf = np.sqrt(R_zz) 


yu += delta         # domain center is at 0; shift so left side is zero
yu = yu[:nunif2]    # plotting to domain center

dudy = (um[1]-um[0])/(yu[1]-yu[0])
utau = np.sqrt(kvisc * np.abs(dudy))
RetauOdt = utau * delta / kvisc

yu *= utau/kvisc    # scale y --> y+ (note: utau should be unity)

odt_data = np.vstack([yu,um,vm,wm,urmsf,vrmsf,wrmsf,R_xx,R_yy,R_zz,R_xy,R_xz,R_yz]).T
fname = "../../data/"+caseN+"/post/ODTstat.dat"
##np.savetxt(fname, odt_data, 
##           header="y+,         u+_mean,     v+_mean,     w+_mean,     u+_rmsf,     v+_rmsf,     w+_rmsf      "\
##                  "R_xx+,       R_yy+,       R_zz+,       R_xy+,       R_xz+,       R_yz+",
##           fmt='%12.5E')
##
##print("Nominal Retau: ", Retau)
##print("Actual  Retau: ", RetauOdt)

