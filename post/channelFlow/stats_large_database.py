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

#------------ ODT data ---------------

# --- Get ODT input parameters ---

with open("../../data/"+caseN+"/input/input.yaml") as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"]
dxmin = yml["params"]["dxmin"]
delta = yml["params"]["domainLength"] * 0.5
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
np.savetxt(fname, odt_data, 
           header="y+,         u+_mean,     v+_mean,     w+_mean,     u+_rmsf,     v+_rmsf,     w+_rmsf      "\
                  "R_xx+,       R_yy+,       R_zz+,       R_xy+,       R_xz+,       R_yz+",
           fmt='%12.5E')

print("Nominal Retau: ", Retau)
print("Actual  Retau: ", RetauOdt)

#------------ Get ODT statistics ---------------

filename_odt = "../../data/"+caseN+"/post/ODTstat.dat"
print(f"Getting ODT data from {filename_odt}")
odt = np.loadtxt(filename_odt)
y_odt    = odt[:,0] # y+
u_odt    = odt[:,1] # u+_mean

urmsf_odt = odt[:,4] # u+_rmsf
vrmsf_odt = odt[:,5] # v+_rmsf
wrmsf_odt = odt[:,6] # w+_rmsf

Rxx_odt   = odt[:,7] # R_xy+
Ryy_odt   = odt[:,8] # R_xz+
Rzz_odt   = odt[:,9] # R_yz+
Rxy_odt   = odt[:,10] # R_xy+
Rxz_odt   = odt[:,11] # R_xz+
Ryz_odt   = odt[:,12] # R_yz+

#------------ Get DNS statistics ---------------

if reynoldsNumber == 590:
    filename_dns = "DNS_statistics/Re590/dnsChannel_Re590_means.dat"
    print(f"Getting DNS-means data from {filename_dns}")
    dns_means = np.loadtxt(filename_dns)
    y_dns = dns_means[:,1] # y+
    u_dns = dns_means[:,2] # Umean normalized by U_tau (= u+_mean)

    filename_dns = "DNS_statistics/Re590/dnsChannel_Re590_reynolds_stress.dat"
    print(f"Getting DNS-reynolds data from {filename_dns}")
    dns_reynolds_stress = np.loadtxt(filename_dns)
    Rxx_dns = dns_reynolds_stress[:,2] # R_xx+, normalized by U_tau^2
    Ryy_dns = dns_reynolds_stress[:,3]
    Rzz_dns = dns_reynolds_stress[:,4]
    Rxy_dns = dns_reynolds_stress[:,5] # R_xy+
    Rxz_dns = dns_reynolds_stress[:,6] # R_xz+
    Ryz_dns = dns_reynolds_stress[:,7] # R_yz+

    urmsf_dns = np.sqrt(Rxx_dns) # = sqrt(mean(u'u')), normalized by U_tau^2
    vrmsf_dns = np.sqrt(Ryy_dns)
    wrmsf_dns = np.sqrt(Rzz_dns)

else:
    filename_dns = f"DNS_statistics/Re{reynoldsNumber}/profiles/Re{reynoldsNumber}.prof"
    dns   = np.loadtxt(filename_dns,comments="%")
    y_dns = dns[:,1] # y+
    u_dns = dns[:,2] # u+_mean 

    urmsf_dns = dns[:,3] # named as u'+, normalized by U_tau
    vrmsf_dns = dns[:,4] # named as v'+, normalized by U_tau
    wrmsf_dns = dns[:,5] # named as w'+, normalized by U_tau
    
    Rxx_dns   = urmsf_dns**2    
    Ryy_dns   = vrmsf_dns**2
    Rzz_dns   = wrmsf_dns**2
    Rxy_dns   = dns[:,10]
    Rxz_dns   = dns[:,11]
    Ryz_dns   = dns[:,12]


#--------------------------------------------------------------------------------------------

print(f"MAKING PLOT OF MEAN U PROFILE: ODT vs DNS in ../../data/{caseN}/post/u_mean.pdf" )

matplotlib.rcParams.update({'font.size':20, 'figure.autolayout': True}) #, 'font.weight':'bold'})

fig, ax = plt.subplots()

ax.semilogx(y_odt, u_odt, 'k-',  label=r'ODT')
ax.semilogx(y_dns, u_dns, 'k--', label=r'DNS')

ax.set_xlabel(r'$y^+$') #, fontsize=22)
ax.set_ylabel(r'$u^+$') #, fontsize=22)
ax.legend(loc='upper left', frameon=False, fontsize=16)
ax.set_ylim([0, 30])
ax.set_xlim([1, 1000])

plt.savefig(f"../../data/{caseN}/post/u_mean")

#--------------------------------------------------------------------------------------------

print(f"MAKING PLOT OF RMS VEL PROFILES: ODT vs DNS in ../../data/{caseN}/post/u_rmsf.pdf" )

matplotlib.rcParams.update({'font.size':20, 'figure.autolayout': True}) #, 'font.weight':'bold'})

fig, ax = plt.subplots()

ax.plot(y_odt,  urmsf_odt, 'k-',  label=r'$u_{rmsf}/u_\tau$')
ax.plot(y_odt,  vrmsf_odt, 'b--', label=r'$v_{rmsf}/u_\tau$')
ax.plot(y_odt,  wrmsf_odt, 'r:',  label=r'$w_{rmsf}/u_\tau$')

ax.plot(-y_dns, urmsf_dns, 'k-',  label='')
ax.plot(-y_dns, vrmsf_dns, 'b--', label='')
ax.plot(-y_dns, wrmsf_dns, 'r:',  label='')

ax.plot([0,0], [0,3], '-', linewidth=0.5, color='gray')
ax.arrow( 30, 0.2,  50, 0, head_width=0.05, head_length=10, color='gray')
ax.arrow(-30, 0.2, -50, 0, head_width=0.05, head_length=10, color='gray')
ax.text(  30, 0.3, "ODT", fontsize=14, color='gray')
ax.text( -80, 0.3, "DNS", fontsize=14, color='gray')

ax.set_xlabel(r'$y^+$')
ax.set_ylabel(r'$u_{i,rmsf}/u_\tau$')
ax.legend(loc='upper right', frameon=False, fontsize=16)
ax.set_xlim([-300, 300])
ax.set_ylim([0, 3])

plt.savefig(f"../../data/{caseN}/post/u_rmsf")

#--------------------------------------------------------------------------------------------

print(f"MAKING PLOT OF REYNOLDS STRESSES PROFILES (NOT-DIAGONAL): ODT vs DNS in ../../data/{caseN}/post/reynolds_stress.pdf" )

matplotlib.rcParams.update({'font.size':20, 'figure.autolayout': True}) #, 'font.weight':'bold'})

fig, ax = plt.subplots()

ax.plot(y_odt,  Rxy_odt, 'k-',  label=r'$R_{xy}/u_\tau^2$')
ax.plot(y_odt,  Rxz_odt, 'b--', label=r'$R_{xz}/u_\tau^2$')
ax.plot(y_odt,  Ryz_odt, 'r:',  label=r'$R_{yz}/u_\tau^2$')

ax.plot(-y_dns, Rxy_dns, 'k-',  label='')
ax.plot(-y_dns, Rxz_dns, 'b--', label='')
ax.plot(-y_dns, Ryz_dns, 'r:',  label='')

ax.plot([0,0], [-1,3], '-', linewidth=0.5, color='gray')
ax.arrow( 30, 0.2,  50, 0, head_width=0.05, head_length=10, color='gray')
ax.arrow(-30, 0.2, -50, 0, head_width=0.05, head_length=10, color='gray')
ax.text(  30, 0.3, "ODT", fontsize=14, color='gray')
ax.text( -80, 0.3, "DNS", fontsize=14, color='gray')

ax.set_xlabel(r'$y^+$')
ax.set_ylabel(r'$R_{ij}/u_\tau^2$')
ax.legend(loc='upper right', frameon=False, fontsize=16)
ax.set_xlim([-300, 300])
ax.set_ylim([-1, 3])

plt.savefig(f"../../data/{caseN}/post/reynolds_stress")

#--------------------------------------------------------------------------------------------

print(f"MAKING PLOT OF REYNOLDS STRESSES PROFILES (DIAGONAL): ODT vs DNS in ../../data/{caseN}/post/reynolds_stress.pdf" )

matplotlib.rcParams.update({'font.size':20, 'figure.autolayout': True}) #, 'font.weight':'bold'})

fig, ax = plt.subplots()

ax.plot(y_odt,  Rxx_odt, 'k-',  label=r'$R_{xx}/u_\tau^2$')
ax.plot(y_odt,  Ryy_odt, 'b--', label=r'$R_{yy}/u_\tau^2$')
ax.plot(y_odt,  Rzz_odt, 'r:',  label=r'$R_{zz}/u_\tau^2$')

ax.plot(-y_dns, Rxx_dns, 'k-',  label='')
ax.plot(-y_dns, Ryy_dns, 'b--', label='')
ax.plot(-y_dns, Rzz_dns, 'r:',  label='')

ax.plot([0,0], [-1,3], '-', linewidth=0.5, color='gray')
ax.arrow( 30, 0.2,  50, 0, head_width=0.05, head_length=10, color='gray')
ax.arrow(-30, 0.2, -50, 0, head_width=0.05, head_length=10, color='gray')
ax.text(  30, 0.3, "ODT", fontsize=14, color='gray')
ax.text( -80, 0.3, "DNS", fontsize=14, color='gray')

ax.set_xlabel(r'$y^+$')
ax.set_ylabel(r'$R_{ii}/u_\tau^2$')
ax.legend(loc='upper right', frameon=False, fontsize=16)
ax.set_xlim([-300, 300])
ax.set_ylim([-1, 8])

plt.savefig(f"../../data/{caseN}/post/reynolds_diagonal")