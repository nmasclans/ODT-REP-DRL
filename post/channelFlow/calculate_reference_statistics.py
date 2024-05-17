"""
Run as:
python3 calculate_reference_statistics.py 180 10 361 901 100
python3 calculate_reference_statistics.py 590 10 1181 901 100

Assumption: 
Statistics stat_dmp_xxxxx.dat files have data columns as:
#         1_posUnif        2_uvel_mean        3_uvel_rmsf   4_uvel_rhsfRatio        5_vvel_mean        6_vvel_rmsf   7_vvel_rhsfRatio        8_wvel_mean        9_wvel_rmsf  10_wvel_rhsfRatio             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz
"""

import os
import sys

import glob as gb
import numpy as np
import matplotlib.pyplot as plt

from utils import get_time

# ---

# ODT repository path
odt_path = os.environ.get("ODT_PATH")

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 16)
plt.rc( 'axes',       labelsize = 16)
plt.rc( 'legend',     fontsize = 10)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')

# --- Get CASE parameters ---

try :
    Retau     = int(sys.argv[1])    # Re_tau = 180, 395, 590
    numRefRlz = int(sys.argv[2])    # number of reference realizations, stored in ./ODT_Reference/Re{Re_tau}/data_xxxxx
    nunif     = int(sys.argv[3])    # number of uniform grid points, common to all realizations
    ntk       = int(sys.argv[4])    # num stat_dmp_xxxxx.dat files per realization, 
                                    # i.e. number of time instances at which statistics data is stored
                                    # e.g. if last dmp file is dmp_00900, set ntk=901 to account for dmp_00000. 
    tBeginAvg = float(sys.argv[5])
except :
    raise ValueError("Missing call arguments")

# initialize data arrays
timeUnif  = np.zeros([numRefRlz, ntk])
posUnif   = np.zeros([numRefRlz, ntk, nunif])
uvel_mean = np.zeros([numRefRlz, ntk, nunif])
uvel_rmsf = np.zeros([numRefRlz, ntk, nunif])
# uvel_rhsfRatio = ...
vvel_mean = np.zeros([numRefRlz, ntk, nunif])
vvel_rmsf = np.zeros([numRefRlz, ntk, nunif])
# vvel_rhsfRatio = ...
wvel_mean = np.zeros([numRefRlz, ntk, nunif])
wvel_rmsf = np.zeros([numRefRlz, ntk, nunif])
# wvel_rhsfRatio = ...
Rxx       = np.zeros([numRefRlz, ntk, nunif])
Ryy       = np.zeros([numRefRlz, ntk, nunif])
Rzz       = np.zeros([numRefRlz, ntk, nunif])
Rxy       = np.zeros([numRefRlz, ntk, nunif])
Rxz       = np.zeros([numRefRlz, ntk, nunif])
Ryz       = np.zeros([numRefRlz, ntk, nunif])

for irlz in range(numRefRlz):
    print(f"irlz = {irlz}")
    
    # get all stat_dmp_*.dat files of data_xxxxx directory of rlz xxxxx 
    dir_path = f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/data_{irlz:05d}/"
    flist    = sorted(gb.glob(os.path.join(dir_path,"stat_dmp_*.dat")))
    nfiles   = len(flist)
    assert ntk == nfiles, f"ntk = != nfiles, with ntk = {ntk} and nfiles = {nfiles}"

    # get data:
    for itk in range(ntk):
        ifile = flist[itk]
        data      = np.loadtxt(ifile)
        timeUnif[irlz, itk]     = get_time(ifile)
        assert nunif == data[:,0].size, f"nunif != num points uniform grid of file {ifile} from rlz {irlz} at time index {itk}"
        posUnif[irlz, itk, :]   = data[:,0]
        uvel_mean[irlz, itk, :] = data[:,1]
        uvel_rmsf[irlz, itk, :] = data[:,2]
        # uvel_rhsfRatio ...    = data[:,3]
        vvel_mean[irlz, itk, :] = data[:,4]
        vvel_rmsf[irlz, itk, :] = data[:,5]
        # vvel_rhsfRatio ...    = data[:,6]
        wvel_mean[irlz, itk, :] = data[:,7]
        wvel_rmsf[irlz, itk, :] = data[:,8]
        # wvel_rhsfRatio ...    = data[:,9]
        Rxx[irlz, itk, :]       = data[:,10]
        Ryy[irlz, itk, :]       = data[:,11]
        Rzz[irlz, itk, :]       = data[:,12]
        Rxy[irlz, itk, :]       = data[:,13]
        Rxz[irlz, itk, :]       = data[:,14]
        Ryz[irlz, itk, :]       = data[:,15]

# ensure posUnif[:,:,i] is the same value, for each i-position
collapsedPosUnif = posUnif.reshape((posUnif.shape[0] * posUnif.shape[1], posUnif.shape[2]))
isEqual = np.allclose(collapsedPosUnif[0], collapsedPosUnif, atol=1e-8)
if isEqual:
    posUnif = posUnif[0,0,:]
else:
    raise ValueError("Not all elements of posUnif[:, :, i] are approximately equal for each i.")
# ensure simulation time along realization is the same for each rlz
isEqual = np.allclose(timeUnif[0,:], timeUnif, atol=1e-8)
if isEqual:
    timeUnif = timeUnif[0,:]
else:
    raise ValueError("Not all elements of timeUnif[:, i] are approximately equal for each i.")

# calculate fields averaged along realizations: from shape [numRefRlz, ntk, nunif] to [ntk, nunif] 
rlzAvg_uvel_mean = np.mean(uvel_mean, axis=0) # shape: [ntk, nunif]
rlzAvg_uvel_rmsf = np.mean(uvel_rmsf, axis=0) 
rlzAvg_vvel_mean = np.mean(vvel_mean, axis=0) 
rlzAvg_vvel_rmsf = np.mean(vvel_rmsf, axis=0) 
rlzAvg_wvel_mean = np.mean(wvel_mean, axis=0) 
rlzAvg_wvel_rmsf = np.mean(wvel_rmsf, axis=0) 
rlzAvg_Rxx       = np.mean(Rxx, axis=0) 
rlzAvg_Ryy       = np.mean(Ryy, axis=0) 
rlzAvg_Rzz       = np.mean(Rzz, axis=0) 
rlzAvg_Rxy       = np.mean(Rxy, axis=0) 
rlzAvg_Rxz       = np.mean(Rxz, axis=0) 
rlzAvg_Ryz       = np.mean(Ryz, axis=0)

# calculate averaged converged fields: realization-averaged field at [ntk=-1, nunif=:]
conv_uvel_mean   = rlzAvg_uvel_mean[-1,:] # shape [nunif]
conv_uvel_rmsf   = rlzAvg_uvel_rmsf[-1,:]
conv_vvel_mean   = rlzAvg_vvel_mean[-1,:]
conv_vvel_rmsf   = rlzAvg_vvel_rmsf[-1,:]
conv_wvel_mean   = rlzAvg_wvel_mean[-1,:]
conv_wvel_rmsf   = rlzAvg_wvel_rmsf[-1,:]
conv_Rxx         = rlzAvg_Rxx[-1,:]
conv_Ryy         = rlzAvg_Ryy[-1,:]
conv_Rzz         = rlzAvg_Rzz[-1,:]
conv_Rxy         = rlzAvg_Rxy[-1,:]
conv_Rxz         = rlzAvg_Rxz[-1,:]
conv_Ryz         = rlzAvg_Ryz[-1,:]

# enforce symmetry to averaged converged fields
nunif2 = int(nunif/2)
if nunif%2 != 0: # is odd
    nunifb = nunif2
    nunift = nunif2+1
else:
    nunifb = nunif2
    nunift = nunif2
symm_half_channel_conv_uvel_mean = 0.5 * (conv_uvel_mean[:nunifb] + np.flipud(conv_uvel_mean[nunift:]))
symm_half_channel_conv_uvel_rmsf = 0.5 * (conv_uvel_rmsf[:nunifb] + np.flipud(conv_uvel_rmsf[nunift:]))
symm_half_channel_conv_vvel_mean = 0.5 * (conv_vvel_mean[:nunifb] + np.flipud(conv_vvel_mean[nunift:]))
symm_half_channel_conv_vvel_rmsf = 0.5 * (conv_vvel_rmsf[:nunifb] + np.flipud(conv_vvel_rmsf[nunift:]))
symm_half_channel_conv_wvel_mean = 0.5 * (conv_wvel_mean[:nunifb] + np.flipud(conv_wvel_mean[nunift:]))
symm_half_channel_conv_wvel_rmsf = 0.5 * (conv_wvel_rmsf[:nunifb] + np.flipud(conv_wvel_rmsf[nunift:]))
symm_half_channel_conv_Rxx       = 0.5 * (conv_Rxx[:nunifb]       + np.flipud(conv_Rxx[nunift:]))
symm_half_channel_conv_Ryy       = 0.5 * (conv_Ryy[:nunifb]       + np.flipud(conv_Ryy[nunift:]))
symm_half_channel_conv_Rzz       = 0.5 * (conv_Rzz[:nunifb]       + np.flipud(conv_Rzz[nunift:]))
symm_half_channel_conv_Rxy       = 0.5 * (conv_Rxy[:nunifb]       + np.flipud(conv_Rxy[nunift:]))
symm_half_channel_conv_Rxz       = 0.5 * (conv_Rxz[:nunifb]       + np.flipud(conv_Rxz[nunift:]))
symm_half_channel_conv_Ryz       = 0.5 * (conv_Ryz[:nunifb]       + np.flipud(conv_Ryz[nunift:]))
conv_uvel_mean[:nunifb] = symm_half_channel_conv_uvel_mean;  conv_uvel_mean[nunift:] = np.flipud(symm_half_channel_conv_uvel_mean)
conv_uvel_rmsf[:nunifb] = symm_half_channel_conv_uvel_rmsf;  conv_uvel_rmsf[nunift:] = np.flipud(symm_half_channel_conv_uvel_rmsf)
conv_vvel_mean[:nunifb] = symm_half_channel_conv_vvel_mean;  conv_vvel_mean[nunift:] = np.flipud(symm_half_channel_conv_vvel_mean)
conv_vvel_rmsf[:nunifb] = symm_half_channel_conv_vvel_rmsf;  conv_vvel_rmsf[nunift:] = np.flipud(symm_half_channel_conv_vvel_rmsf)
conv_wvel_mean[:nunifb] = symm_half_channel_conv_wvel_mean;  conv_wvel_mean[nunift:] = np.flipud(symm_half_channel_conv_wvel_mean)
conv_wvel_rmsf[:nunifb] = symm_half_channel_conv_wvel_rmsf;  conv_wvel_rmsf[nunift:] = np.flipud(symm_half_channel_conv_wvel_rmsf)
conv_Rxx[:nunifb]       = symm_half_channel_conv_Rxx;        conv_Rxx[nunift:]       = np.flipud(symm_half_channel_conv_Rxx)
conv_Ryy[:nunifb]       = symm_half_channel_conv_Ryy;        conv_Ryy[nunift:]       = np.flipud(symm_half_channel_conv_Ryy)
conv_Rzz[:nunifb]       = symm_half_channel_conv_Rzz;        conv_Rzz[nunift:]       = np.flipud(symm_half_channel_conv_Rzz)
conv_Rxy[:nunifb]       = symm_half_channel_conv_Rxy;        conv_Rxy[nunift:]       = np.flipud(symm_half_channel_conv_Rxy)
conv_Rxz[:nunifb]       = symm_half_channel_conv_Rxz;        conv_Rxz[nunift:]       = np.flipud(symm_half_channel_conv_Rxz)
conv_Ryz[:nunifb]       = symm_half_channel_conv_Ryz;        conv_Ryz[nunift:]       = np.flipud(symm_half_channel_conv_Ryz)

# Calculate NRMSE (relative L2 error) of each rlz along time, compared with rlz-averaged field at tEnd
NRMSE_uvel_mean  = np.zeros([numRefRlz, ntk]); NRMSE_denum_uvel_mean = np.linalg.norm(conv_uvel_mean, 2)
NRMSE_uvel_rmsf  = np.zeros([numRefRlz, ntk]); NRMSE_denum_uvel_rmsf = np.linalg.norm(conv_uvel_rmsf, 2)
NRMSE_vvel_mean  = np.zeros([numRefRlz, ntk]); NRMSE_denum_vvel_mean = np.linalg.norm(conv_vvel_mean, 2)
NRMSE_vvel_rmsf  = np.zeros([numRefRlz, ntk]); NRMSE_denum_vvel_rmsf = np.linalg.norm(conv_vvel_rmsf, 2)
NRMSE_wvel_mean  = np.zeros([numRefRlz, ntk]); NRMSE_denum_wvel_mean = np.linalg.norm(conv_wvel_mean, 2)
NRMSE_wvel_rmsf  = np.zeros([numRefRlz, ntk]); NRMSE_denum_wvel_rmsf = np.linalg.norm(conv_wvel_rmsf, 2)
NRMSE_Rxx        = np.zeros([numRefRlz, ntk]); NRMSE_denum_Rxx       = np.linalg.norm(conv_Rxx, 2)
NRMSE_Ryy        = np.zeros([numRefRlz, ntk]); NRMSE_denum_Ryy       = np.linalg.norm(conv_Ryy, 2)
NRMSE_Rzz        = np.zeros([numRefRlz, ntk]); NRMSE_denum_Rzz       = np.linalg.norm(conv_Rzz, 2)
NRMSE_Rxy        = np.zeros([numRefRlz, ntk]); NRMSE_denum_Rxy       = np.linalg.norm(conv_Rxy, 2)
NRMSE_Rxz        = np.zeros([numRefRlz, ntk]); NRMSE_denum_Rxz       = np.linalg.norm(conv_Rxz, 2)
NRMSE_Ryz        = np.zeros([numRefRlz, ntk]); NRMSE_denum_Ryz       = np.linalg.norm(conv_Ryz, 2)
for irlz in range(numRefRlz):
    for itk in range(ntk):
        NRMSE_uvel_mean[irlz, itk] = np.linalg.norm(uvel_mean[irlz, itk, :] - conv_uvel_mean, 2) / NRMSE_denum_uvel_mean
        NRMSE_uvel_rmsf[irlz, itk] = np.linalg.norm(uvel_rmsf[irlz, itk, :] - conv_uvel_rmsf, 2) / NRMSE_denum_uvel_rmsf
        NRMSE_vvel_mean[irlz, itk] = np.linalg.norm(vvel_mean[irlz, itk, :] - conv_vvel_mean, 2) / NRMSE_denum_vvel_mean
        NRMSE_vvel_rmsf[irlz, itk] = np.linalg.norm(vvel_rmsf[irlz, itk, :] - conv_vvel_rmsf, 2) / NRMSE_denum_vvel_rmsf
        NRMSE_wvel_mean[irlz, itk] = np.linalg.norm(wvel_mean[irlz, itk, :] - conv_wvel_mean, 2) / NRMSE_denum_wvel_mean
        NRMSE_wvel_rmsf[irlz, itk] = np.linalg.norm(wvel_rmsf[irlz, itk, :] - conv_wvel_rmsf, 2) / NRMSE_denum_wvel_rmsf
        NRMSE_Rxx[irlz, itk]       = np.linalg.norm(Rxx[irlz, itk, :]       - conv_Rxx      , 2) / NRMSE_denum_Rxx
        NRMSE_Ryy[irlz, itk]       = np.linalg.norm(Ryy[irlz, itk, :]       - conv_Ryy      , 2) / NRMSE_denum_Ryy
        NRMSE_Rzz[irlz, itk]       = np.linalg.norm(Rzz[irlz, itk, :]       - conv_Rzz      , 2) / NRMSE_denum_Rzz
        NRMSE_Rxy[irlz, itk]       = np.linalg.norm(Rxy[irlz, itk, :]       - conv_Rxy      , 2) / NRMSE_denum_Rxy
        NRMSE_Rxz[irlz, itk]       = np.linalg.norm(Rxz[irlz, itk, :]       - conv_Rxz      , 2) / NRMSE_denum_Rxz
        NRMSE_Ryz[irlz, itk]       = np.linalg.norm(Ryz[irlz, itk, :]       - conv_Ryz      , 2) / NRMSE_denum_Ryz
rlzAvg_NRMSE_uvel_mean = np.mean(NRMSE_uvel_mean, axis=0)   # shape [ntk]
rlzAvg_NRMSE_uvel_rmsf = np.mean(NRMSE_uvel_rmsf, axis=0)
rlzAvg_NRMSE_vvel_mean = np.mean(NRMSE_vvel_mean, axis=0)
rlzAvg_NRMSE_vvel_rmsf = np.mean(NRMSE_vvel_rmsf, axis=0)
rlzAvg_NRMSE_wvel_mean = np.mean(NRMSE_wvel_mean, axis=0)
rlzAvg_NRMSE_wvel_rmsf = np.mean(NRMSE_wvel_rmsf, axis=0)
rlzAvg_NRMSE_Rxx       = np.mean(NRMSE_Rxx,       axis=0)
rlzAvg_NRMSE_Ryy       = np.mean(NRMSE_Ryy,       axis=0)
rlzAvg_NRMSE_Rzz       = np.mean(NRMSE_Rzz,       axis=0)
rlzAvg_NRMSE_Rxy       = np.mean(NRMSE_Rxy,       axis=0)
rlzAvg_NRMSE_Rxz       = np.mean(NRMSE_Rxz,       axis=0)
rlzAvg_NRMSE_Ryz       = np.mean(NRMSE_Ryz,       axis=0)

# ------------------------------------------------------------------------------------------------------
#                                           Save non-RL & baseline data
# ------------------------------------------------------------------------------------------------------

# averaged data directory
dir_path   = f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/data_rlz_avg/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
else:
    print(f"Directory '{dir_path}' already exists.")

# At t = tk, save realizations-averaged (non-converged) statistical quantities
zeros_arr = np.zeros(nunif) # for columns  4_uvel_rhsfRatio, 7_vvel_rhsfRatio, 10_wvel_rhsfRatio
for tk in range(ntk):
    fname = os.path.join(dir_path, f"stat_dmp_{tk:05d}.dat")
    odt_data  = np.vstack([posUnif, rlzAvg_uvel_mean[tk,:], rlzAvg_uvel_rmsf[tk,:], zeros_arr, rlzAvg_vvel_mean[tk,:], rlzAvg_vvel_rmsf[tk,:], zeros_arr, rlzAvg_wvel_mean[tk,:], rlzAvg_wvel_rmsf[tk,:], zeros_arr, rlzAvg_Rxx[tk,:], rlzAvg_Ryy[tk,:], rlzAvg_Rzz[tk,:], rlzAvg_Rxy[tk,:], rlzAvg_Rxz[tk,:], rlzAvg_Ryz[tk,:]]).T
    np.savetxt(fname, odt_data, 
               header="#         1_posUnif        2_uvel_mean        3_uvel_rmsf   4_uvel_rhsfRatio        5_vvel_mean        6_vvel_rmsf   7_vvel_rhsfRatio        8_wvel_mean        9_wvel_rmsf  10_wvel_rhsfRatio             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz", 
               comments=f"# time = {timeUnif[tk]}\n# FINE UNIFORM Grid points = {nunif}\n# Pressure (Pa) = 101325\n",
               fmt="%18.10E")

# At tEnd, save converged baseline data for umean & urmsf
fname = f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/statistics_reference_udata.dat"
odt_data = np.vstack([posUnif, conv_uvel_mean, conv_uvel_rmsf]).T
np.savetxt(fname, odt_data, 
           header="#         1_posUnif        2_uvel_mean        3_uvel_rmsf",
           comments=f"# time = {timeUnif[-1]}\n# FINE UNIFORM Grid points = {nunif}\n# Pressure (Pa) = 101325\n",
           fmt='%18.10E')

# At tEnd, save converged baseline data for all converged statistics quantities
fname = f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/statistics_reference.dat"
zeros_arr = np.zeros(nunif) # for columns  4_uvel_rhsfRatio, 7_vvel_rhsfRatio, 10_wvel_rhsfRatio
odt_data  = np.vstack([posUnif, conv_uvel_mean, conv_uvel_rmsf, zeros_arr, conv_vvel_mean, conv_vvel_rmsf, zeros_arr, conv_wvel_mean, conv_wvel_rmsf, zeros_arr, conv_Rxx, conv_Ryy, conv_Rzz, conv_Rxy, conv_Rxz, conv_Ryz]).T
np.savetxt(fname, odt_data, 
           header="#         1_posUnif        2_uvel_mean        3_uvel_rmsf   4_uvel_rhsfRatio        5_vvel_mean        6_vvel_rmsf   7_vvel_rhsfRatio        8_wvel_mean        9_wvel_rmsf  10_wvel_rhsfRatio             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz",
           comments=f"# time = {timeUnif[-1]}\n# FINE UNIFORM Grid points = {nunif}\n# Pressure (Pa) = 101325\n",
           fmt="%18.10E")

# ------------------------------------------------------------------------------------------------------
#                                           Plot non-RL errors vs. baseline
# ------------------------------------------------------------------------------------------------------

timeUnifAvg = timeUnif - tBeginAvg

# Plot umean NRMSE along time for each rlz (and averaged along realizations) 
# Note: neglect first time instant ([1:]) as it corresponds to the statistics initialization time, when statistics data is initialized as 0 everywhere (which leads to NRMSE=1)
plt.figure()
for irlz in range(numRefRlz):
    plt.semilogy(timeUnifAvg[1:], NRMSE_uvel_mean[irlz,1:], alpha=0.5, label=f"Rlz {irlz}")
plt.semilogy(timeUnifAvg[1:], rlzAvg_NRMSE_uvel_mean[1:], 'k', label=f"Rlz Avg")
plt.xlabel(r"$t^{+}$")
plt.ylabel(r"NRMSE $\overline{u}^{+}$")
plt.ylim([0.001, 0.1])
plt.legend()
plt.tight_layout()
plt.savefig(f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/NRMSE_u_mean.jpg", dpi=600)

# Plot urmsf NRMSE along time for each rlz (and averaged along realizations) 
plt.figure()
for irlz in range(numRefRlz):
    plt.semilogy(timeUnifAvg[1:], NRMSE_uvel_rmsf[irlz,1:], alpha=0.5, label=f"Rlz {irlz}")
plt.semilogy(timeUnifAvg[1:], rlzAvg_NRMSE_uvel_rmsf[1:], 'k', label=f"Rlz Avg")
plt.xlabel(r"$t^{+}$")
plt.ylabel(r"NRMSE $u^{+}_{\textrm{rms}}$")
plt.ylim([0.005, 0.5])
plt.legend()
plt.tight_layout()
plt.savefig(f"{odt_path}/post/channelFlow/ODT_reference/Re{Retau}/NRMSE_u_rmsf.jpg", dpi=600)



