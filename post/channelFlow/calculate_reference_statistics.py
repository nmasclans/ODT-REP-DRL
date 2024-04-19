"""
Assumption: 
Statistics stat_dmp_xxxxx.dat files have data columns as:
#         1_posUnif        2_uvel_mean        3_uvel_rmsf   4_uvel_rhsfRatio        5_vvel_mean        6_vvel_rmsf   7_vvel_rhsfRatio        8_wvel_mean        9_wvel_rmsf  10_wvel_rhsfRatio             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz
"""


import os
import tqdm

import glob as gb
import numpy as np
import matplotlib.pyplot as plt

from utils import get_time

Retau     = 180
numRefRlz = 10
nunif     = 361
ntk       = 901     # num stat_dmp_xxxxx.dat files per realization, 
                    # i.e. number of time instances at which statistics data is stored

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

#for irlz in tqdm(range(numRefRlz), desc="Getting statistics data from dmp files"):
for irlz in range(numRefRlz):
    print(f"irlz = {irlz}")
    
    # get all stat_dmp_*.dat files of data_xxxxx directory of rlz xxxxx 
    dir    = f"./ODT_reference/Re{Retau}/data_{irlz:05d}/"
    flist  = sorted(gb.glob(os.path.join(dir,"stat_dmp_*.dat")))
    nfiles = len(flist)
    assert ntk == nfiles

    # get data:
    for itk in range(ntk):
        ifile = flist[itk]
        data      = np.loadtxt(ifile)
        timeUnif[irlz, itk]     = get_time(ifile)
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

# save reference data for umean & urmsf
odt_data = np.vstack([posUnif, conv_uvel_mean, conv_uvel_rmsf]).T
fname = f"./ODT_reference/Re{Retau}/statistics_reference.dat"
np.savetxt(fname, odt_data, header="#         1_posUnif        2_uvel_mean        3_uvel_rmsf", fmt='%12.5E')

# Plot umean NRMSE along time for each rlz (and averaged along realizations) 
plt.figure()
for irlz in range(numRefRlz):
    plt.semilogy(timeUnif, NRMSE_uvel_mean[irlz,:], alpha=0.5, label=f"Rlz {irlz}")
plt.semilogy(timeUnif, rlzAvg_NRMSE_uvel_mean[:], 'k', label=f"Rlz Avg")
plt.xlabel("time [s]")
plt.ylabel(r"NRMSE $<u>$")
plt.legend()
plt.savefig(f"./ODT_reference/Re{Retau}/NRMSE_u_mean.jpg", dpi=600)

# Plot urmsf NRMSE along time for each rlz (and averaged along realizations) 
plt.figure()
for irlz in range(numRefRlz):
    plt.semilogy(timeUnif, NRMSE_uvel_rmsf[irlz,:], alpha=0.5, label=f"Rlz {irlz}")
plt.semilogy(timeUnif, rlzAvg_NRMSE_uvel_rmsf[:], 'k', label=f"Rlz Avg")
plt.xlabel("time [s]")
plt.ylabel(r"NRMSE $u'$")
plt.legend()
plt.savefig(f"./ODT_reference/Re{Retau}/NRMSE_u_rmsf.jpg", dpi=600)



