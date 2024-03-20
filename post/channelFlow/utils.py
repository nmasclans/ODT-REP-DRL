"""
Utils functions for post/channelFlow post-processing scripts
"""

import os
import yaml
import math
import glob as gb
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def get_nunif2_walls(nunif, nunif2):
    # -> the unse of nunif2 to mirror channel data leads to operands broadcast ERRORS if nunif is ODD 
    # -> bottom & top walls what different shape by 1 element if nunif2 is used as [:nunif] for bottom, and [nunif:] for top walls indexes
    # To solve this error, we define different nunifb & nunift if nunif is odd:
    if nunif%2 != 0: # is odd
        nunifb = nunif2
        nunift = nunif2+1
    else:
        nunifb = nunif2
        nunift = nunif2
    return nunifb, nunift


def get_odt_instantaneous(input_params):
    """
    Get ODT instantaneous fields

    Parameters:
        input_params (dict): ODT input parameters dictionary

    Returns:
        ODT instantaneous coordinates and fields
        ydelta, yplus, time, uvel, vvel, wvel (np.ndarrays)
    """
    # --- Get ODT input parameters ---

    caseName = input_params["caseN"]
    rlzStr   = input_params["rlzStr"]
    delta    = input_params["delta"]
    kvisc    = input_params["kvisc"]
    nunif    = input_params["nunif"]
    utau     = input_params["utau"]

    # half of the channel
    nunif2   = int(nunif/2)                             # half of num. points (for ploting to domain center, symmetry in y-axis)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)    # added for compatibility with nunif odd

    # --- Get ODT data ---

    flist     = sorted(gb.glob(f'../../data/{caseName}/data/data_{rlzStr}/dmp_*.dat'))
    num_files = len(flist)
    
    yu   = np.linspace(-delta, delta, nunif)    # shape [nunif,] uniform grid in y-axis
    time = np.zeros(num_files)                  # shape [num_files,]
    uvel = np.zeros([nunif, num_files])         # shape [nunif, num_files]
    vvel = np.zeros([nunif, num_files])
    wvel = np.zeros([nunif, num_files])

    for i in range(num_files):

        ifile = flist[i]
        data = np.loadtxt(ifile)
        y    = data[:,0] # = y/delta, as delta = 1
        u    = data[:,2] # normalized by u_tau, u is in fact u+
        v    = data[:,3] # normalized by u_tau, v is in fact v+
        w    = data[:,4] # normalized by u_tau, w is in fact w+ 

        # interpolate to uniform grid
        uu = interp1d(y, u, fill_value='extrapolate')(yu)  
        vv = interp1d(y, v, fill_value='extrapolate')(yu)
        ww = interp1d(y, w, fill_value='extrapolate')(yu)

        # store time instant data
        uvel[:,i] = uu
        vvel[:,i] = vv
        wvel[:,i] = ww

        # get and store time
        time[i] = get_time(ifile)

    # mirror data -> half channel
    uvel = 0.5*(uvel[:nunifb,:] + np.flipud(uvel[nunift:,:]))
    vvel = 0.5*(vvel[:nunifb,:] + np.flipud(vvel[nunift:,:]))
    wvel = 0.5*(wvel[:nunifb,:] + np.flipud(wvel[nunift:,:]))

    # coordinates in wall units y -> y+
    yplus  = yu * utau / kvisc
    ydelta = yu / delta

    return (ydelta, yplus, time, uvel, vvel, wvel)


def get_odt_instantaneous_in_adaptative_grid(input_params):
    """
    Get ODT instantaneous fields in adaptative grid

    Parameters:
        input_params (dict): ODT input parameters dictionary

    Returns:
        ODT instantaneous coordinates and fields in adaptative grid
        y, yf, dy, time, uvel, vvel, wvel (np.ndarrays)
    """
    # --- Get ODT input parameters ---

    caseName = input_params["caseN"]
    rlzStr   = input_params["rlzStr"]

    # --- Get ODT data ---

    flist     = sorted(gb.glob(f'../../data/{caseName}/data/data_{rlzStr}/dmp_*.dat'))
    num_files = len(flist)
    
    # Attention: we take data from adaptative grid: y,dy,u,v,w change size for each instantaneous data file
    time = []       # will have shape [num_files,]
    y    = []       # list of 1-d np.arrays of different length, as grid is adaptative
    yf   = []       # idem. , but each np.array will be of +1 element than the other quantities
    dy   = []       # idem.
    uvel = []       # idem.
    vvel = []       # idem.
    wvel = []       # idem.

    for i in range(num_files):

        ifile = flist[i]
        data = np.loadtxt(ifile)
        y.append(data[:,0])    # = y/delta, as delta = 1
        uvel.append(data[:,2]) # normalized by u_tau, u is in fact u+
        vvel.append(data[:,3]) # normalized by u_tau, v is in fact v+
        wvel.append(data[:,4]) # normalized by u_tau, w is in fact w+ 

        # frontier positions of each cell, of length +1 than y,u,v,w
        yf_ = data[:,1]     # missing last point of yf = 1, from how ODT stores data; dy is bigger by 1 element than y,u,v,w, but last element 1 is never stored because it is constant along simulation
        yf_ = np.append(yf_, 1)
        yf.append(yf_)

        # dy of each cell, of length == than y,u,v,w
        dy_ = yf_[1:] - yf_[:-1]
        dy.append(dy_)

        # get and store time
        time.append(get_time(ifile))

    return (y, yf, dy, time, uvel, vvel, wvel)


def compute_odt_statistics_post(odt_statistics_filepath, input_params, plot_reynolds_stress_terms=False):
    """
    Compute ODT statistics from multiple .dat files with instantaneous data
    at increasing simulation time

    Parameters:
        odt_statistics_filepath (str): filepath where the computed ODT statistics will be saved
        input_params (dict): ODT input parameters dictionary
        
    Comments: 
        The ODT statistics are saved in a .dat files with the following columns data (included as header in the file): 
        y/delta, y+, u+_mean, v+_mean, w+_mean, u+_rmsf, v+_rmsf, w+_rmsf, <u'u'>+, <v'v'>+, <w'w'>+, <u'v'>+, <u'w'>+, <v'w'>+, tau_viscous, tau_reynolds, tau_total, vt_u+, d_u+
    """

    # --- Get ODT input parameters ---
    rho        = input_params["rho"]
    utau       = input_params["utau"]
    kvisc      = input_params["kvisc"] # = nu = mu / rho 
    dxmin      = input_params["dxmin"]
    delta      = input_params["delta"]
    Retau      = input_params["Retau"]
    nunif      = input_params["nunif"]
    case_name  = input_params["caseN"]
    rlzStr     = input_params["rlzStr"]
    tBeginAvg  = input_params["tBeginAvg"]
    tEndAvg    = input_params["tEndAvg"]
    
    # un-normalize
    domainLength = input_params["domainLength"]
    dxmin *= domainLength

    # --- Compute ODT computational data ---

    flist = sorted(gb.glob(f'../../data/{case_name}/data/data_{rlzStr}/dmp_*.dat'))
    flist_stat = sorted(gb.glob(f'../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_*.dat'))

    nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)

    # initialize arrays
    yu  = np.linspace(-delta,delta,nunif) # uniform grid in y-axis
    # empty vectors of time-averaged quantities
    um  = np.zeros(nunif)        # mean velocity, calulated in post-processing from instantaneous velocity
    vm  = np.zeros(nunif)
    wm  = np.zeros(nunif)
    u2m = np.zeros(nunif)        # mean square velocity (for rmsf and reynolds stresses)
    v2m = np.zeros(nunif)
    w2m = np.zeros(nunif)
    uvm = np.zeros(nunif)        # mean velocity correlations (for reynolds stresses)
    uwm = np.zeros(nunif)
    vwm = np.zeros(nunif)
    dudy2m = np.zeros(nunif2-2)

    logging_files_period = 1000
    file_counter = 0
    ifile_total   = len(flist)
    for ifile in flist :

        # Check file tBeginAvg <= currentTime <= tEndAvg
        currentTime = get_time(ifile)
        if tBeginAvg <= currentTime and currentTime <= tEndAvg: 
                
            # ------------------ (get) Instantaneous velocity ------------------

            data = np.loadtxt(ifile)
            y    = data[:,0] # = y/delta, as delta = 1
            u    = data[:,2] # normalized by u_tau, u is in fact u+
            v    = data[:,3] # normalized by u_tau, v is in fact v+
            w    = data[:,4] # normalized by u_tau, w is in fact w+ 

            # interpolate to uniform grid
            uu = interp1d(y, u, fill_value='extrapolate')(yu)  
            vv = interp1d(y, v, fill_value='extrapolate')(yu)
            ww = interp1d(y, w, fill_value='extrapolate')(yu)

            # ------------------ (compute) Velocity statistics, from instantaneous values ------------------

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
            
            # update mean profile dudy2m = avg((du/dy)**2)
            # top half of channel
            uut    = uu[nunift:]
            yut    = yu[nunift:]
            dudy2t = ( (uut[2:]-uut[:-2])/(yut[2:]-yut[:-2]) )**2
            # bottom half of channel
            uub     = np.flip(uu[:nunifb])
            yub     = - np.flip(yu[:nunifb])
            dudy2b  = ( (uub[2:]-uub[:-2])/(yub[2:]-yub[:-2]) )**2
            dudy2m += 0.5*(dudy2b + dudy2t)  # mirror data (symmetric)

            # Logging info
            file_counter += 1
            if file_counter % logging_files_period == 1:
                print(f"Calculating ODT statistics... {file_counter/ifile_total*100:.0f}%")

    # (computed) means
    um /= file_counter
    vm /= file_counter
    wm /= file_counter
    um = 0.5*(um[:nunifb] + np.flipud(um[nunift:]))  # mirror data (symmetric)
    vm = 0.5*(vm[:nunifb] + np.flipud(vm[nunift:]))
    wm = 0.5*(wm[:nunifb] + np.flipud(wm[nunift:]))

    # squared means
    u2m /= file_counter
    v2m /= file_counter
    w2m /= file_counter
    u2m = 0.5*(u2m[:nunifb] + np.flipud(u2m[nunift:]))
    v2m = 0.5*(v2m[:nunifb] + np.flipud(v2m[nunift:]))
    w2m = 0.5*(w2m[:nunifb] + np.flipud(w2m[nunift:]))

    # velocity correlations
    uvm /= file_counter
    uwm /= file_counter
    vwm /= file_counter
    uvm = 0.5*(uvm[:nunifb] + np.flipud(uvm[nunift:]))
    uwm = 0.5*(uwm[:nunifb] + np.flipud(uwm[nunift:]))
    vwm = 0.5*(vwm[:nunifb] + np.flipud(vwm[nunift:]))

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

    # averaged terms for calculating TKE budgets
    dudy2m /= file_counter 

    # --- y-coordinate, y+ ---
    yu += delta         # domain center is at 0; shift so left side is zero
    yu = yu[:nunif2]    # plotting to domain center
    ydelta = yu/delta
    yplus  = yu * utau / kvisc
    
    dudyOdt = (um[1]-um[0])/(yu[1]-yu[0])
    utauOdt = np.sqrt(kvisc * np.abs(dudyOdt) / rho)
    RetauOdt = utau * delta / kvisc
    print(f"\n[compute_odt_statistics] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauOdt}")
    print(f"[compute_odt_statistics] Expected u_tau = {utau} vs. Effective u_tau = {utauOdt}")

    # --- Compute Stress Decomposition ---
    # Stress decomposition: Viscous, Reynolds and Total stress
    dumdy = (um[1:] - um[:-1])/(yu[1:] - yu[:-1])
    viscous_stress_  = kvisc * rho * dumdy
    reynolds_stress_ = - rho * ufvfm[:-1]
    total_stress_    = viscous_stress_ + reynolds_stress_

    # --- Viscous Transport budget ---
    f_uk  = urmsf**2 - um**2 
    ###f_vk  = vrmsf**2 - vm**2 
    ###f_wk  = wrmsf**2 - wm**2 
    Deltay_k = (yu[2:]-yu[:-2])/2 # length in y-coordinate of cell k from ODT grid 
    vt_u_ = kvisc * ( (f_uk[2:] - 2*f_uk[1:-1] + f_uk[:-2])/(Deltay_k**2) )
    ###vt_v_ = kvisc * ( (f_vk[2:] - 2*f_vk[1:-1] + f_vk[:-2])/(Deltay_k**2) )
    ###vt_w_ = kvisc * ( (f_wk[2:] - 2*f_wk[1:-1] + f_wk[:-2])/(Deltay_k**2) )
    vt_u_plus_ = vt_u_ * ( delta / utau**3 ) # not necessary, as um, urmsf, y is already um+, urmsf+, y+, and utau = 1, delta = 1
    ###vt_v_plus_ = vt_v_ * ( delta / utau**3 )
    ###vt_w_plus_ = vt_w_ * ( delta / utau**3 )

    # --- Dissipation budget ---
    dumdy = (um[2:] - um[:-2])/(yu[2:] - yu[:-2]) # 1st-order central finite difference
    d_u_ = 2 * kvisc * (dudy2m - dumdy**2)
    d_u_plus_ = d_u_ * delta / utau

    # Add "0" at grid boundaries to quantities computed from 1st- and 2nd-order derivatives to vstack vectors of the same size
    # -> involve 1st-order forward finite differences
    viscous_stress  = np.zeros(nunif2); viscous_stress[:-1]  = viscous_stress_  
    reynolds_stress = np.zeros(nunif2); reynolds_stress[:-1] = reynolds_stress_  
    total_stress    = np.zeros(nunif2); total_stress[:-1]    = total_stress_  
    # -> involve 2nd-order centered finite differences
    vt_u_plus = np.zeros(nunif2); vt_u_plus[1:-1] = vt_u_plus_
    d_u_plus  = np.zeros(nunif2); d_u_plus[1:-1]  = d_u_plus_

    # --- Save ODT computational data ---

    odt_data = np.vstack([ydelta,yplus, 
                          um,vm,wm,urmsf,vrmsf,wrmsf,
                          ufufm,vfvfm,wfwfm,ufvfm,ufwfm,vfwfm,
                          viscous_stress,reynolds_stress,total_stress, # -> stress decomposition
                          vt_u_plus, d_u_plus]).T                      # -> TKE budgets for u-component
    np.savetxt(odt_statistics_filepath, odt_data, 
            header= "y/delta,    y+,          u+_mean,     v+_mean,     w+_mean,     u+_rmsf,     v+_rmsf,     w+_rmsf,     "\
                    "<u'u'>+,     <v'v'>+,     <w'w'>+,     <u'v'>+,     <u'w'>+,     <v'w'>+,     " \
                    "tau_viscous, tau_reynolds,tau_totafilel,   " \
                    "vt_u+,       d_u+,        ",
            fmt='%12.5E')

    # ---------- non-diagonal terms of reynolds stress tensor --------
    if plot_reynolds_stress_terms:
        fs=20
        fs_leg=20
        fig, ax = plt.subplots(3, figsize=(10,10))
        ax[0].plot(yplus, ufvfm,'o',label=r"$R_{01}=<u'v'>$")
        ax[0].plot(yplus, ufwfm,    label=r"$R_{02}=<u'w'>$")
        ax[0].plot(yplus, vfwfm,    label=r"$R_{12}=<v'w'>$")
        ax[0].legend(loc="upper right", fontsize=fs_leg)
        ax[0].set_title(r"$R_{ij}=<u_i'u_j'> = <u_iu_j> - <u_i><u_j>$ vs. $y^{+}$", fontsize=fs)
        ##
        ax[1].plot(yplus, uvm,'o',label=r"$<uv>$")
        ax[1].plot(yplus, uwm,    label=r"$<uw>$")
        ax[1].plot(yplus, vwm,    label=r"$<vw>$")
        ax[1].legend(loc="upper right", fontsize=fs_leg)
        ax[1].set_title(r"$<u_iu_j>$ vs. $y^{+}$", fontsize=fs)
        ##
        ax[2].plot(yplus, um*vm,'o',label=r"$<u><v>$")
        ax[2].plot(yplus, um*wm,    label=r"$<u><w>$")
        ax[2].plot(yplus, vm*wm,    label=r"$<v><w>$")
        ax[2].legend(loc="upper right", fontsize=fs_leg)
        ax[2].set_title(r"$<u_i><u_j>$ vs. $y^{+}$", fontsize=fs)
        ##
        plt.tight_layout()
        plt.savefig('reynStressTens_nonDiag_odt.jpg', dpi=600)
        plt.close()
        # ---------- non-diagonal terms of reynolds stress tensor --------
        fig, ax = plt.subplots(3, figsize=(10,10))
        ax[0].plot(yplus, ufufm,    label=r"$R_{00}=<u'u'>$")
        ax[0].plot(yplus, vfvfm,'o',label=r"$R_{11}=<v'v'>$")
        ax[0].plot(yplus, wfwfm,    label=r"$R_{22}=<w'w'>$")
        ax[0].legend(loc="upper right", fontsize=fs_leg)
        ax[0].set_title(r"$R_{ii}=<u_i'u_i'> = <u_iu_i> - <u_i><u_i>$ vs. $y^{+}$", fontsize=fs)
        ##
        ax[1].plot(yplus, u2m,    label=r"$<uu>$")
        ax[1].plot(yplus, v2m,'o',label=r"$<vv>$")
        ax[1].plot(yplus, w2m,    label=r"$<ww>$")
        ax[1].legend(loc="upper right", fontsize=fs_leg)
        ax[1].set_title(r"$<u_iu_i>$ vs. $y^{+}$", fontsize=fs)
        ##
        ax[2].plot(yplus, um*um,    label=r"$<u><u>$")
        ax[2].plot(yplus, vm*vm,'o',label=r"$<v><v>$")
        ax[2].plot(yplus, wm*wm,    label=r"$<w><w>$")
        ax[2].legend(loc="upper right", fontsize=fs_leg)
        ax[2].set_title(r"$<u_i><u_i>$ vs. $y^{+}$", fontsize=fs)
        ##
        plt.tight_layout()
        plt.savefig("reynStressTens_diag_odt.jpg", dpi=600)
        plt.close()
        # ---------- plot all terms of reynolds stress tensor --------
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(yplus, ufufm,    linewidth=2, zorder = 3,                label=r"$R_{00}=<u'u'>$")
        ax.plot(yplus, vfvfm,'o',linewidth=2, zorder = 2, markersize=5,  label=r"$R_{11}=<v'v'>$")
        ax.plot(yplus, wfwfm,    linewidth=2, zorder = 3,                label=r"$R_{22}=<w'w'>$")
        ax.plot(yplus, ufvfm,'o',linewidth=2, zorder = 2, markersize=5,  label=r"$R_{01}=<u'v'>$")
        ax.plot(yplus, ufwfm,    linewidth=2, zorder = 3,                label=r"$R_{02}=<u'w'>$")
        ax.plot(yplus, vfwfm,'o',linewidth=2, zorder = 1, markersize=10, label=r"$R_{12}=<v'w'>$")
        ax.set_xlabel(r"$y^{+}$")
        ax.legend(loc="upper right", fontsize=fs_leg)
        ax.set_title(r"$R_{ij}=<u_i'u_i'>$, for $i,j=0,1,2$", fontsize=fs)
        plt.tight_layout()
        plt.savefig("reynStressTens_allTerms_odt.jpg", dpi=600)
        plt.close()


def get_odt_statistics_post(odt_statistics_filepath):
    """
    Get ODT statistics, previously saved in a .dat file using 'compute_odt_statistics' function

    Parameters:
        odt_statistics_filepath (str): ODT statistics filepath
        input_params (dict): ODT input parameters dictionary

    Returns:
        ODT statistics calculated over statistic time by function 'compute_odt_statistics', 
        saved by 'compute_odt_statistics' in odt_statistics_filepath .dat file with columns:
        y/delta, y+,    u+_mean, v+_mean, w+_mean, u+_rmsf, v+_rmsf, w+_rmsf, <u'u'>+, <v'v'>+, <w'w'>+, <u'v'>+, <u'w'>+, <v'w'>+, tau_viscous,    tau_reynolds,    tau_total,     vt_u+, d_u+
        (named here for simplicity, as u_tau = 1, delta = 1):
        ydelta,  yplus, um,      vm,      wm,      urmsf,   vrmsf,   wrmsf,   ufufm,   vfvfm,   wfwfm,   ufvfm,   ufwfm,   vfwfm,   viscous_stress, reynolds_stress, total_stress,  vt_u+, d_u+ (np.ndarrays)
    """
    # --- Get ODT statistics ---

    print(f"\nGetting ODT data from {odt_statistics_filepath}")
    odt = np.loadtxt(odt_statistics_filepath)

    ydelta = odt[:,0]  # y/delta
    yplus  = odt[:,1]  # y+

    um     = odt[:,2]  # u+_mean
    vm     = odt[:,3]  # v+_mean
    wm     = odt[:,4]  # w+_mean

    urmsf  = odt[:,5]  # u+_rmsf
    vrmsf  = odt[:,6]  # v+_rmsf
    wrmsf  = odt[:,7]  # w+_rmsf

    ufufm  = odt[:,8]  # R_xx+
    vfvfm  = odt[:,9]  # R_yy+
    wfwfm  = odt[:,10] # R_zz+
    ufvfm  = odt[:,11] # R_xy+
    ufwfm  = odt[:,12] # R_xz+
    vfwfm  = odt[:,13] # R_yz+

    viscous_stress  = odt[:,14]
    reynolds_stress = odt[:,15]
    total_stress    = odt[:,16]

    # TKE budgets
    vt_u = odt[:,17]  # Viscous transport 
    d_u  = odt[:,18]  # Dissipation

    return (ydelta, yplus, um, vm, wm, urmsf, vrmsf, wrmsf, 
            ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm, 
            viscous_stress, reynolds_stress, total_stress, 
            vt_u, d_u)


def get_odt_statistics_rt(input_params):
    """
    Get ODT statistics calculated during runtime, stored in data/case_name/data/data_xxxxx/statistics/

    Parameters:
        input_params (dict): ODT input parameters dictionary

    Returns:
        ODT statistics calculated during runtime
        ( ydelta, 
          um_data, urmsf_data, uFpert_data, 
          vm_data, vrmsf_data, vFpert_data,
          wm_data, wrmsf_data, wFpert_data, 
          # TODO: update return information as returned variables changed
    """

    # --- Get ODT input parameters ---
    
    utau       = input_params["utau"]
    Retau      = input_params["Retau"]
    delta      = input_params["delta"]
    rho        = input_params["rho"]
    kvisc      = input_params["kvisc"]
    case_name  = input_params["caseN"]
    rlzStr     = input_params["rlzStr"]
    dTimeStart = input_params["dTimeStart"]
    dTimeEnd   = input_params["dTimeEnd"]
    dTimeStep  = input_params["dTimeStep"]
    tEndAvg    = input_params["tEndAvg"]
    
    # --- Get dmp file with dump number ***** corresponding to tEndAvg, or the closest one from below ---
    dTimes     = np.round(np.arange(dTimeStart, dTimeEnd+1e-6, dTimeStep), 6)
    tEndAvgDmpIdx = np.sum(tEndAvg > dTimes) 
    tEndAvgDmpStr = f"{tEndAvgDmpIdx:05d}"

    # --- Get vel. statistics computed during runtime at last time increment ---

    fstat     = f"../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_{tEndAvgDmpStr}.dat"
    data_stat = np.loadtxt(fstat)

    # -> check the file rows correspond to the expected variables:
    with open(fstat,'r') as f:
        rows_info = f.readlines()[3].split() # 4th line of the file
    rows_info_expected = '#         1_posUnif        2_uvel_mean        3_uvel_rmsf       4_uvel_Fpert        5_vvel_mean        6_vvel_rmsf       7_vvel_Fpert        8_wvel_mean        9_wvel_rmsf      10_wvel_Fpert             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz\n'.split()
    assert rows_info == rows_info_expected, f"statistic files rows do not correspond to the expected variables" \
        f"rows variables (expected): \n{rows_info_expected} \n" \
        f"rows variables (current): \n{rows_info}"
    
    # -> get data
    yu           = data_stat[:,0]
    #
    um_data      = data_stat[:,1] 
    urmsf_data   = data_stat[:,2] 
    uFpert_data  = data_stat[:,3]
    #
    vm_data      = data_stat[:,4] 
    vrmsf_data   = data_stat[:,5] 
    vFpert_data  = data_stat[:,6] 
    #
    wm_data      = data_stat[:,7] 
    wrmsf_data   = data_stat[:,8] 
    wFpert_data  = data_stat[:,9]  
    # reynolds stress terms
    ufufm_data   = data_stat[:,10]  # column header: 11_Rxx
    vfvfm_data   = data_stat[:,11]  # 12_Ryy
    wfwfm_data   = data_stat[:,12]  # 13_Rzz
    ufvfm_data   = data_stat[:,13]  # 14_Rxy
    ufwfm_data   = data_stat[:,14]  # 15_Rxz
    vfwfm_data   = data_stat[:,15]  # 16_Ryz
    # anisotropy tensor eigenvalues & barycentric map
    # lambda0_data = data_stat[:,16] 
    # lambda1_data = data_stat[:,17] 
    # lambda2_data = data_stat[:,18]
    # xmap1_data   = data_stat[:,19] 
    # xmap2_data   = data_stat[:,20] 

    ### Mirror data (symmetric in the y-direction from the channel center)
    # mirror data indexs
    nunif        = len(um_data)
    nunif2       = int(nunif/2)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)
    # mirror data
    um_data      = 0.5 * ( um_data[:nunifb]      + np.flipud(um_data[nunift:])      )
    urmsf_data   = 0.5 * ( urmsf_data[:nunifb]   + np.flipud(urmsf_data[nunift:])   )
    uFpert_data  = 0.5 * ( uFpert_data[:nunifb]  + np.flipud(uFpert_data[nunift:])  )
    vm_data      = 0.5 * ( vm_data[:nunifb]      + np.flipud(vm_data[nunift:])      )
    vrmsf_data   = 0.5 * ( vrmsf_data[:nunifb]   + np.flipud(vrmsf_data[nunift:])   )
    vFpert_data  = 0.5 * ( vFpert_data[:nunifb]  + np.flipud(vFpert_data[nunift:])  )
    wm_data      = 0.5 * ( wm_data[:nunifb]      + np.flipud(wm_data[nunift:])      )
    wrmsf_data   = 0.5 * ( wrmsf_data[:nunifb]   + np.flipud(wrmsf_data[nunift:])   )
    wFpert_data  = 0.5 * ( wFpert_data[:nunifb]  + np.flipud(wFpert_data[nunift:])  )
    ufufm_data   = 0.5 * ( ufufm_data[:nunifb]   + np.flipud(ufufm_data[nunift:])   )
    vfvfm_data   = 0.5 * ( vfvfm_data[:nunifb]   + np.flipud(vfvfm_data[nunift:])   )
    wfwfm_data   = 0.5 * ( wfwfm_data[:nunifb]   + np.flipud(wfwfm_data[nunift:])   )
    ufvfm_data   = 0.5 * ( ufvfm_data[:nunifb]   + np.flipud(ufvfm_data[nunift:])   )
    ufwfm_data   = 0.5 * ( ufwfm_data[:nunifb]   + np.flipud(ufwfm_data[nunift:])   )
    vfwfm_data   = 0.5 * ( vfwfm_data[:nunifb]   + np.flipud(vfwfm_data[nunift:])   )
    # lambda0_data = 0.5 * ( lambda0_data[:nunifb] + np.flipud(lambda0_data[nunift:]) )
    # lambda1_data = 0.5 * ( lambda1_data[:nunifb] + np.flipud(lambda1_data[nunift:]) )
    # lambda2_data = 0.5 * ( lambda2_data[:nunifb] + np.flipud(lambda2_data[nunift:]) )
    # xmap1_data   = 0.5 * ( xmap1_data[:nunifb]   + np.flipud(xmap1_data[nunift:])   )
    # xmap2_data   = 0.5 * ( xmap2_data[:nunifb]   + np.flipud(xmap2_data[nunift:])   )

    # ------------ scale y to y+ ------------

    # y-coordinates
    yu     = yu[:nunifb] + delta
    ydelta = yu/delta
    yplus  = yu * utau / kvisc

    # Check: Re_tau and u_tau of ODT data
    dumdy0     = (um_data[1]-um_data[0])/(yu[1]-yu[0])
    utauOdt  = np.sqrt(kvisc * np.abs(dumdy0))
    RetauOdt = utauOdt * delta / kvisc
    print(f"\n[get_odt_statistics_rt] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauOdt}")
    print(f"[get_odt_statistics_rt] Expected u_tau = {utau} vs. Effective u_tau = {utauOdt}")

    # ------------ Compute Stress Decomposition ------------
    
    # Stress decomposition: Viscous, Reynolds and Total stress
    dumdy = (um_data[1:] - um_data[:-1])/(yu[1:] - yu[:-1])
    viscous_stress_  = kvisc * rho * dumdy
    reynolds_stress_ = - rho * ufvfm_data[:-1]
    total_stress_    = viscous_stress_ + reynolds_stress_

    # Add "0" at grid boundaries to quantities computed from 1st- and 2nd-order derivatives to vstack vectors of the same size
    # -> involve 1st-order forward finite differences
    viscous_stress  = np.zeros(nunifb); viscous_stress[:-1]  = viscous_stress_  
    reynolds_stress = np.zeros(nunifb); reynolds_stress[:-1] = reynolds_stress_  
    total_stress    = np.zeros(nunifb); total_stress[:-1]    = total_stress_  

    return (ydelta, yplus, 
            um_data, urmsf_data, uFpert_data,
            vm_data, vrmsf_data, vFpert_data,
            wm_data, wrmsf_data, wFpert_data,
            ufufm_data, vfvfm_data, wfwfm_data, ufvfm_data, ufwfm_data, vfwfm_data,
            viscous_stress, reynolds_stress, total_stress,
    )


def get_odt_statistics_reference(input_params):
    
    # --- Get ODT input parameters ---
    
    utau       = input_params["utau"]
    Retau      = input_params["Retau"]
    delta      = input_params["delta"]
    rho        = input_params["rho"]
    kvisc      = input_params["kvisc"]

    # get filename of reference statistics
    reference_dir = os.path.join(f"./ODT_reference/Re{Retau}")
    reference_stat_file = os.path.join(reference_dir, "statistics_reference.dat")    
    data_stat = np.loadtxt(reference_stat_file)

    # -> check the file rows correspond to the expected variables:
    with open(reference_stat_file,'r') as f:
        rows_info = f.readlines()[3].split() # 4th line of the file
    rows_info_expected = '#         1_posUnif        2_uvel_mean        3_uvel_rmsf       4_uvel_Fpert        5_vvel_mean        6_vvel_rmsf       7_vvel_Fpert        8_wvel_mean        9_wvel_rmsf      10_wvel_Fpert             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz\n'.split()
    assert rows_info == rows_info_expected, f"statistic files rows do not correspond to the expected variables" \
        f"rows variables (expected): \n{rows_info_expected} \n" \
        f"rows variables (current): \n{rows_info}"
    
    # -> get data
    yu           = data_stat[:,0]
    #
    um_data      = data_stat[:,1] 
    urmsf_data   = data_stat[:,2] 
    uFpert_data  = data_stat[:,3]
    #
    vm_data      = data_stat[:,4] 
    vrmsf_data   = data_stat[:,5] 
    vFpert_data  = data_stat[:,6] 
    #
    wm_data      = data_stat[:,7] 
    wrmsf_data   = data_stat[:,8] 
    wFpert_data  = data_stat[:,9]  
    # reynolds stress terms
    ufufm_data   = data_stat[:,10]  # column header: 11_Rxx
    vfvfm_data   = data_stat[:,11]  # 12_Ryy
    wfwfm_data   = data_stat[:,12]  # 13_Rzz
    ufvfm_data   = data_stat[:,13]  # 14_Rxy
    ufwfm_data   = data_stat[:,14]  # 15_Rxz
    vfwfm_data   = data_stat[:,15]  # 16_Ryz

    ### Mirror data (symmetric in the y-direction from the channel center)
    # mirror data indexs
    nunif        = len(um_data)
    nunif2       = int(nunif/2)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)
    # mirror data
    um_data      = 0.5 * ( um_data[:nunifb]      + np.flipud(um_data[nunift:])      )
    urmsf_data   = 0.5 * ( urmsf_data[:nunifb]   + np.flipud(urmsf_data[nunift:])   )
    uFpert_data  = 0.5 * ( uFpert_data[:nunifb]  + np.flipud(uFpert_data[nunift:])  )
    vm_data      = 0.5 * ( vm_data[:nunifb]      + np.flipud(vm_data[nunift:])      )
    vrmsf_data   = 0.5 * ( vrmsf_data[:nunifb]   + np.flipud(vrmsf_data[nunift:])   )
    vFpert_data  = 0.5 * ( vFpert_data[:nunifb]  + np.flipud(vFpert_data[nunift:])  )
    wm_data      = 0.5 * ( wm_data[:nunifb]      + np.flipud(wm_data[nunift:])      )
    wrmsf_data   = 0.5 * ( wrmsf_data[:nunifb]   + np.flipud(wrmsf_data[nunift:])   )
    wFpert_data  = 0.5 * ( wFpert_data[:nunifb]  + np.flipud(wFpert_data[nunift:])  )
    ufufm_data   = 0.5 * ( ufufm_data[:nunifb]   + np.flipud(ufufm_data[nunift:])   )
    vfvfm_data   = 0.5 * ( vfvfm_data[:nunifb]   + np.flipud(vfvfm_data[nunift:])   )
    wfwfm_data   = 0.5 * ( wfwfm_data[:nunifb]   + np.flipud(wfwfm_data[nunift:])   )
    ufvfm_data   = 0.5 * ( ufvfm_data[:nunifb]   + np.flipud(ufvfm_data[nunift:])   )
    ufwfm_data   = 0.5 * ( ufwfm_data[:nunifb]   + np.flipud(ufwfm_data[nunift:])   )
    vfwfm_data   = 0.5 * ( vfwfm_data[:nunifb]   + np.flipud(vfwfm_data[nunift:])   )

    # ------------ scale y to y+ ------------

    # y-coordinates
    yu     = yu[:nunifb] + delta
    ydelta = yu/delta
    yplus  = yu * utau / kvisc

    # Check: Re_tau and u_tau of ODT data
    dumdy0     = (um_data[1]-um_data[0])/(yu[1]-yu[0])
    utauOdt  = np.sqrt(kvisc * np.abs(dumdy0))
    RetauOdt = utauOdt * delta / kvisc
    print(f"\n[get_odt_statistics_reference] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauOdt}")
    print(f"[get_odt_statistics_reference] Expected u_tau = {utau} vs. Effective u_tau = {utauOdt}")

    # ------------ Compute Stress Decomposition ------------
    
    # Stress decomposition: Viscous, Reynolds and Total stress
    dumdy = (um_data[1:] - um_data[:-1])/(yu[1:] - yu[:-1])
    viscous_stress_  = kvisc * rho * dumdy
    reynolds_stress_ = - rho * ufvfm_data[:-1]
    total_stress_    = viscous_stress_ + reynolds_stress_

    # Add "0" at grid boundaries to quantities computed from 1st- and 2nd-order derivatives to vstack vectors of the same size
    # -> involve 1st-order forward finite differences
    viscous_stress  = np.zeros(nunifb); viscous_stress[:-1]  = viscous_stress_  
    reynolds_stress = np.zeros(nunifb); reynolds_stress[:-1] = reynolds_stress_  
    total_stress    = np.zeros(nunifb); total_stress[:-1]    = total_stress_  

    return (ydelta, yplus, 
            um_data, urmsf_data, uFpert_data,
            vm_data, vrmsf_data, vFpert_data,
            wm_data, wrmsf_data, wFpert_data,
            ufufm_data, vfvfm_data, wfwfm_data, ufvfm_data, ufwfm_data, vfwfm_data,
            viscous_stress, reynolds_stress, total_stress,
    )


def get_odt_udata_rt(input_params):
    """
    """

    # --- Get ODT input parameters ---
    
    utau      = input_params["utau"]
    delta      = input_params["delta"]
    kvisc      = input_params["kvisc"]
    Retau      = input_params["Retau"]
    case_name  = input_params["caseN"]
    rlzStr     = input_params["rlzStr"]
    dTimeStart = input_params["dTimeStart"]
    dTimeEnd   = input_params["dTimeEnd"]
    dTimeStep  = input_params["dTimeStep"]
    tEndAvg    = input_params["tEndAvg"]
    
    # --- Get dmp file with dump number ***** corresponding to tEndAvg, or the closest one from below ---
    dTimes     = np.round(np.arange(dTimeStart, dTimeEnd+1e-6, dTimeStep), 6)
    tEndAvgDmpIdx = np.sum(tEndAvg > dTimes) 
    tEndAvgDmpStr = f"{tEndAvgDmpIdx:05d}"

    # --- Get vel. statistics computed during runtime at last time increment ---

    fstat     = f"../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_{tEndAvgDmpStr}.dat"
    print(f"Get statistics data from file: {fstat}")
    data_stat = np.loadtxt(fstat)

    # -> check the file rows correspond to the expected variables:
    with open(fstat,'r') as f:
        rows_info = f.readlines()[3].split() # 4th line of the file
    rows_info_expected = '#         1_posUnif        2_uvel_mean        3_uvel_rmsf       4_uvel_Fpert        5_vvel_mean        6_vvel_rmsf       7_vvel_Fpert        8_wvel_mean        9_wvel_rmsf      10_wvel_Fpert             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz\n'.split()
    assert rows_info == rows_info_expected, f"statistic files rows do not correspond to the expected variables" \
        f"rows variables (expected): \n{rows_info_expected} \n" \
        f"rows variables (current): \n{rows_info}"
    
    # --- get data
    yu           = data_stat[:,0]
    um_data      = data_stat[:,1] 
    urmsf_data   = data_stat[:,2] 

    # --- Mirror data (symmetric in the y-direction from the channel center)
    # mirror data indexs
    nunif        = len(um_data)
    nunif2       = int(nunif/2)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)
    # mirror data
    um_data      = 0.5 * ( um_data[:nunifb]      + np.flipud(um_data[nunift:])      )
    urmsf_data   = 0.5 * ( urmsf_data[:nunifb]   + np.flipud(urmsf_data[nunift:])   )
    
    # --- scale y to y+
    yu     = yu[:nunifb] + delta # mirror and add delta value
    ydelta = yu/delta
    yplus  = yu * utau / kvisc

    # Check: Re_tau and u_tau of ODT data
    dudy     = (um_data[1]-um_data[0])/(yu[1]-yu[0])
    utauOdt  = np.sqrt(kvisc * np.abs(dudy))
    RetauOdt = utauOdt * delta / kvisc
    print(f"\n[get_odt_udata_rt] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauOdt}")
    print(f"[get_odt_udata_rt] Expected u_tau = {utau} vs. Effective u_tau = {utauOdt}")

    return (ydelta, yplus, um_data, urmsf_data)


def get_dns_statistics(Retau, input_params):

    # --- Get DNS input parameters (prescribed) ---
    rho    = 1.0
    delta  = 1.0
    utau   = 1.0
    nu     = utau * delta / Retau
    mu     = rho * nu
    kvisc  = mu / rho # = nu 
    
    if (Retau == 590):
        filename_dns = "DNS_statistics/Re590/dnsChannel_Re590_means.dat"
        print(f"\nGetting DNS-means data from {filename_dns}")
        # Dataset columns
        # 0    | 1    | 2     | 3         | 4     | 5         | 6
        # y/h  | y+   | Umean | dUmean/dy | Wmean | dWmean/dy | Pmean

        dns_means = np.loadtxt(filename_dns)
        ydelta = dns_means[:,0] # y/delta = y, as delta = 1
        yplus  = dns_means[:,1]
        um = dns_means[:,2]     # Umean normalized by U_tau (= u+_mean)

        filename_dns = "DNS_statistics/Re590/dnsChannel_Re590_reynolds_stress.dat"
        print(f"Getting DNS-reynolds data from {filename_dns}")
        # Dataset columns
        # 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7
        # y    | y+   | R_uu | R_vv | R_ww | R_uv | R_uw | R_vw
        dns_reynolds_stress = np.loadtxt(filename_dns)
        ufufm = dns_reynolds_stress[:,2] # = Rxx+, component of the reynolds stress tensor component, normalized by U_tau^2 = 1
        vfvfm = dns_reynolds_stress[:,3] # = Ryy+
        wfwfm = dns_reynolds_stress[:,4] # = Rzz+
        ufvfm = dns_reynolds_stress[:,5] # = Rxy+
        ufwfm = dns_reynolds_stress[:,6] # = Rxz+
        vfwfm = dns_reynolds_stress[:,7] # = Ryz+

        urmsf = np.sqrt(ufufm) # = sqrt(mean(u'u')), normalized by U_tau^2 = 1 prescribed
        vrmsf = np.sqrt(vfvfm)
        wrmsf = np.sqrt(wfwfm)
    
    else:
        filename_dns = f"DNS_statistics/Re{Retau}/profiles/Re{Retau}.prof"
        print(f"\nGetting DNS-reynolds data from {filename_dns}")
        # Dataset columns:
        # 0    | 1    | 2    | 3    | 4    | 5    | 6      | 7      | 8      | 9      | 10   | 11   | 12   | 13   | 14   | 15     | 16   
        # y/h  | y+   | U+   | u'+  | v'+  | w'+  | -Om_z+ | om_x'+ | om_y'+ | om_z'+ | uv'+ | uw'+ | vw'+ | pr'+ | ps'+ | psto'+ | p'    
        dns    = np.loadtxt(filename_dns,comments="%")
        ydelta = dns[:,0]*2 # y/delta = (y/h)*2 = y, as delta = 1
        yplus  = dns[:,1]   # y+ = y * u_tau / nu 
        um     = dns[:,2]   # u+_mean 

        urmsf  = dns[:,3] # named as u'+, normalized by U_tau
        vrmsf  = dns[:,4] # named as v'+, normalized by U_tau
        wrmsf  = dns[:,5] # named as w'+, normalized by U_tau
        
        ufufm  = urmsf**2    
        vfvfm  = vrmsf**2
        wfwfm  = wrmsf**2
        ufvfm  = dns[:,10]
        ufwfm  = dns[:,11]
        vfwfm  = dns[:,12]

    # Compare stablished and computed Retau and utau
    dudy_wall = (um[1]-um[0])/(ydelta[1]-ydelta[0])
    utauDns  = np.sqrt(kvisc * np.abs(dudy_wall) / rho)
    RetauDns = utauDns * delta / kvisc
    print(f"\n[get_dns_statistics] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauDns}")
    print(f"[get_dns_statistics] Expected u_tau = {utau} vs. Effective u_tau = {utauDns}")

    # --- Stress decomposition: Viscous, Reynolds and Total stress ---
    dumdy = (um[1:] - um[:-1])/(ydelta[1:] - ydelta[:-1])
    viscous_stress_  = kvisc * rho * dumdy
    reynolds_stress_ = - rho * ufvfm[:-1]
    total_stress_    = viscous_stress_ + reynolds_stress_

    # --- Viscous Transport budget ---
    # vt_i = d^2(avg(u_{i,rmsf}^2))/dy^2
    f_uk  = urmsf**2 - um**2 
    Deltay_k = (ydelta[2:]-ydelta[:-2])/2 # length in y-coordinate of cell k from ODT grid 
    vt_u_ = (kvisc) * ( (f_uk[2:] - 2*f_uk[1:-1] + f_uk[:-2])/(Deltay_k**2) )
    vt_u_plus_ = vt_u_ * ( delta / utau**3 )

    # --- Dissipation budget ---
    # d_i = avg( ( d(u_{i,rmsf})/dy )^2 )
    # -> cannot be calculated with available DNS data :/

    # --- Production budget ---
    dumdy = (um[2:] - um[:-2])/(ydelta[2:] - ydelta[:-2]) # 1st-order central finite difference
    p_u_  = - 2 * ufvfm[1:-1] * dumdy
    p_u_plus_ = p_u_ * ( delta / utau**3 ) 

    # Add "0" at grid boundaries to quantities computed from 1st- and 2nd-order derivatives to vstack vectors of the same size
    # -> involve 1st-order forward finite differences
    nunif2 = um.size
    viscous_stress  = np.zeros(nunif2); viscous_stress[:-1]  = viscous_stress_  
    reynolds_stress = np.zeros(nunif2); reynolds_stress[:-1] = reynolds_stress_  
    total_stress    = np.zeros(nunif2); total_stress[:-1]    = total_stress_  
    # -> involve 2nd-order centered finite differences
    vt_u_plus = np.zeros(nunif2); vt_u_plus[1:-1] = vt_u_plus_
    p_u_plus  = np.zeros(nunif2); p_u_plus[1:-1]  = p_u_plus_

    return (ydelta, yplus, um, urmsf, vrmsf, wrmsf, 
            ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm,
            viscous_stress, reynolds_stress, total_stress, 
            vt_u_plus, p_u_plus)


def get_time(file):
    """
    Obtain time instant of a datafile (.dat) of instantaneous odt data.

    Parameters:
        file (str): filepath of the .dat of instantaneous data, from which it is obtained the time instant. 
        
    Returns:
        time (float): time instant of the fluid data of the .dat file 

    Comments: 
        Time is stored as a comment in the first line of the .dat file, e.g.
        # time = 50.0
    """
    with open(file,"r") as f:
        first_line = f.readline().strip()
    if first_line.startswith("# time = "):
        time = float(first_line.split(" = ")[1])
    else:
        print(f"No valid format found in the first line of {file}.")
        time = None
    return time


def get_odt_statistics_post_at_chosen_averaging_times(input_params, averaging_times):
    
    # --- Get ODT input parameters ---
    
    utau         = input_params["utau"]
    delta        = input_params["delta"]
    kvisc        = input_params["kvisc"]
    Retau        = input_params["Retau"]
    case_name    = input_params["caseN"]
    rlzStr       = input_params["rlzStr"]
    dTimeStart   = input_params["dTimeStart"]
    dTimeEnd     = input_params["dTimeEnd"]
    dTimeStep    = input_params["dTimeStep"]
    domainLength = input_params["domainLength"]
    dxmin        = input_params["dxmin"]
    nunif        = input_params["nunif"]

    # un-normalize
    dxmin       *= domainLength
    
    # uniform fine grid
    nunif2       = int(nunif/2)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)

    # --- Averaging times and files identification ---
    
    dTimeVec = np.arange(dTimeStart, dTimeEnd+1e-4, dTimeStep).round(4)
    averaging_times_num = len(averaging_times)
    averaging_times_idx = []
    for t_idx in range(averaging_times_num):
         averaging_times_idx.append(np.where(dTimeVec==averaging_times[t_idx])[0][0])
    averaging_times_str = [str(idx).zfill(5) for idx in averaging_times_idx]
    if (len(averaging_times_str) != averaging_times_num):
        raise ValueError("Not all averaging_times where found!")
    
    # --- Get vel. statistics computed during runtime at chosen averaging times ---

    flist = sorted(gb.glob(f"../../data/{case_name}/data/data_{rlzStr}/dmp_*.dat"))

    # -> check the file rows correspond to the expected variables:
    with open(flist[0],'r') as f:
        rows_info = f.readlines()[3].split() # 4th line of the file
    rows_info_expected = "#             1_pos             2_posf             3_uvel             4_vvel             5_wvel\n".split()
    assert rows_info == rows_info_expected, f"statistic files rows do not correspond to the expected variables" \
        f"rows variables (expected): \n{rows_info_expected} \n" \
        f"rows variables (current): \n{rows_info}"
    
    # --- initialize variables ---

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
    um      = np.zeros([nunif, averaging_times_num])   # mean velocity
    vm      = np.zeros([nunif, averaging_times_num])
    wm      = np.zeros([nunif, averaging_times_num])
    u2m     = np.zeros([nunif, averaging_times_num])   # mean square velocity (for rmsf and reynolds stresses)
    v2m     = np.zeros([nunif, averaging_times_num])
    w2m     = np.zeros([nunif, averaging_times_num])
    uvm     = np.zeros([nunif, averaging_times_num])   # mean velocity correlations (for reynolds stresses)
    uwm     = np.zeros([nunif, averaging_times_num])
    vwm     = np.zeros([nunif, averaging_times_num])

    file_counter = 0
    for ifile in flist :
        file_counter += 1
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
        if len(idx_averaging_time)>0: # instance found
            idx = idx_averaging_time[0]
            um[:,idx]  = um_aux/file_counter   
            vm[:,idx]  = vm_aux/file_counter   
            wm[:,idx]  = wm_aux/file_counter   
            u2m[:,idx] = u2m_aux/file_counter    
            v2m[:,idx] = v2m_aux/file_counter    
            w2m[:,idx] = w2m_aux/file_counter    
            uvm[:,idx] = uvm_aux/file_counter    
            uwm[:,idx] = uwm_aux/file_counter    
            vwm[:,idx] = vwm_aux/file_counter    

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
    yu += delta          # domain center is at 0; shift so left side is zero
    yu  = yu[:nunifb]    # plotting to domain center
    ydelta = yu/delta
    yplus  = yu * utau / kvisc 
    
    # Check: Re_tau and u_tau of ODT data
    dudy     = (um[1,-1]-um[0,-1])/(yu[1]-yu[0])
    utauOdt  = np.sqrt(kvisc * np.abs(dudy))
    RetauOdt = utauOdt * delta / kvisc
    print(f"\n[get_odt_statistics_post_at_chosen_averaging_times] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauOdt}")
    print(f"[get_odt_statistics_post_at_chosen_averaging_times] Expected u_tau = {utau} vs. Effective u_tau = {utauOdt}")
    # scale y --> y+ (note: utau should be unity)
    yplus = yu * utau / kvisc    

    return (ydelta, yplus,
            um, urmsf,
            vm, vrmsf,
            wm, wrmsf,
            ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm,
    )


def get_odt_statistics_rt_at_chosen_averaging_times_um_symmetry(input_params, averaging_times):
    """
    Parameters:
        input_params (dict): ODT input parameters dictionary
        averaging_times #TODO

    Returns:
        ODT statistics calculated during runtime, output of the simulation at each dumpTime instant
        um (np.ndarrays)
    """

    # --- Get ODT input parameters ---
    
    utau         = input_params["utau"]
    kvisc        = input_params["kvisc"] # = nu = mu / rho 
    domainLength = input_params["domainLength"]
    dxmin        = input_params["dxmin"]
    delta        = input_params["delta"]
    Retau        = input_params["Retau"]
    case_name    = input_params["caseN"]
    rlzStr       = input_params["rlzStr"]
    dTimeStart   = input_params["dTimeStart"]
    dTimeEnd     = input_params["dTimeEnd"]
    dTimeStep    = input_params["dTimeStep"]
    nunif        = input_params["nunif"]
    dxmin       *= domainLength

    # --- Averaging times and files identification ---

    dTimeVec = np.arange(dTimeStart, dTimeEnd+1e-3, dTimeStep).round(2)
    averaging_times_num = len(averaging_times)
    averaging_times_idx = []
    for t_idx in range(averaging_times_num):
         averaging_times_idx.append(np.where(dTimeVec==averaging_times[t_idx])[0][0])
    averaging_times_str = [str(idx).zfill(5) for idx in averaging_times_idx]
    if (len(averaging_times_str) != averaging_times_num):
        raise ValueError("Not all averaging_times where found!")

    # --- Compute ODT computational data ---

    flist = [f'../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_{s}.dat' for s in averaging_times_str]

    # Num points uniform grid
    nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)

    um = np.zeros([nunif, averaging_times_num]) 

    for i in range(averaging_times_num):

        data = np.loadtxt(flist[i])
        yu_data = data[:,0] # not normalized
        um_data = data[:,1] # normalized by u_tau, u is in fact u+
        
        if i == 0:
            yu = np.copy(yu_data)
        else: 
            assert (yu==yu_data).all(), "ERROR: statistics uniform fine grid 'yu_data' should be equal to linspace 'yu'"
        
        um[:,i] = um_data

    # All data, not symmetrical
    um_all = np.copy(um)
    yu_all = np.copy(yu)

    # mirror data (symmetric channel in y-axis)
    um = 0.5 * (um[:nunifb,:]  + np.flipud(um[nunift:,:]))  # um_

    # ------------------------ Calculate symmetric field ------------------------
    
    # calculate 'symmetric' part of um in all fine grid
    um_symmetric_all  = np.copy(um_all)
    um_symmetric_half = np.copy(um)
    um_symmetric_all[:nunifb,:]  = um_symmetric_half
    um_symmetric_all[nunift:,:]  = np.flipud(um_symmetric_half)

    # ----------- Calculate indicate - Integral deviation from symmetry -----------

    # CI = Eps_s : Convergence Indicator, from Pirozzoli207-A --- along fine uniform grid
    CI  = np.sqrt( 0.5 * np.sum( (um_all - um_symmetric_all)**2 , axis = 0) ) # rmse
    ###print("\n(ODT) Convergence Indicator (CI, Esp_s) for each averaging time:")
    ###for i in range(len(averaging_times)):
    ###    print(f"     At tavg = {averaging_times[i]:.1f} : CI = {CI[i]:.3f}")

    # ------------ scale y to y+ ------------

    # y-coordinates
    yu        = yu[:nunifb] + delta   # plotting to domain center
    ydelta    = yu / delta
    yplus     = yu * utau / kvisc 
    ydelta_all = yu_all / delta
    # Check: Re_tau and u_tau of ODT data
    dudy = (um[1,-1]-um[0,-1])/(yu[1]-yu[0])
    utauOdt = np.sqrt(kvisc * np.abs(dudy))
    RetauOdt = utauOdt * delta / kvisc
    print(f"\n[get_odt_statistics_rt_at_chosen_averaging_times_um_symmetry] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauOdt}")
    print(f"[get_odt_statistics_rt_at_chosen_averaging_times_um_symmetry] Expected u_tau = {utau} vs. Effective u_tau = {utauOdt}")
    
    return (ydelta, yplus, um, CI, ydelta_all, um_all, um_symmetric_all)


def get_odt_statistics_rt_at_chosen_averaging_times(input_params, averaging_times):

    # --- Get ODT input parameters ---
    
    utau         = input_params["utau"]
    delta        = input_params["delta"]
    kvisc        = input_params["kvisc"]
    Retau        = input_params["Retau"]
    case_name    = input_params["caseN"]
    rlzStr       = input_params["rlzStr"]
    dTimeStart   = input_params["dTimeStart"]
    dTimeEnd     = input_params["dTimeEnd"]
    dTimeStep    = input_params["dTimeStep"]
    domainLength = input_params["domainLength"]
    dxmin        = input_params["dxmin"]
    nunif        = input_params["nunif"]

    # un-normalize
    dxmin       *= domainLength
    
    # uniform fine grid
    nunif2       = int(nunif/2)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)

    # --- Averaging times and files identification ---
    
    dTimeVec = np.arange(dTimeStart, dTimeEnd+1e-4, dTimeStep).round(4)
    averaging_times_num = len(averaging_times)
    averaging_times_idx = []
    for t_idx in range(averaging_times_num):
         averaging_times_idx.append(np.where(dTimeVec==averaging_times[t_idx])[0][0])
    averaging_times_str = [str(idx).zfill(5) for idx in averaging_times_idx]
    if (len(averaging_times_str) != averaging_times_num):
        raise ValueError("Not all averaging_times where found!")

    # --- Get vel. statistics computed during runtime at chosen averaging times ---

    flist = [f'../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_{s}.dat' for s in averaging_times_str]

    # -> check the file rows correspond to the expected variables:
    with open(flist[0],'r') as f:
        rows_info = f.readlines()[3].split() # 4th line of the file
    rows_info_expected = '#         1_posUnif        2_uvel_mean        3_uvel_rmsf       4_uvel_Fpert        5_vvel_mean        6_vvel_rmsf       7_vvel_Fpert        8_wvel_mean        9_wvel_rmsf      10_wvel_Fpert             11_Rxx             12_Ryy             13_Rzz             14_Rxy             15_Rxz             16_Ryz\n'.split()
    assert rows_info == rows_info_expected, f"statistic files rows do not correspond to the expected variables" \
        f"rows variables (expected): \n{rows_info_expected} \n" \
        f"rows variables (current): \n{rows_info}"

    # --- initialize variables ---

    yu           = np.zeros([nunif, averaging_times_num])
    # velocity data & F-perturbation
    um_data      = np.zeros([nunif, averaging_times_num])
    urmsf_data   = np.zeros([nunif, averaging_times_num])
    uFpert_data  = np.zeros([nunif, averaging_times_num])
    vm_data      = np.zeros([nunif, averaging_times_num])
    vrmsf_data   = np.zeros([nunif, averaging_times_num])
    vFpert_data  = np.zeros([nunif, averaging_times_num])
    wm_data      = np.zeros([nunif, averaging_times_num])
    wrmsf_data   = np.zeros([nunif, averaging_times_num])
    wFpert_data  = np.zeros([nunif, averaging_times_num])
    # reynolds str
    ufufm_data   = np.zeros([nunif, averaging_times_num])
    vfvfm_data   = np.zeros([nunif, averaging_times_num])
    wfwfm_data   = np.zeros([nunif, averaging_times_num])
    ufvfm_data   = np.zeros([nunif, averaging_times_num])
    ufwfm_data   = np.zeros([nunif, averaging_times_num])
    vfwfm_data   = np.zeros([nunif, averaging_times_num])
    # anisotropy t
    # lambda0_data = np.zeros([nunif, averaging_times_num])
    # lambda1_data = np.zeros([nunif, averaging_times_num])
    # lambda2_data = np.zeros([nunif, averaging_times_num])
    # xmap1_data   = np.zeros([nunif, averaging_times_num])
    # xmap2_data   = np.zeros([nunif, averaging_times_num])

    # --- get data of statistics calc. at runtime at chosen averaging times ---

    for i in range(averaging_times_num):

        data_stat         = np.loadtxt(flist[i])

        yu[:,i]           = data_stat[:,0]
        um_data[:,i]      = data_stat[:,1] 
        urmsf_data[:,i]   = data_stat[:,2] 
        uFpert_data[:,i]  = data_stat[:,3]
        vm_data[:,i]      = data_stat[:,4] 
        vrmsf_data[:,i]   = data_stat[:,5] 
        vFpert_data[:,i]  = data_stat[:,6] 
        wm_data[:,i]      = data_stat[:,7] 
        wrmsf_data[:,i]   = data_stat[:,8] 
        wFpert_data[:,i]  = data_stat[:,9]  
        ufufm_data[:,i]   = data_stat[:,10]
        vfvfm_data[:,i]   = data_stat[:,11]
        wfwfm_data[:,i]   = data_stat[:,12]
        ufvfm_data[:,i]   = data_stat[:,13]
        ufwfm_data[:,i]   = data_stat[:,14]
        vfwfm_data[:,i]   = data_stat[:,15]
        # lambda0_data[:,i] = data_stat[:,16] 
        # lambda1_data[:,i] = data_stat[:,17] 
        # lambda2_data[:,i] = data_stat[:,18]
        # xmap1_data[:,i]   = data_stat[:,19] 
        # xmap2_data[:,i]   = data_stat[:,20] 

    # mirror data (symmetric in the y-direction from the channel center)
    um_data      = 0.5 * ( um_data[:nunifb,:]      + np.flipud(um_data[nunift:,:])      )
    urmsf_data   = 0.5 * ( urmsf_data[:nunifb,:]   + np.flipud(urmsf_data[nunift:,:])   )
    uFpert_data  = 0.5 * ( uFpert_data[:nunifb,:]  + np.flipud(uFpert_data[nunift:,:])  )
    vm_data      = 0.5 * ( vm_data[:nunifb,:]      + np.flipud(vm_data[nunift:,:])      )
    vrmsf_data   = 0.5 * ( vrmsf_data[:nunifb,:]   + np.flipud(vrmsf_data[nunift:,:])   )
    vFpert_data  = 0.5 * ( vFpert_data[:nunifb,:]  + np.flipud(vFpert_data[nunift:,:])  )
    wm_data      = 0.5 * ( wm_data[:nunifb,:]      + np.flipud(wm_data[nunift:,:])      )
    wrmsf_data   = 0.5 * ( wrmsf_data[:nunifb,:]   + np.flipud(wrmsf_data[nunift:,:])   )
    wFpert_data  = 0.5 * ( wFpert_data[:nunifb,:]  + np.flipud(wFpert_data[nunift:,:])  )
    ufufm_data   = 0.5 * ( ufufm_data[:nunifb,:]   + np.flipud(ufufm_data[nunift:,:])   )
    vfvfm_data   = 0.5 * ( vfvfm_data[:nunifb,:]   + np.flipud(vfvfm_data[nunift:,:])   )
    wfwfm_data   = 0.5 * ( wfwfm_data[:nunifb,:]   + np.flipud(wfwfm_data[nunift:,:])   )
    ufvfm_data   = 0.5 * ( ufvfm_data[:nunifb,:]   + np.flipud(ufvfm_data[nunift:,:])   )
    ufwfm_data   = 0.5 * ( ufwfm_data[:nunifb,:]   + np.flipud(ufwfm_data[nunift:,:])   )
    vfwfm_data   = 0.5 * ( vfwfm_data[:nunifb,:]   + np.flipud(vfwfm_data[nunift:,:])   )
    # lambda0_data = 0.5 * ( lambda0_data[:nunifb,:] + np.flipud(lambda0_data[nunift:,:]) )
    # lambda1_data = 0.5 * ( lambda1_data[:nunifb,:] + np.flipud(lambda1_data[nunift:,:]) )
    # lambda2_data = 0.5 * ( lambda2_data[:nunifb,:] + np.flipud(lambda2_data[nunift:,:]) )
    # xmap1_data   = 0.5 * ( xmap1_data[:nunifb,:]   + np.flipud(xmap1_data[nunift:,:])   )
    # xmap2_data   = 0.5 * ( xmap2_data[:nunifb,:]   + np.flipud(xmap2_data[nunift:,:])   )

    # ------------ scale y to y+ ------------

    yu     = yu[:nunifb,-1] + delta     # only from last snapshot, it is the same along all snapshots
    ydelta = yu / delta
    yplus  = yu * utau / kvisc 
    
    # Check: Re_tau and u_tau of ODT data
    dudy     = (um_data[1,-1]-um_data[0,-1])/(yu[1]-yu[0])
    utauOdt  = np.sqrt(kvisc * np.abs(dudy))
    RetauOdt = utauOdt * delta / kvisc
    print(f"\n[get_odt_statistics_rt_at_chosen_averaging_times] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauOdt}")
    print(f"[get_odt_statistics_rt_at_chosen_averaging_times] Expected u_tau = {utau} vs. Effective u_tau = {utauOdt}")

    return (ydelta, yplus,                      # shape [nunif]               
            um_data, urmsf_data, uFpert_data,   # shape [nunif, len(averaging_times)]
            vm_data, vrmsf_data, vFpert_data,
            wm_data, wrmsf_data, wFpert_data,
            ufufm_data, vfvfm_data, wfwfm_data, ufvfm_data, ufwfm_data, vfwfm_data,
            #lambda0_data, lambda1_data, lambda2_data, xmap1_data, xmap2_data,
    )


def compute_convergence_indicator_odt_tEndAvg(input_params):
    
    # --- Get ODT input parameters ---
    utau         = input_params["utau"]
    domainLength = input_params["domainLength"]
    dxmin        = input_params["dxmin"]
    delta        = input_params["delta"]
    case_name    = input_params["caseN"]
    rlzStr       = input_params["rlzStr"]
    kvisc        = input_params["kvisc"]
    nunif        = input_params["nunif"]
    dTimeStart   = input_params["dTimeStart"]
    dTimeEnd     = input_params["dTimeEnd"]
    dTimeStep    = input_params["dTimeStep"]
    tEndAvg      = input_params["tEndAvg"]
    
    # --- Get dmp file with dump number ***** corresponding to tEndAvg, or the closest one from below ---
    dTimes        = np.round(np.arange(dTimeStart, dTimeEnd+1e-6, dTimeStep), 6)
    tEndAvgDmpIdx = np.sum(tEndAvg > dTimes) 
    tEndAvgDmpStr = f"{tEndAvgDmpIdx:05d}"
    
    # un-normalize
    dxmin *= domainLength
    # uniform fine grid
    nunif2 = int(nunif/2)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)

    # --- Get ODT statistics computed during runtime at last time increment ---

    fstat     = f"../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_{tEndAvgDmpStr}.dat"
    data_stat = np.loadtxt(fstat)

    # uniform fine grid (u.f.g) - '1_posUnif' 
    yu = data_stat[:,0]
    # u_avg statistic runtime-calculated in u.f.g - '2_uvelmean'
    um = data_stat[:,1]
    assert len(yu) == nunif, "ERROR: size error, uniform fine grid should have length 'nunif'"

    # ------------------------ Calculate symmetric field ------------------------

    # calculate 'symmetric' part of um
    um_symmetric_half = 0.5*(um[:nunifb] + np.flipud(um[nunift:]))  
    um_symmetric = np.copy(um)
    um_symmetric[:nunifb]  = um_symmetric_half
    um_symmetric[nunift:]  = np.flipud(um_symmetric_half)

    # ----------- Calculate indicate - Integral deviation from symmetry -----------

    # dy necessary for computing the integral of CI - really not necessary for the integral, as fine grid is uniform 
    ### dyu = yu[1:] - yu[:-1] 
    # CI = Eps_s : Convergence Indicator, from Pirozzoli207-A
    CI  = np.sqrt( 0.5 * np.sum( (um - um_symmetric)**2 ) ) # rmse
    print("\n(ODT) Convergence Indicator at tEnd (CI, Esp_s) = ", CI)

    # ------------ scale y to y+ ------------
    ydelta = yu / delta 

    return (CI, ydelta, um, um_symmetric)


def compute_convergence_indicator_odt_along_avg_time(input_params):
    
    # --- Get ODT input parameters ---

    domainLength = input_params["domainLength"]
    dxmin        = input_params["dxmin"]
    case_name    = input_params["caseN"]
    rlzStr       = input_params["rlzStr"]
    nunif        = input_params["nunif"]
    tBeginAvg    = input_params["tBeginAvg"]
    tEndAvg      = input_params["tEndAvg"]
    # un-normalize
    dxmin *= domainLength
    # uniform fine grid
    nunif2 = int(nunif/2)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)

    # --------------- Get ODT statistics ---------------

    flist_stat     = sorted(gb.glob(f'../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_*.dat'))
    CI_list = []
    time_list = []

    for ifile in flist_stat: 

        # Check file time is tBeginAvg <= currentTime <= tEndAvg
        currentTime = get_time(ifile)
        if tBeginAvg <= currentTime and currentTime <= tEndAvg: 
            
            data = np.loadtxt(ifile)
            # u_avg statistic runtime-calculated in u.f.g - '2_uvelmean'
            um = data[:,1]

            # ------------------------ Calculate symmetric field ------------------------

            # calculate 'symmetric' part of um
            um_symmetric_half      = 0.5*(um[:nunifb] + np.flipud(um[nunift:]))  
            um_symmetric           = np.copy(um)
            um_symmetric[:nunifb]  = um_symmetric_half
            um_symmetric[nunift:]  = np.flipud(um_symmetric_half)

            # ----------- Calculate indicate - Integral deviation from symmetry -----------

            # CI = Eps_s : Convergence Indicator, from Pirozzoli207-A
            CI = np.sqrt( 0.5 * np.sum( (um - um_symmetric)**2 ) ) # rmse
            CI_list.append(CI)

            time_list.append(get_time(ifile))

    return (time_list, CI_list)


def compute_odt_statistics_at_chosen_time(input_params, time_end):
    """
    Compute ODT statistics from multiple .dat files with instantaneous data
    at increasing simulation time until chosen time time_end is reached

    Parameters:
        input_params (dict): ODT input parameters dictionary
    
    Returns:
        ODT statistics calculated up until time = time_end, specifically
        (ydelta, um, urmsf, vrmsf, wrmsf, ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm)

    """

    # --- Get ODT input parameters ---
    
    rho       = input_params["rho"]
    kvisc     = input_params["kvisc"] # = nu = mu / rho 
    dxmin     = input_params["dxmin"]
    delta     = input_params["delta"]
    Retau     = input_params["Retau"]
    utau      = input_params["utau"]
    case_name = input_params["caseN"]
    rlzStr    = input_params["rlzStr"]
    nunif     = input_params["nunif"]
    
    # un-normalize
    domainLength = input_params["domainLength"]
    dxmin *= domainLength

    # --- Compute ODT computational data ---

    flist = sorted(gb.glob(f'../../data/{case_name}/data/data_{rlzStr}/dmp_*.dat'))
    flist_stat = sorted(gb.glob(f'../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_*.dat'))

    nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)
    nunifb, nunift = get_nunif2_walls(nunif, nunif2)

    yu  = np.linspace(-delta,delta,nunif) # uniform grid in y-axis
    # empty vectors of time-averaged quantities
    um  = np.zeros(nunif)        # mean velocity, calulated in post-processing from instantaneous velocity
    vm  = np.zeros(nunif)
    wm  = np.zeros(nunif)
    u2m = np.zeros(nunif)        # mean square velocity (for rmsf and reynolds stresses)
    v2m = np.zeros(nunif)
    w2m = np.zeros(nunif)
    uvm = np.zeros(nunif)        # mean velocity correlations (for reynolds stresses)
    uwm = np.zeros(nunif)
    vwm = np.zeros(nunif)

    fcounter = 0
    for ifile in flist :

        # ------------------ Check instantaneous time < time_end ------------------
        
        time_current = get_time(ifile)
        if time_current > time_end:
            break
    
        # ------------------ (get) Instantaneous velocity ------------------

        data = np.loadtxt(ifile)
        y    = data[:,0] # = y/delta, as delta = 1
        u    = data[:,2] # normalized by u_tau, u is in fact u+
        v    = data[:,3] # normalized by u_tau, v is in fact v+
        w    = data[:,4] # normalized by u_tau, w is in fact w+ 

        # interpolate to uniform grid
        uu = interp1d(y, u, fill_value='extrapolate')(yu)  
        vv = interp1d(y, v, fill_value='extrapolate')(yu)
        ww = interp1d(y, w, fill_value='extrapolate')(yu)

        # ------------------ (compute) Velocity statistics, from instantaneous values ------------------

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

        fcounter += 1

    # (computed) means
    um /= fcounter
    vm /= fcounter
    wm /= fcounter
    um = 0.5*(um[:nunifb] + np.flipud(um[nunift:]))  # mirror data (symmetric)
    vm = 0.5*(vm[:nunifb] + np.flipud(vm[nunift:]))
    wm = 0.5*(wm[:nunifb] + np.flipud(wm[nunift:]))

    # squared means
    u2m /= fcounter
    v2m /= fcounter
    w2m /= fcounter
    u2m = 0.5*(u2m[:nunifb] + np.flipud(u2m[nunift:]))
    v2m = 0.5*(v2m[:nunifb] + np.flipud(v2m[nunift:]))
    w2m = 0.5*(w2m[:nunifb] + np.flipud(w2m[nunift:]))

    # velocity correlations
    uvm /= fcounter
    uwm /= fcounter
    vwm /= fcounter
    uvm = 0.5*(uvm[:nunifb] + np.flipud(uvm[nunift:]))
    uwm = 0.5*(uwm[:nunifb] + np.flipud(uwm[nunift:]))
    vwm = 0.5*(vwm[:nunifb] + np.flipud(vwm[nunift:]))

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

    # --- y-coordinate, y+ ---
    yu     = yu[:nunifb] + delta   # plotting to domain center, domain center is at 0; shift so left side is zero
    ydelta = yu / delta
    yplus = yu * utau / kvisc

    # Check: Re_tau and u_tau from ODT simulation
    dudy = (um[1]-um[0])/(yu[1]-yu[0])
    utauOdt = np.sqrt(kvisc * np.abs(dudy) / rho)
    RetauOdt = utauOdt * delta / kvisc
    print(f"\n[compute_odt_statistics_at_chosen_time] Expected Re_tau = {Retau} vs. Effective Re_tau = {RetauOdt}")
    print(f"[compute_odt_statistics_at_chosen_time] Expected u_tau = {utau} vs. Effective u_tau = {utauOdt}")

    return (ydelta, yplus, um, urmsf, vrmsf, wrmsf, 
            ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm)


def get_effective_dTimeEnd(case_name, rlzStr):

    # --- Get vel. statistics computed during runtime at last time increment ---

    flist_stat = sorted(gb.glob(f'../../data/{case_name}/data/data_{rlzStr}/statistics/stat_dmp_*.dat'))
    flast      = flist_stat[-1]

    return get_time(flast)


#-----------------------------------------------------------------------------------------
#           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
#-----------------------------------------------------------------------------------------

def check_realizability_conditions(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, verbose=False):

    #------------ Realizability conditions ---------------

    # help: .all() ensures the condition is satisfied in all grid points

    # COND 1: Rii >= 0, for i = 1,2,3

    cond0_0 = ( Rxx >= 0 ).all()    # i = 1
    cond0_1 = ( Ryy >= 0 ).all()    # i = 2
    cond0_2 = ( Rzz >= 0 ).all()    # i = 3
    cond0   = cond0_0 and cond0_1 and cond0_2

    # COND 2: Rij^2 <= Rii*Rjj, for i!=j

    cond1_0 = ( Rxy**2 <= Rxx * Ryy ).all()     # i = 0, j = 1
    cond1_1 = ( Rxz**2 <= Rxx * Rzz ).all()     # i = 0, j = 2
    cond1_2 = ( Ryz**2 <= Ryy * Rzz ).all()     # i = 1, j = 2
    cond1   = cond1_0 and cond1_1 and cond1_2

    # COND 3: det(Rij) >= 0
    detR  = Rxx * Ryy * Rzz + 2 * Rxy * Rxz * Ryz - (Rxx * Ryz * Ryz + Ryy * Rxz * Rxz + Rzz * Rxy * Rxy) # np.linalg.det(R_ij), length #num_points
    # detR  = np.linalg.det(R_ij)
    # enforce detR == 0 if -eps < detR < 0 due to computational error  
    for i in range(len(detR)):
        if detR[i] > -1e-15 and detR[i] < 0.0:
            detR[i] = 0.0
    cond2 = ( detR >= 0 ).all()

    if cond0 and cond1 and cond2:
        if verbose:
            print("\nCONGRATULATIONS, the reynolds stress tensor satisfies REALIZABILITY CONDITIONS.")
    else:
        message = f"The reynolds stress tensor does not satisfy REALIZABILITY CONDITIONS: cond0 = {cond0}, cond1 = {cond1}, cond2 = {cond2}"
        print(detR)
        raise Exception(message)


def compute_reynolds_stress_dof(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, 
                                tensor_kk_tolerance   = 1.0e-8, 
                                eigenvalues_tolerance = 1.0e-8, 
                                verbose = False,
                                x1c = np.array( [ 1.0 , 0.0 ] ),
                                x2c = np.array( [ 0.0 , 0.0 ] ),
                                x3c = np.array( [ 0.5 , math.sqrt(3.0)/2.0 ] )):

    check_realizability_conditions(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz)
    
    # Computed for each point of the grid
    # If the trace of the reynolds stress tensor (2 * TKE) is too small, the corresponding 
    # datapoint is omitted, because the anisotropy tensor would -> infinity, as its equation
    # contains the multiplier ( 1 / (2*TKE) )
    
    # initialize arrays
    num_points = len(Rxx)
    Rkk     = np.zeros(num_points)
    lambda1 = np.zeros(num_points)
    lambda2 = np.zeros(num_points)
    lambda3 = np.zeros(num_points)
    xmap1   = np.zeros(num_points)
    xmap2   = np.zeros(num_points)

    for p in range(num_points):

        #------------ Reynolds stress tensor ---------------

        R_ij      = np.zeros([3, 3])
        R_ij[0,0] = Rxx[p]
        R_ij[0,1] = Rxy[p]
        R_ij[0,2] = Rxz[p]
        R_ij[1,0] = Rxy[p]
        R_ij[1,1] = Ryy[p]
        R_ij[1,2] = Ryz[p]
        R_ij[2,0] = Rxz[p]
        R_ij[2,1] = Ryz[p]
        R_ij[2,2] = Rzz[p]

        #------------ Anisotropy Tensor ------------

        # identity tensor
        delta_ij = np.eye(3)                                        # shape: [3,3]

        # calculate trace -> 2 * (Turbulent kinetic energy)
        Rkk[p] = R_ij[0,0] + R_ij[1,1] + R_ij[2,2]
        TKE = 0.5 * Rkk[p] #  -> same formula!                      # shape: scalar
        ###TKE = 0.5 * (urmsf[p]**2 + vrmsf[p]**2 + wrmsf[p]**2)    # shape: scalar

        # omit grid point if reynolds stress tensor trace (2 * TKE) is too small
        if np.abs(Rkk[p]) < tensor_kk_tolerance:
            print(f"Discarded point #{p}")
            continue

        # construct anisotropy tensor
        a_ij = (1.0 / (2*TKE)) * R_ij - (1.0 / 3.0) * delta_ij   # shape: [3,3]

        #------------ eigen-decomposition of the SYMMETRIC TRACE-FREE anisotropy tensor ------------

        # ensure a_ij is trace-free
        # -> calculate trace
        a_kk = a_ij[0,0] + a_ij[1,1] + a_ij[2,2]
        # -> substract the trace
        a_ij[0,0] -= a_kk/3.0
        a_ij[1,1] -= a_kk/3.0
        a_ij[2,2] -= a_kk/3.0

        # Calculate the eigenvalues and eigenvectors
        eigenvalues_a_ij, eigenvectors_a_ij = np.linalg.eigh( a_ij )
        eigenvalues_a_ij_sum = sum(eigenvalues_a_ij)
        assert eigenvalues_a_ij_sum < eigenvalues_tolerance, f"ERROR: The sum of the anisotropy tensor eigenvalues should be 0; in point #{p} the sum is = {eigenvalues_a_ij_sum}"

        # Sort eigenvalues and eigenvectors in decreasing order, so that eigval_1 >= eigval_2 >= eigval_3
        idx = eigenvalues_a_ij.argsort()[::-1]   
        eigenvalues_a_ij  = eigenvalues_a_ij[idx]
        eigenvectors_a_ij = eigenvectors_a_ij[:,idx]
        (lambda1[p], lambda2[p], lambda3[p]) = eigenvalues_a_ij

        if verbose:
            inspected_eigenvalue = (-Rxx[p]+Ryy[p]-3*Ryz[p])/(3*Rxx[p]+6*Ryy[p])
            print(f"\nPoint p = {p}")
            print(f"3rd eigenvalue lambda_2 = {eigenvalues_a_ij[2]}")
            print(f"3rd eigenvector v_2     = {eigenvectors_a_ij[:,2]}")
            print(f"(expected from equations) \lambda_2 = (-R_00+R_11-3R_12)/(3R_00+6R_11) = {inspected_eigenvalue}")
            print(f"(expected from equations) v_2 = (0, -1, 1)$, not normalized")
            print(f"R_11 = {Ryy[p]:.5f}, R_12 = {Ryz[p]:.5f}")

        # Calculate Barycentric map point
        # where eigenvalues_a_ij[0] >= eigenvalues_a_ij[1] >= eigenvalues_a_ij[2] (eigval in decreasing order)
        bar_map_xy =   x1c * (     eigenvalues_a_ij[0] -     eigenvalues_a_ij[1]) \
                     + x2c * ( 2 * eigenvalues_a_ij[1] - 2 * eigenvalues_a_ij[2]) \
                     + x3c * ( 3 * eigenvalues_a_ij[2] + 1)
        xmap1[p]   = bar_map_xy[0]
        xmap2[p]   = bar_map_xy[1]

    return (Rkk, lambda1, lambda2, lambda3, xmap1, xmap2)