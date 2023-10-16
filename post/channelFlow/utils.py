"""
Utils functions for post/channelFlow post-processing scripts
"""

import yaml

import glob as gb
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def get_odt_instantaneous(input_params):
    """
    Get ODT instantaneous fields

    Parameters:
        input_params (dict): ODT input parameters dictionary

    Returns:
        ODT instantaneous coordinates and fields
        uvel, vvel, wvel (np.ndarrays)
    """

    print(f"\nGetting ODT instantaneous data from {odt_statistics_filepath}")

    # --- Get ODT input parameters ---

    dxmin = input_params["dxmin"]
    domainLength = input_params["domainLength"]

    # un-normalize
    dxmin *= domainLength       

    # --- Get ODT data ---

    flist     = sorted(gb.glob('../../data/' + case_name + '/data/data_00000/dmp_*.dat'))
    num_files = len(flist)

    nunif  = int(1/dxmin)        # num. points uniform grid (using smallest grid size)   
    nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)
    
    yu    = np.linspace(-delta,delta,nunif) # uniform grid in y-axis
    uvel = np.zeros([nunif, num_files])
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

    # mirror data -> half channel
    uvel = 0.5*(uvel[:nunif2,:] + np.flipud(uvel[nunif2:,:]))
    vvel = 0.5*(vvel[:nunif2,:] + np.flipud(vvel[nunif2:,:]))
    wvel = 0.5*(wvel[:nunif2,:] + np.flipud(wvel[nunif2:,:]))

    return (uvel, vvel, wvel)



def compute_odt_statistics(odt_statistics_filepath, input_params):
    """
    Compute ODT statistics from multiple .dat files with instantaneous data
    at increasing simulation time

    Parameters:
        odt_statistics_filepath (str): filepath where the computed ODT statistics will be saved
        input_params (dict): ODT input parameters dictionary
        
    Comments: 
        The ODT statistics are saved in a .dat files with the following columns data (included as header in the file): 
        y/delta, y+, u+_mean, v+_mean, w+_mean, u+_rmsf, v+_rmsf, w+_rmsf, <u'u'>+, <v'v'>+, <w'w'>+, <u'v'>+, <u'w'>+, <v'w'>+
    """

    # --- Get ODT input parameters ---
    
    rho   = input_params["rho"]
    kvisc = input_params["kvisc"] # = nu = mu / rho 
    dxmin = input_params["dxmin"]
    delta = input_params["delta"]
    Retau = input_params["Retau"]
    case_name = input_params["caseN"]
    
    # un-normalize
    domainLength = input_params["domainLength"]
    dxmin *= domainLength

    # --- Compute ODT computational data ---

    flist = sorted(gb.glob('../../data/' + case_name + '/data/data_00000/dmp_*.dat'))
    flist_stat = sorted(gb.glob('../../data/' + case_name + '/data/data_00000/statistics/dmp_*_stat.dat'))

    nunif  = int(1/dxmin)        # num. points uniform grid (using smallest grid size)   
    nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)
    nfiles = len(flist)          # num. files of instantaneous data, i.e. num. discrete time instants
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

    logging_files_period = 10000
    ifile_counter = 0
    ifile_total   = len(flist)
    for ifile in flist :

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
        uut    = uu[nunif2:]
        yut    = yu[nunif2:]
        dudy2t = ( (uut[2:]-uut[:-2])/(yut[2:]-yut[:-2]) )**2
        # bottom half of channel
        uub     = np.flip(uu[:nunif2])
        yub     = - np.flip(yu[:nunif2])
        dudy2b  = ( (uub[2:]-uub[:-2])/(yub[2:]-yub[:-2]) )**2
        dudy2m += 0.5*(dudy2b + dudy2t)  # mirror data (symmetric)

        # Logging info
        ifile_counter += 1
        if ifile_counter % logging_files_period == 1:
            print(f"Calculating ODT statistics... {ifile_counter/ifile_total*100:.0f}%")

    # (computed) means
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

    # averaged terms for calculating TKE budgets
    dudy2m /= nfiles 

    # ------------------ (get) Velocity statistics computed during odt simulation ------------------
    # the mean velocities are taken from the last discrete time datafile from odt execution
    # todo: this will lead to error, run-time statistics data is now stored in a subfolder
    data_stat = np.loadtxt(flist_stat[-1])
    yum      = data_stat[:,0]
    um_data_ = data_stat[:,1] 
    data2_   = data_stat[:,2] 
    data3_   = data_stat[:,3]
    
    um_data  = interp1d(yum, um_data_, fill_value='extrapolate')(yu)  
    data2    = interp1d(yum, data2_,   fill_value='extrapolate')(yu)  
    data3    = interp1d(yum, data3_,   fill_value='extrapolate')(yu)  

    um_data  = 0.5*(um_data[:nunif2] + np.flipud(um_data[nunif2:]))  # mirror data (symmetric)
    data2    = 0.5*(data2[:nunif2] + np.flipud(data2[nunif2:]))
    data3    = 0.5*(data3[:nunif2] + np.flipud(data3[nunif2:]))

    # --- y-coordinate, y+ ---
    yu += delta         # domain center is at 0; shift so left side is zero
    yu = yu[:nunif2]    # plotting to domain center
    dudy = (um[1]-um[0])/(yu[1]-yu[0])
    utau = np.sqrt(kvisc * np.abs(dudy) / rho)
    RetauOdt = utau * delta / kvisc
    yuplus = yu * utau/kvisc    # scale y --> y+ (note: utau is close to unity)

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

    odt_data = np.vstack([yu/delta,yuplus,um,vm,wm,urmsf,vrmsf,wrmsf,
                          ufufm,vfvfm,wfwfm,ufvfm,ufwfm,vfwfm,
                          viscous_stress,reynolds_stress,total_stress, # -> stress decomposition
                          vt_u_plus, d_u_plus,                         # -> TKE budgets for u-component
                          um_data, data2, data3]).T            
    np.savetxt(odt_statistics_filepath, odt_data, 
            header= "y/delta,    y+,          u+_mean,     v+_mean,     w+_mean,     u+_rmsf,     v+_rmsf,     w+_rmsf,     "\
                    "<u'u'>+,     <v'v'>+,     <w'w'>+,     <u'v'>+,     <u'w'>+,     <v'w'>+,     " \
                    "tau_viscous, tau_reynolds,tau_total,   " \
                    "vt_u+,       d_u+,        " \
                    "u+_mean_rt,  dumean+_dt,  F_statConv" ,
            fmt='%12.5E')

    print("(ODT) Nominal Retau: ", Retau)
    print("(ODT) Actual  Retau: ", RetauOdt)
    print("(ODT) Nominal utau:  1")
    print("(ODT) Actual  utau: ", utau)
    print("(ODT) kvisc :", kvisc)
    print("(ODT) dumdy|y0 :", dudy, "    du: ", um[1]-um[0], "    dy: ", yu[1]-yu[0], "\n")


def get_odt_statistics(odt_statistics_filepath, input_params):
    """
    Get ODT statistics, previously saved in a .dat file using 'compute_odt_statistics' function

    Parameters:
        odt_statistics_filepath (str): ODT statistics filepath
        input_params (dict): ODT input parameters dictionary

    Returns:
        ODT statistics calculated over statistic time by 'compute_odt_statistics'
        ydelta, yplus, um, urmsf, vrmsf, wrmsf, ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm, viscous_stress, reynolds_stress, total_stress (np.ndarrays)
    """
    # --- Get ODT statistics ---

    print(f"\nGetting ODT data from {odt_statistics_filepath}")
    odt = np.loadtxt(odt_statistics_filepath)

    ydelta = odt[:,0]  # y/delta
    yplus  = odt[:,1]  # y+
    um     = odt[:,2]  # u+_mean

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

    # velocity statistics calculated during odt execution
    
    um_rt = odt[:,19]  # u+_mean_data

    return (ydelta, yplus, um, urmsf, vrmsf, wrmsf, 
            ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm, 
            viscous_stress, reynolds_stress, total_stress, 
            vt_u, d_u,
            um_rt)


def get_dns_statistics(Re_tau, input_params):

    # --- Get DNS input parameters (prescribed) ---
    rho    = 1.0
    delta  = 1.0
    u_tau  = 1.0
    nu     = u_tau * delta / Re_tau
    mu     = rho * nu
    kvisc  = mu / rho # = nu 

    if Re_tau == 590:
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
        filename_dns = f"DNS_statistics/Re{Re_tau}/profiles/Re{Re_tau}.prof"
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
    u_tauDns  = np.sqrt(kvisc * np.abs(dudy_wall) / rho)
    Re_tauDns = u_tauDns * delta / kvisc
    print("\n(DNS) Nominal Re_tau: ", Re_tau)
    print("(DNS) Actual  Re_tau: ", Re_tauDns)
    print("(DNS) Nominal u_tau:  ", u_tau)
    print("(DNS) Actual  u_tau:  ", u_tauDns)
    print("(DNS) kvisc    :", kvisc)
    print("(DNS) dumdy|y0 :", dudy_wall, "    du: ", um[1]-um[0], "    dy: ", ydelta[1]-ydelta[0], "\n")

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
    vt_u_plus_ = vt_u_ * ( delta / u_tau**3 )

    # --- Dissipation budget ---
    # d_i = avg( ( d(u_{i,rmsf})/dy )^2 )
    # -> cannot be calculated with available DNS data :/

    # --- Production budget ---
    dumdy = (um[2:] - um[:-2])/(ydelta[2:] - ydelta[:-2]) # 1st-order central finite difference
    p_u_  = - 2 * ufvfm[1:-1] * dumdy
    p_u_plus_ = p_u_ * ( delta / u_tau**3 ) 

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
        print(f"No valid format found in the first line of {ifile}.")
        time = None
    return time


def get_odt_statistics_during_runtime(input_params, averaging_times):
    """
    Get ODT statistics, previously saved in a .dat file using 'compute_odt_statistics' function

    Parameters:
        odt_statistics_filepath (str): ODT statistics filepath
        input_params (dict): ODT input parameters dictionary
        time_vec (np.array): vector of times at which the statistic is evaluated

    Returns:
        ODT statistics calculated during runtime, output of the simulation at each dumpTime instant
        um (np.ndarrays)
    """

    # --- Get ODT input parameters ---
    
    rho          = input_params["rho"]
    kvisc        = input_params["kvisc"] # = nu = mu / rho 
    domainLength = input_params["domainLength"]
    dxmin        = input_params["dxmin"]
    delta        = input_params["delta"]
    Retau        = input_params["Retau"]
    case_name    = input_params["caseN"]
    dTimeStart   = input_params["dTimeStart"]
    dTimeEnd     = input_params["dTimeEnd"]
    dTimeStep    = input_params["dTimeStep"]
    dxmin       *= domainLength

    # Averaging times and files identification
    dTimeVec = np.arange(dTimeStart, dTimeEnd+1e-3, dTimeStep).round(2)
    averaging_times_num = len(averaging_times)
    averaging_times_idx = []
    for t_idx in range(averaging_times_num):
         averaging_times_idx.append(np.where(dTimeVec==averaging_times[t_idx])[0][0])
    averaging_times_str = [str(idx).zfill(5) for idx in averaging_times_idx]
    if (len(averaging_times_str) != averaging_times_num):
        raise ValueError("Not all averaging_times where found!")

    # --- Compute ODT computational data ---

    flist = ['../../data/' + case_name + '/data/data_00000/statistics/dmp_' + s  + '_stat.dat' for s in averaging_times_str]

    # Num points uniform grid
    nunif  = int(1/dxmin)        # num. points uniform grid (using smallest grid size)   
    nunif2 = int(nunif/2)        # half of num. points (for ploting to domain center, symmetry in y-axis)

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
    um = 0.5 * (um[:nunif2,:]  + np.flipud(um[nunif2:,:]))  # um_

    # ------------------------ Calculate symmetric field ------------------------
    
    # calculate 'symmetric' part of um in all fine grid
    um_symmetric_all  = np.copy(um_all)
    um_symmetric_half = np.copy(um)
    um_symmetric_all[:nunif2,:]  = um_symmetric_half
    um_symmetric_all[nunif2:,:]  = np.flipud(um_symmetric_half)

    # ----------- Calculate indicate - Integral deviation from symmetry -----------

    # CI = Eps_s : Convergence Indicator, from Pirozzoli207-A --- along fine uniform grid
    CI  = np.sqrt( 0.5 * np.sum( (um_all - um_symmetric_all)**2 , axis = 0) ) # rmse
    print("\n(ODT) Convergence Indicator (CI, Esp_s) for each averaging time:")
    for i in range(len(averaging_times)):
        print(f"     At tavg = {averaging_times[i]:.1f} : CI = {CI[i]:.3f}")

    # ------------ scale y to y+ ------------

    # y-coordinates
    ydelta = yu/delta
    yu += delta         # domain center is at 0; shift so left side is zero
    yu = yu[:nunif2]    # plotting to domain center
    # Re_tau of ODT data
    dudy = (um[1,-1]-um[0,-1])/(yu[1]-yu[0])
    utau = np.sqrt(kvisc * np.abs(dudy))
    # scale y --> y+ (note: utau should be unity)
    yuplus     = yu     * utau / kvisc 
    yuplus_all = yu_all * utau / kvisc

    # --- Save ODT computational data ---

    return (ydelta, yuplus, um, CI, yuplus_all, um_all, um_symmetric_all)


def compute_convergence_indicator_odt_tEnd(input_params):
    
    # --- Get ODT input parameters ---

    domainLength = input_params["domainLength"]
    dxmin        = input_params["dxmin"]
    delta        = input_params["delta"]
    case_name    = input_params["caseN"]
    kvisc        = input_params["kvisc"]
    utau         = input_params["utau"]
    # un-normalize
    dxmin *= domainLength
    # uniform fine grid
    nunif  = int(1/dxmin) 
    nunif2 = int(nunif/2)

    # --------------- Get ODT statistics ---------------

    flist_stat     = sorted(gb.glob('../../data/' + case_name + '/data/data_00000/statistics/dmp_*_stat.dat'))
    file_stat_last = flist_stat[-1]
    print("file_stat_last = ", file_stat_last)
    data = np.loadtxt(file_stat_last)

    # uniform fine grid (u.f.g) - '1_posUnif' 
    yu = data[:,0]
    # u_avg statistic runtime-calculated in u.f.g - '2_uvelmean'
    um = data[:,1]
    assert len(yu) == nunif, "ERROR: size error, uniform fine grid should have length 'nunif'"

    # ------------------------ Calculate symmetric field ------------------------

    # calculate 'symmetric' part of um
    um_symmetric_half = 0.5*(um[:nunif2] + np.flipud(um[nunif2:]))  
    um_symmetric = np.copy(um)
    um_symmetric[:nunif2]  = um_symmetric_half
    um_symmetric[nunif2:]  = np.flipud(um_symmetric_half)

    # ----------- Calculate indicate - Integral deviation from symmetry -----------

    # dy necessary for computing the integral of CI - really not necessary for the integral, as fine grid is uniform 
    ### dyu = yu[1:] - yu[:-1] 
    # CI = Eps_s : Convergence Indicator, from Pirozzoli207-A
    CI  = np.sqrt( 0.5 * np.sum( (um - um_symmetric)**2 ) ) # rmse
    print("\n(ODT) Convergence Indicator at tEnd (CI, Esp_s) = ", CI)

    # ------------ scale y to y+ ------------

    # Re_tau of ODT data
    dudy = (um[1]-um[0])/(yu[1]-yu[0])
    utau = np.sqrt(kvisc * np.abs(dudy))
    # scale y --> y+ (note: utau should be unity)
    yuplus = yu * utau/kvisc 

    return (CI, yuplus, um, um_symmetric)



def compute_convergence_indicator_odt_along_time(input_params):
    
    # --- Get ODT input parameters ---

    domainLength = input_params["domainLength"]
    dxmin        = input_params["dxmin"]
    delta        = input_params["delta"]
    case_name    = input_params["caseN"]
    kvisc        = input_params["kvisc"]
    utau         = input_params["utau"]
    # un-normalize
    dxmin *= domainLength
    # uniform fine grid
    nunif  = int(1/dxmin) 
    nunif2 = int(nunif/2)

    # --------------- Get ODT statistics ---------------

    flist_stat     = sorted(gb.glob('../../data/' + case_name + '/data/data_00000/statistics/dmp_*_stat.dat'))
    CI_list = []
    time_list = []

    for ifile in flist_stat: 

        data = np.loadtxt(ifile)
        # u_avg statistic runtime-calculated in u.f.g - '2_uvelmean'
        um = data[:,1]

        # ------------------------ Calculate symmetric field ------------------------

        # calculate 'symmetric' part of um
        um_symmetric_half = 0.5*(um[:nunif2] + np.flipud(um[nunif2:]))  
        um_symmetric = np.copy(um)
        um_symmetric[:nunif2]  = um_symmetric_half
        um_symmetric[nunif2:]  = np.flipud(um_symmetric_half)

        # ----------- Calculate indicate - Integral deviation from symmetry -----------

        # CI = Eps_s : Convergence Indicator, from Pirozzoli207-A
        CI = np.sqrt( 0.5 * np.sum( (um - um_symmetric)**2 ) ) # rmse
        CI_list.append(CI)

        time_list.append(get_time(ifile))

    return (time_list, CI_list)






