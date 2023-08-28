import yaml

import glob as gb
import numpy as np
from scipy.interpolate import interp1d


# Utils functions for post/channelFlow post-processing scripts

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

    kvisc = input_params["kvisc"]
    dxmin = input_params["dxmin"]
    delta = input_params["delta"]
    Retau = input_params["Retau"]
    case_name = input_params["caseN"]

    # --- Compute ODT computational data ---

    flist = sorted(gb.glob('../../data/' + case_name + '/data/data_00000/dmp_*.dat'))

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

    logging_files_period = 1000
    ifile_counter = 0
    ifile_total   = len(flist)
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

        # Logging info
        ifile_counter += 1
        if ifile_counter % logging_files_period == 1:
            print(f"Calculating ODT statistics... {ifile_counter/ifile_total*100:.0f}%")

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

    # --- Save ODT computational data ---

    odt_data = np.vstack([yu/delta,yuplus,um,vm,wm,urmsf,vrmsf,wrmsf,ufufm,vfvfm,wfwfm,ufvfm,ufwfm,vfwfm]).T
    np.savetxt(odt_statistics_filepath, odt_data, 
            header="y/delta,    y+,          u+_mean,     v+_mean,     w+_mean,     u+_rmsf,     v+_rmsf,     w+_rmsf,     "\
                    "<u'u'>+,     <v'v'>+,     <w'w'>+,     <u'v'>+,     <u'w'>+,     <v'w'>+ ",
            fmt='%12.5E')

    print("Nominal Retau: ", Retau)
    print("Actual  Retau: ", RetauOdt)
    print("")


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

    # --- Get ODT input parameters ---

    kvisc = input_params["kvisc"]
    rho   = input_params["rho"]

    # --- Get ODT statistics ---

    print(f"Getting ODT data from {odt_statistics_filepath}")
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

    # Stress decomposition: Viscous, Reynolds and Total stress
    dumdy = (um[1:] - um[:-1])/(ydelta[1:] - ydelta[:-1])
    viscous_stress  = kvisc * rho * dumdy
    reynolds_stress = - rho * ufvfm[:-1]
    total_stress    = viscous_stress + reynolds_stress

    return (ydelta, yplus, um, urmsf, vrmsf, wrmsf, ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm, viscous_stress, reynolds_stress, total_stress)


def get_dns_statistics(reynolds_number, input_params):

    # --- Get ODT input parameters ---

    kvisc = input_params["kvisc"]
    rho   = input_params["rho"]

    # --- Get DNS input parameters ---

    if reynolds_number == 590:
        filename_dns = "DNS_statistics/Re590/dnsChannel_Re590_means.dat"
        print(f"Getting DNS-means data from {filename_dns}")
        # Dataset columns
        # 0    | 1    | 2     | 3         | 4     | 5         | 6
        # y/h  | y+   | Umean | dUmean/dy | Wmean | dWmean/dy | Pmean

        dns_means = np.loadtxt(filename_dns)
        ydelta = dns_means[:,0] * 2 # y/delta = (y/h)*2
        yplus  = dns_means[:,1]
        um = dns_means[:,2] # Umean normalized by U_tau (= u+_mean)

        filename_dns = "DNS_statistics/Re590/dnsChannel_Re590_reynolds_stress.dat"
        print(f"Getting DNS-reynolds data from {filename_dns}")
        # Dataset columns
        # 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7
        # y    | y+   | R_uu | R_vv | R_ww | R_uv | R_uw | R_vw
        dns_reynolds_stress = np.loadtxt(filename_dns)
        Rxx = dns_reynolds_stress[:,2] # R_xx+, normalized by U_tau^2
        Ryy = dns_reynolds_stress[:,3]
        Rzz = dns_reynolds_stress[:,4]
        Rxy = dns_reynolds_stress[:,5] # R_xy+
        Rxz = dns_reynolds_stress[:,6] # R_xz+
        Ryz = dns_reynolds_stress[:,7] # R_yz+

        urmsf = np.sqrt(Rxx_dns) # = sqrt(mean(u'u')), normalized by U_tau^2
        vrmsf = np.sqrt(Ryy_dns)
        wrmsf = np.sqrt(Rzz_dns)

    else:
        filename_dns = f"DNS_statistics/Re{reynolds_number}/profiles/Re{reynolds_number}.prof"
        print(f"Getting DNS-reynolds data from {filename_dns}")
        # Dataset columns:
        # 0    | 1    | 2    | 3    | 4    | 5    | 6      | 7      | 8      | 9      | 10   | 11   | 12   | 13   | 14   | 15     | 16   
        # y/h  | y+   | U+   | u'+  | v'+  | w'+  | -Om_z+ | om_x'+ | om_y'+ | om_z'+ | uv'+ | uw'+ | vw'+ | pr'+ | ps'+ | psto'+ | p'    
        dns   = np.loadtxt(filename_dns,comments="%")
        ydelta = dns[:,0]*2 # y/delta = (y/h)*2
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

    # --- Stress decomposition: Viscous, Reynolds and Total stress ---
    dumdy = (um[1:] - um[:-1])/(ydelta[1:] - ydelta[:-1])
    viscous_stress  = kvisc * rho * dumdy
    reynolds_stress = - rho * ufvfm[:-1]
    total_stress    = viscous_stress + reynolds_stress

    return (ydelta, yplus, um, urmsf, vrmsf, wrmsf, ufufm, vfvfm, wfwfm, ufvfm, ufwfm, vfwfm,
            viscous_stress, reynolds_stress, total_stress)


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