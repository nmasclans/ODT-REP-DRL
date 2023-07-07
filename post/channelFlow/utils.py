import numpy as np


# Utils functions for post/channelFlow post-processing scripts

def get_dns_data(reynolds_number):
    
    if reynolds_number == 590:
        filename_dns = "DNS_statistics/Re590/dnsChannel_Re590_means.dat"
        print(f"Getting DNS-means data from {filename_dns}")
        # Dataset columns
        # 0    | 1    | 2     | 3         | 4     | 5         | 6
        # y    | y+   | Umean | dUmean/dy | Wmean | dWmean/dy | Pmean

        dns_means = np.loadtxt(filename_dns)
        y_dns = dns_means[:,1] # y+
        u_dns = dns_means[:,2] # Umean normalized by U_tau (= u+_mean)

        filename_dns = "DNS_statistics/Re590/dnsChannel_Re590_reynolds_stress.dat"
        print(f"Getting DNS-reynolds data from {filename_dns}")
        # Dataset columns
        # 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7
        # y    | y+   | R_uu | R_vv | R_ww | R_uv | R_uw | R_vw
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
        filename_dns = f"DNS_statistics/Re{reynolds_number}/profiles/Re{reynolds_number}.prof"
        print(f"Getting DNS-reynolds data from {filename_dns}")
        # Dataset columns:
        # 0    | 1    | 2    | 3    | 4    | 5    | 6      | 7      | 8      | 9      | 10   | 11   | 12   | 13   | 14   | 15     | 16   
        # y/h  | y+   | U+   | u'+  | v'+  | w'+  | -Om_z+ | om_x'+ | om_y'+ | om_z'+ | uv'+ | uw'+ | vw'+ | pr'+ | ps'+ | psto'+ | p'    
        dns   = np.loadtxt(filename_dns,comments="%")
        ydelta_dns = dns[:,0] # y/delta
        yplus_dns  = dns[:,1] # y+
        um_dns     = dns[:,2] # u+_mean 

        urmsf_dns  = dns[:,3] # named as u'+, normalized by U_tau
        vrmsf_dns  = dns[:,4] # named as v'+, normalized by U_tau
        wrmsf_dns  = dns[:,5] # named as w'+, normalized by U_tau
        
        ufufm_dns  = urmsf_dns**2    
        vfvfm_dns  = vrmsf_dns**2
        wfwfm_dns  = wrmsf_dns**2
        ufvfm_dns  = dns[:,10]
        ufwfm_dns  = dns[:,11]
        vfwfm_dns  = dns[:,12]

    return (ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns)


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