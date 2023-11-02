# Description: 
# 1. Compute ODT reynolds stresses, anisotropy tensor, TKE
# 2. Calculate eigenvectors and eigenvalues of anisotropy tensor
# 3. Project the anisotropy eigenvalues into a baricentric map by linear mapping
# 4. Check: is the anisotropy tensor / Reynolds stresses satisfying the realizability conditions?

# Usage
# python3 anisotropy_tensor_odt_vs_dns.py [case_name] [reynolds_number]

# Arguments:
# case_name (str): Name of the case
# reynolds_number (int): reynolds number of the odt case, to get comparable dns result.

# Example Usage:
# python3 anisotropy_tensor.py channel180 180

# Comments:
# Values are in wall units (y+, u+) for both ODT and DNS results,
# Scaling is done in the input file (not explicitly here).

import yaml
import sys
import os
import math

import matplotlib
import numpy as np
import pandas as pd

from utils import *

plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 14 )
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb} \usepackage{color}")

#--------------------------------------------------------------------------------------------

# --- Define parameters ---
tensor_kk_tolerance   = 1.0e-8;	# [-]
eigenvalues_tolerance = 1.0e-8;	# [-]
nbins = 50;			            # [-]
simulation_list = ["odt"]#, "dns"]

# --- Location of Barycentric map corners ---
x1c = np.array( [ 1.0 , 0.0 ] )
x2c = np.array( [ 0.0 , 0.0 ] )
x3c = np.array( [ 0.5 , math.sqrt(3.0)/2.0 ] )

# --- Get CASE parameters ---

try :
    caseN = sys.argv[1]
    Retau = int(sys.argv[2])
except :
    raise ValueError("Include the case name in the call")

if not os.path.exists("../../data/"+caseN+"/post") :
    os.mkdir("../../data/"+caseN+"/post")

# --- Get ODT input parameters ---

odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho   = yml["params"]["rho0"]
dxmin = yml["params"]["dxmin"]
domainLength = yml["params"]["domainLength"] 
tEnd  = yml["dumpTimesGen"]["dTimeEnd"]
delta = domainLength * 0.5
utau  = 1.0
inputParams = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "caseN": caseN, "utau": utau} 

#------------ Get ODT data ---------------

# post-processed statistics
odtStatisticsFilepath = "../../data/" + caseN + "/post/ODTstat.dat"
compute_odt_statistics(odtStatisticsFilepath, inputParams)
(ydelta_odt, yplus_odt, um_odt, vm_odt, wm_odt, urmsf_odt, vrmsf_odt, wrmsf_odt, ufufm_odt, vfvfm_odt, wfwfm_odt, ufvfm_odt, ufwfm_odt, vfwfm_odt, viscous_stress_odt, reynolds_stress_odt, total_stress_odt, vt_u_plus_odt, d_u_plus_odt) \
    = get_odt_statistics(odtStatisticsFilepath, inputParams)
# calculated-at-runtime statistics
(_, _,_,_, _,_,_, _,_,_, _,_,_,_,_,_, \
    lambda0_odt_rt, lambda1_odt_rt, lambda2_odt_rt, xmap1_odt_rt, xmap2_odt_rt) \
    = get_odt_statistics_rt(inputParams)

#------------ Get DNS statistics ---------------

(ydelta_dns, yplus_dns, um_dns, urmsf_dns, vrmsf_dns, wrmsf_dns, ufufm_dns, vfvfm_dns, wfwfm_dns, ufvfm_dns, ufwfm_dns, vfwfm_dns, viscous_stress_dns, reynolds_stress_dns, total_stress_dns, vt_u_plus_dns,               p_u_plus_dns) \
    = get_dns_statistics(Retau, inputParams)

for sim in simulation_list:

    # ----------- Get data of interest -----------------
    if sim == "odt":
        print("\n----------------- ODT simulation data -----------------")
        R00   = ufufm_odt
        R01   = ufvfm_odt
        R02   = ufwfm_odt
        R11   = vfvfm_odt
        R12   = vfwfm_odt
        R22   = wfwfm_odt
        urmsf = urmsf_odt 
        vrmsf = vrmsf_odt 
        wrmsf = wrmsf_odt 
        yplus = yplus_odt
        yplus_max = np.max(yplus)
    elif sim == "dns":
        print("\n----------------- DNS simulation data -----------------")
        R00   = ufufm_dns
        R01   = ufvfm_dns
        R02   = ufwfm_dns
        R11   = vfvfm_dns
        R12   = vfwfm_dns
        R22   = wfwfm_dns
        urmsf = urmsf_dns 
        vrmsf = vrmsf_dns 
        wrmsf = wrmsf_dns 
        yplus = yplus_dns
        yplus_max = np.max(yplus)
    else:
        raise ValueError("'simulation_list' can only take values 'odt' or 'dns', but element '{sim}' was chosen.")

    #------------ Reynolds stress tensor ---------------

    # Build tensor (for each grid point)
    num_points  = len(R11)
    R_ij        = np.zeros([num_points, 3, 3])
    R_ij[:,0,0] = R00
    R_ij[:,0,1] = R01
    R_ij[:,0,2] = R02
    R_ij[:,1,0] = R01
    R_ij[:,1,1] = R11
    R_ij[:,1,2] = R12
    R_ij[:,2,0] = R02
    R_ij[:,2,1] = R12
    R_ij[:,2,2] = R22

    #------------ Realizability conditions ---------------

    # help: .all() ensures the condition is satisfied in all grid points

    # COND 1: Rii >= 0, for i = 1,2,3

    cond0_0 = ( R00 >= 0 ).all()    # i = 1
    cond0_1 = ( R11 >= 0 ).all()    # i = 2
    cond0_2 = ( R22 >= 0 ).all()    # i = 3
    cond0   = cond0_0 and cond0_1 and cond0_2

    # COND 2: Rij^2 <= Rii*Rjj, for i!=j

    cond1_0 = ( R01**2 <= R00 * R11 ).all()     # i = 0, j = 1
    cond1_1 = ( R02**2 <= R00 * R22 ).all()     # i = 0, j = 2
    cond1_2 = ( R12**2 <= R11 * R22 ).all()     # i = 1, j = 2
    cond1   = cond1_0 and cond1_1 and cond1_2

    # COND 3: det(Rij) >= 0

    detR  = np.linalg.det(R_ij)    # length(detR) = num_points
    cond2 = ( detR >= 0 ).all()

    if cond0 and cond1 and cond2:
        print("\nCONGRATULATIONS, the reynolds stress tensor satisfies REALIZABILITY CONDITIONS.")
    else:
        raise Exception("The reynolds stress tensor does not satisfy REALIZABILITY CONDITIONS")

    #-----------------------------------------------------------------------------------------
    #           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
    #-----------------------------------------------------------------------------------------

    # Computed for each point of the grid
    # If the trace of the reynolds stress tensor (2 * TKE) is too small, the corresponding 
    # datapoint is omitted, because the anisotropy tensor would -> infinity, as its equation
    # contains the multiplier ( 1 / (2*TKE) )

    bar_map_x = []; bar_map_y = []
    bar_map_color = []

    for p in range(num_points):

        #------------ Anisotropy Tensor ------------

        # identity tensor
        delta_ij = np.eye(3)                                        # shape: [3,3]

        # calculate trace -> 2 * (Turbulent kinetic energy)
        Rkk = R00[p] + R11[p] + R22[p]                              # shape: scalar
        TKE = 0.5 * Rkk #  -> same formula!                         # shape: scalar
        ###TKE = 0.5 * (urmsf[p]**2 + vrmsf[p]**2 + wrmsf[p]**2)    # shape: scalar

        # omit grid point if reynolds stress tensor trace (2 * TKE) is too small
        if np.abs(Rkk) < tensor_kk_tolerance:
            print(f"Discarded point #{p}")
            continue

        # construct anisotropy tensor
        a_ij = (1.0 / (2*TKE)) * R_ij[p,:,:] - (1.0 / 3.0) * delta_ij   # shape: [3,3]

        #------------ eigen-decomposition of the SYMMETRIC TRACE-FREE anisotropy tensor ------------

        # ensure a_ij is trace-free
        # -> calculate trace
        a_kk = a_ij[0,0] + a_ij[1,1] + a_ij[2,2]                    # shape [num_points]
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
        print(f"\nPoint p = {p}")
        print(f"3rd eigenvalue lambda_2 = {eigenvalues_a_ij[2]}")
        print(f"3rd eigenvector v_2     = {eigenvectors_a_ij[:,2]}")
        inspected_eigenvalue = (-R00[p]+R11[p]-3*R12[p])/(3*R00[p]+6*R11[p])
        print(f"(expected from equations) \lambda_2 = (-R_00+R_11-3R_12)/(3R_00+6R_11) = {inspected_eigenvalue}")
        print(f"(expected from equations) v_2 = (0, -1, 1)$, not normalized")
        print(f"R_11 = {R11[p]:.5f}, R_12 = {R12[p]:.5f}")

        # Calculate Barycentric map point
        # where eigenvalues_a_ij[0] >= eigenvalues_a_ij[1] >= eigenvalues_a_ij[2] (eigval in decreasing order)
        bar_map_xy = x1c * (     eigenvalues_a_ij[0] -     eigenvalues_a_ij[1])  \
                   + x2c * ( 2 * eigenvalues_a_ij[1] - 2 * eigenvalues_a_ij[2]) \
                   + x3c * ( 3 * eigenvalues_a_ij[2] + 1)
        bar_map_x.append(bar_map_xy[0])
        bar_map_y.append(bar_map_xy[1])
        bar_map_color.append(yplus[p])

    # ---------------------- Plot Barycentric Map (post-processing & runtime calculations)---------------------- 

    plt.figure()

    # Plot markers Barycentric map
    cmap   = matplotlib.colormaps["Greys"] # idem. as cm.get_cmap( 'Greys' ), deprecated in Matplotlib 3.7 
    norm   = matplotlib.colors.Normalize(vmin = 0, vmax = yplus_max)

    # Plot data into the barycentric map
    ##plt.scatter( bar_x, bar_y, color='red', zorder = 4, marker = 'o', s = 85, edgecolor = 'black', linewidth = 0.8 )
    plt.scatter( bar_map_x, bar_map_y, c = bar_map_color, cmap = cmap, norm=norm, zorder = 3, marker = 'o', s = 85, edgecolor = 'black', linewidth = 0.8 )

    # Plot barycentric map lines
    plt.plot( [x1c[0], x2c[0]],[x1c[1], x2c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
    plt.plot( [x2c[0], x3c[0]],[x2c[1], x3c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
    plt.plot( [x3c[0], x1c[0]],[x3c[1], x1c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )

    # Configure plot
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.axis( 'off' )
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.text( 1.0047, -0.025, r'$\textbf{x}_{1_{c}}$' )
    plt.text( -0.037, -0.025, r'$\textbf{x}_{2_{c}}$' )
    plt.text( 0.4850, 0.9000, r'$\textbf{x}_{3_{c}}$' )
    cbar = plt.colorbar()
    cbar.set_label( r'$y^{+}$' )
    plt.title(f"averaging time = {tEnd:.1f}")

    # save figure
    filename = f"../../data/{caseN}/post/anisotropy_tensor_barycentric_map_{sim}_postproc.jpg"
    print(f"\nMAKING PLOT OF BARYCENTRIC MAP OF ANISOTROPY TENSOR from {sim} data for POST-PROCESSING calculations in {filename}" )
    plt.savefig(filename, dpi=600)


    # ---------------------- Plot Barycentric Map (runtime calculations)---------------------- 

    plt.figure()

    # Plot markers Barycentric map
    cmap   = matplotlib.colormaps["Greys"] # idem. as cm.get_cmap( 'Greys' ), deprecated in Matplotlib 3.7 
    norm   = matplotlib.colors.Normalize(vmin = 0, vmax = yplus_max)

    # Plot data into the barycentric map
    ##plt.scatter( bar_x, bar_y, color='red', zorder = 4, marker = 'o', s = 85, edgecolor = 'black', linewidth = 0.8 )
    plt.scatter( xmap1_odt_rt, xmap2_odt_rt, c = bar_map_color, cmap = cmap, norm=norm, zorder = 3, marker = 'o', s = 85, edgecolor = 'black', linewidth = 0.8 )

    # Plot barycentric map lines
    plt.plot( [x1c[0], x2c[0]],[x1c[1], x2c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
    plt.plot( [x2c[0], x3c[0]],[x2c[1], x3c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
    plt.plot( [x3c[0], x1c[0]],[x3c[1], x1c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )

    # Configure plot
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.axis( 'off' )
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.text( 1.0047, -0.025, r'$\textbf{x}_{1_{c}}$' )
    plt.text( -0.037, -0.025, r'$\textbf{x}_{2_{c}}$' )
    plt.text( 0.4850, 0.9000, r'$\textbf{x}_{3_{c}}$' )
    cbar = plt.colorbar()
    cbar.set_label( r'$y^{+}$' )
    plt.title(f"averaging time = {tEnd:.1f}")

    # save figure
    filename = f"../../data/{caseN}/post/anisotropy_tensor_barycentric_map_{sim}_runtime.jpg"
    print(f"\nMAKING PLOT OF BARYCENTRIC MAP OF ANISOTROPY TENSOR from {sim} data for RUNTIME calculations in {filename}" )
    plt.savefig(filename, dpi=600)