# TODO: add description
#
# Usage
# python3 anisotropy_tensor_odt_convergence.py [case_name] [reynolds_number] [delta_time_stats]

import yaml
import sys
import os
import math

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
#from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

from PIL import Image
from utils import *
plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 14 )
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb} \usepackage{color}")

#--------------------------------------------------------------------------------------------

# --- Define parameters ---
tensor_kk_tolerance   = 1.0e-8;	# [-]
eigenvalues_tolerance = 1.0e-8;	# [-]
nbins = 50;			            # [-]

# --- Location of Barycentric map corners ---
x1c = np.array( [ 1.0 , 0.0 ] )
x2c = np.array( [ 0.0 , 0.0 ] )
x3c = np.array( [ 0.5 , math.sqrt(3.0)/2.0 ] )

# --- Animation frames (gif) ---
frames = []

# --- Get CASE parameters ---

try :
    caseN = sys.argv[1]
    Retau = int(sys.argv[2])
    delta_aver_time = int(sys.argv[3])
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
delta = domainLength * 0.5
utau  = 1.0
tEnd   = yml["params"]["tEnd"]              # = 150.0
tStart = yml["dumpTimesGen"]["dTimeStart"]  # = 50.0
inputParams = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "caseN": caseN, "utau": utau} 


#------------ Averaging times ---------------

averaging_times = np.arange(tStart, tEnd+0.1, delta_aver_time)
num_aver_times  = len(averaging_times)

for avg_time in averaging_times:

    #------------ Compute statistics until avg_time is reached ---------------
    
    print(f"\n\n------------ Averaging Time = {avg_time:.2f} ---------------")
    (ydelta, _, _, urmsf, vrmsf, wrmsf, R11, R22, R33, R12, R13, R23) \
        = compute_odt_statistics_at_chosen_time(inputParams,avg_time)

    #------------ Reynolds stress tensor ---------------

    # Build tensor (for each grid point)
    num_points  = len(R11)
    R_ij        = np.zeros([num_points, 3, 3])
    R_ij[:,0,0] = R11
    R_ij[:,0,1] = R12
    R_ij[:,0,2] = R13
    R_ij[:,1,0] = R12
    R_ij[:,1,1] = R22
    R_ij[:,1,2] = R23
    R_ij[:,2,0] = R13
    R_ij[:,2,1] = R23
    R_ij[:,2,2] = R33

    #------------ Realizability conditions ---------------

    # help: .all() ensures the condition is satisfied in all grid points

    # COND 1: Rii >= 0, for i = 1,2,3

    cond1_1 = ( R11 >= 0 ).all()    # i = 1
    cond1_2 = ( R22 >= 0 ).all()    # i = 2
    cond1_3 = ( R33 >= 0 ).all()    # i = 3
    cond1   = cond1_1 and cond1_2 and cond1_3

    # COND 2: Rij^2 <= Rii*Rjj, for i!=j

    cond2_1 = ( R12**2 <= R11 * R22 ).all()     # i = 1, j = 2
    cond2_2 = ( R13**2 <= R11 * R33 ).all()     # i = 1, j = 3
    cond2_3 = ( R23**2 <= R22 * R33 ).all()     # i = 1, j = 3
    cond2   = cond2_1 and cond2_2 and cond2_3

    # COND 3: det(Rij) >= 0

    detR  = np.linalg.det(R_ij)    # length(detR) = num_points
    cond3 = ( detR >= 0 ).all()

    if cond1 and cond2 and cond3:
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
        Rkk = R11[p] + R22[p] + R33[p]                              # shape: scalar
        ###TKE = 0.5 * Rkk -> WRONG FORMULA!                        # shape: scalar
        TKE = 0.5 * (urmsf[p]**2 + vrmsf[p]**2 + wrmsf[p]**2)       # shape: scalar

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
        #print(f"EigVal[{p}] = ", eigenvalues_a_ij)

        # Calculate Barycentric map point
        # where eigenvalues_a_ij[0] >= eigenvalues_a_ij[1] >= eigenvalues_a_ij[2] (eigval in decreasing order)
        bar_map_xy = x1c * (     eigenvalues_a_ij[0] -     eigenvalues_a_ij[1])  \
                + x2c * ( 2 * eigenvalues_a_ij[1] - 2 * eigenvalues_a_ij[2]) \
                + x3c * ( 3 * eigenvalues_a_ij[2] + 1)
        bar_map_x.append(bar_map_xy[0])
        bar_map_y.append(bar_map_xy[1])
        bar_map_color.append(ydelta[p])

    # ---------------------- Plot Barycentric Map ---------------------- 

    plt.figure()

    # Plot markers Barycentric map
    #cmap = cm.get_cmap( 'Greys' ) ## deprecated from matplotlib 3.7
    cmap  = matplotlib.colormaps['Greys']
    norm  = colors.Normalize(vmin = 0, vmax = 1.0)

    # Plot data into the barycentric map
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
    cbar.set_label( r'$y/\delta$' )
    plt.title(f"averaging time = {avg_time:.1f}")
    ###plt.clim( 0.0, 20.0 )

    # ------ save figure ------
    #filename = f"../../data/{caseN}/post/anisotropy_tensor_barycentric_map_odt_avgTime_{avg_time:.0f}.jpg"
    #print(f"\nMAKING PLOT OF BARYCENTRIC MAP OF ANISOTROPY TENSOR from ODT data at Averaging Time = {avg_time:.2f}, in filename: {filename}" )
    #plt.savefig(filename, dpi=600)

    # ------ gif frame by pillow ---------
    # Save the current figure as an image frame
    fig = plt.gcf()
    fig.canvas.draw()
    img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    frames.append(img)

    plt.close()

# ---------- Create the animation from the frames ----------

filename = f"../../data/{caseN}/post/anisotropy_tensor_barycentric_map_odt_convergence.gif"
frames[0].save(filename, save_all=True, append_images=frames[1:], duration=100, loop=0)

