# TODO: add description
#
# Usage
# python3 anisotropy_tensor_odt_convergence.py [case_name] [reynolds_number] [delta_time_stats]

import yaml
import sys
import os
import numpy as np
import pandas as pd

from utils import *
from ChannelVisualizer import ChannelVisualizer

plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 14 )
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb} \usepackage{color}")

#--------------------------------------------------------------------------------------------

verbose = False

# --- Define parameters ---
tensor_kk_tolerance   = 1.0e-8;	# [-]
eigenvalues_tolerance = 1.0e-8;	# [-]
nbins = 50;			            # [-]

# --- Get CASE parameters ---

try :
    caseN = sys.argv[1]
    Retau = int(sys.argv[2])
    delta_aver_time = int(sys.argv[3])
except :
    raise ValueError("Include the case name in the call")

if not os.path.exists("../../data/"+caseN+"/post") :
    os.mkdir("../../data/"+caseN+"/post")

# --- Location of Barycentric map corners ---
x1c = np.array( [ 1.0 , 0.0 ] )
x2c = np.array( [ 0.0 , 0.0 ] )
x3c = np.array( [ 0.5 , np.sqrt(3.0)/2.0 ] )

# --- Animation frames (gif) ---
visualizer      = ChannelVisualizer(caseN)
frames_eig_post = [];   frames_bar_post = []
frames_eig_rt   = [];   frames_bar_rt   = []

# --- Get ODT input parameters ---

odtInputDataFilepath  = "../../data/" + caseN + "/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc = yml["params"]["kvisc0"] # kvisc = nu = mu / rho
rho   = yml["params"]["rho0"]
dxmin = yml["params"]["dxmin"]
nunif = yml["params"]["nunif"]
domainLength = yml["params"]["domainLength"] 
delta = domainLength * 0.5
utau  = 1.0
dTimeStart  = yml["dumpTimesGen"]["dTimeStart"]
#dTimeEnd   = yml["dumpTimesGen"]["dTimeEnd"]
dTimeEnd    = get_provisional_tEnd(caseN)  # tEnd, valid even while running odt, instead of yml["dumpTimesGen"]["dTimeEnd"]
dTimeStep   = yml["dumpTimesGen"]["dTimeStep"]
inputParams = {"kvisc":kvisc, "rho":rho, "dxmin": dxmin, "nunif": nunif, "domainLength" : domainLength, "delta": delta, "Retau": Retau, "caseN": caseN, "utau": utau, 'dTimeStart':dTimeStart, 'dTimeEnd':dTimeEnd, 'dTimeStep':dTimeStep} 

#------------ Averaging times ---------------

averaging_times = np.arange(dTimeStart, dTimeEnd+0.1, delta_aver_time)
# remove first time, as only 1 file is used to calculate the statistics, therefore 
# they are just instantaneous, and make the reynolds stress tensor not satisfy realizability conditions
averaging_times = averaging_times[1:] 
num_aver_times  = len(averaging_times)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ----------------------- Post-processed statistics ----------------------- 
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

for avg_time in averaging_times:

    #------------ Compute statistics until avg_time is reached ---------------
    
    if verbose:
        print(f"\n\n------------ Averaging Time = {avg_time:.2f} ---------------")
    (ydelta, _, _, urmsf, vrmsf, wrmsf, R11, R22, R33, R12, R13, R23) \
        = compute_odt_statistics_at_chosen_time(inputParams, avg_time)

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
    
    if verbose:
        if cond1 and cond2 and cond3:
            print("\nCONGRATULATIONS, the reynolds stress tensor satisfies REALIZABILITY CONDITIONS.")
        else:
            print(f"\nREALIZABILITY ERROR in AVERAGING TIME t_avg = {avg_time:.2f}")
            print("\nERROR: The reynolds stress tensor does not satisfy REALIZABILITY CONDITIONS")
            print("\nERROR: Cond 1 is ", cond1," - Cond 2 is ", cond2, "- Cond 3 is ", cond3)
            #print("EXECUTION TERMINATED")
            #exit(0)

    #-----------------------------------------------------------------------------------------
    #           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
    #-----------------------------------------------------------------------------------------

    # Computed for each point of the grid
    # If the trace of the reynolds stress tensor (2 * TKE) is too small, the corresponding 
    # datapoint is omitted, because the anisotropy tensor would -> infinity, as its equation
    # contains the multiplier ( 1 / (2*TKE) )

    # initialize quantities
    eigenvalues   = np.zeros([num_points, 3])
    bar_map_x     = np.zeros(num_points)
    bar_map_y     = np.zeros(num_points)
    bar_map_color = np.zeros(num_points)

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
        
        # Store quantities
        eigenvalues[p,:] = eigenvalues_a_ij[:]
        bar_map_x[p]     = bar_map_xy[0]
        bar_map_y[p]     = bar_map_xy[1]
        bar_map_color[p] = ydelta[p]

    # ---------------------- Build frame at averaging time ---------------------- 
    
    frames_eig_post = visualizer.build_anisotropy_tensor_eigenvalues_frame(frames_eig_post, ydelta, eigenvalues, avg_time)
    frames_bar_post = visualizer.build_anisotropy_tensor_barycentric_map_frame(frames_bar_post, bar_map_x, bar_map_y, bar_map_color, avg_time)


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------- Runtime statistics --------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

(ydelta_rt, _,_,_, _,_,_, _,_,_, Rxx_rt, Ryy_rt, Rzz_rt, Rxy_rt, Rxz_rt, Ryz_rt) = get_odt_statistics_rt_at_chosen_averaging_times(inputParams, averaging_times)
(Rkk_rt, lambda0_rt, lambda1_rt, lambda2_rt, xmap1_rt, xmap2_rt) = compute_reynolds_stress_dof(Rxx_rt, Ryy_rt, Rzz_rt, Rxy_rt, Rxz_rt, Ryz_rt)

for i in range(num_aver_times): 
    eigenvalues_rt = np.array([lambda0_rt[:,i], lambda1_rt[:,i], lambda2_rt[:,i]]).transpose()
    frames_eig_rt  = visualizer.build_anisotropy_tensor_eigenvalues_frame(frames_eig_rt, ydelta_rt[:,i], eigenvalues_rt, averaging_times[i])
    frames_bar_rt  = visualizer.build_anisotropy_tensor_barycentric_map_frame(frames_bar_rt, xmap1_rt[:,i], xmap2_rt[:,i], ydelta_rt[:,i], averaging_times[i])


# -------------------------------------------------------------------------
# ------------------ Create the animation from the frames -----------------
# -------------------------------------------------------------------------

# post-processed statistics
filename = f"../../data/{caseN}/post/anisotropy_tensor_eigenvalues_odt_convergence_post.gif"
print(f"\nMAKING GIF EIGENVALUES OF ANISOTROPY TENSOR for POST-PROCESSING calculations ALONG AVG. TIME in {filename}" )
frames_eig_post[0].save(filename, save_all=True, append_images=frames_eig_post[1:], duration=100, loop=0)
filename = f"../../data/{caseN}/post/anisotropy_tensor_barycentric_map_odt_convergence_post.gif"
print(f"\nMAKING GIF OF BARYCENTRIC MAP OF ANISOTROPY TENSOR for POST-PROCESSING calculations ALONG AVG. TIME in {filename}" )
frames_bar_post[0].save(filename, save_all=True, append_images=frames_bar_post[1:], duration=100, loop=0)

# runtime statistics
filename = f"../../data/{caseN}/post/anisotropy_tensor_eigenvalues_odt_convergence_rt.gif"
print(f"\nMAKING GIF EIGENVALUES OF ANISOTROPY TENSOR for RUNTIME calculations ALONG AVG. TIME in {filename}" )
frames_eig_rt[0].save(filename, save_all=True, append_images=frames_eig_rt[1:], duration=100, loop=0)
print(f"\nMAKING GIF OF BARYCENTRIC MAP OF ANISOTROPY TENSOR for RUNTIME calculations ALONG AVG. TIME in {filename}" )
filename = f"../../data/{caseN}/post/anisotropy_tensor_barycentric_map_odt_convergence_rt.gif"
frames_bar_rt[0].save(filename, save_all=True, append_images=frames_bar_rt[1:], duration=100, loop=0)

