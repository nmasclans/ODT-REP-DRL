"""
Script to calculate total kinetic energy of the system in J/kg
-> Turbulence is developed when kinetic energy converges
-> Valid to inspect J/kg, because density rho is constant along the channel, 
   and adaptative grid is interpolated to uniform grid
"""

import os
import sys
import yaml

import numpy as np
import glob as gb
import matplotlib.pyplot as plt

from utils import *

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 18)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')

# ------------ Input parameters ---------------

try :
    caseN     = sys.argv[1]
    rlzN      = int(sys.argv[2]); rlzStr = f"{rlzN:05d}"
    print(f"Script parameters: \n- Case name: {caseN} \n- Realization Number: {rlzN}")
except :
    raise ValueError("Missing call arguments, should be: <case_name> <realization_number>")

# ------------ ODT input parameters ------------

odtInputDataFilepath  = f"../../data/{caseN}/input/input.yaml"
with open(odtInputDataFilepath) as ifile :
    yml = yaml.load(ifile, Loader=yaml.FullLoader)
kvisc        = yml["params"]["kvisc0"]  # kvisc = nu = mu / rho
nunif        = yml["params"]["nunif"]
rho          = yml["params"]["rho0"]    # ct!
domainLength = yml["params"]["domainLength"] 
delta        = domainLength * 0.5
utau         = 1.0
inputParams  = {"delta": delta, "kvisc":kvisc, "nunif": nunif, "utau": utau, "rho": rho,
                "caseN": caseN, "rlzStr": rlzStr}

# ------------ Results directory ------------

# post-processing directory
postDir = f"../../data/{caseN}/post"
if not os.path.exists(postDir):
    os.mkdir(postDir)
# post-processing sub-directory for single realization
rlzStr = f"{rlzN:05d}"
postRlzDir = os.path.join(postDir, f"post_{rlzStr}")
if not os.path.exists(postRlzDir):
    os.mkdir(postRlzDir)

# ------------ Total kinetic energy of the system ---------------

# --- APPROXIMATION calculation of Total Kinetiv Energy [J/kg]

# Get instantaneous velocities u,v,w along time in Half channel + Uniform grid, using:
# > interpolation from adaptative grid to uniform grid
# > constant density
(ydelta_1, yplus_1, time_1, uvel_1, vvel_1, wvel_1) = get_odt_instantaneous(inputParams)

# Calculate Total Kinetic Energy [J/kg]
E_1  = 0.5 * (uvel_1**2 + vvel_1**2 + wvel_1**2)    # 2-d np.array of shape [nunif, ntimes], where nunif = #num_points of uniform grid
Et_1 = np.sum(E_1, axis=0)                          # 1-d np.array of shape [ntimes]

# --- EXACT calculation of Total Kinetic Energy [J/m2]

# Get instantaneous velocities u,v,w along time in All channel + Adaptative grid, using:
# > data from adaptative grid (no interpolation)
# > constant density
(y_2, yf_2, dy_2, time_2, uvel_2, vvel_2, wvel_2) = get_odt_instantaneous_in_adaptative_grid(inputParams)

# Calculate Total Kinetic Energy [J/m2]
Et_2 = np.zeros(len(time_2))                                                        # 1-d np.array of length ntimes
for tidx in range(len(time_2)):
    E_1 = 0.5 * dy_2[tidx] * (uvel_2[tidx]**2 + vvel_2[tidx]**2 + wvel_2[tidx]**2)  # 1-d np.array of length #num_points of adaptative grid, non-constant along time
    Et_2[tidx] = np.sum(E_1)

# ------------ Build plots ------------

fig, ax = plt.subplots(2, figsize=(10,8))

# Approximation E[J/kg], includes interpolation
ax[0].plot(time_1, Et_1)
ax[0].set_xlabel(r"time $[s]$")
ax[0].set_ylabel(r"Total Kinetic Energy $[J/kg]$")

# Exact [E/m2]
ax[1].plot(time_2, Et_2)
ax[1].set_xlabel(r"time $[s]$")
ax[1].set_ylabel(r"Total Kinetic Energy $[J/m^2]$")

plt.tight_layout()
plt.savefig(os.path.join(postRlzDir, "total_kinetic_energy.jpg"))
plt.close()

