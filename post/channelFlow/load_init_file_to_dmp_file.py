import sys
import shutil
import glob
import os

from utils import *

#--------------------------------------------------------------------------------------------

try :
    caseN    = sys.argv[1]
    rlzN     = int(sys.argv[2])
    print(f"Script parameters: \n- Case name: {caseN} \n- Realization Number: {rlzN}")
except :
    raise ValueError("Missing call arguments, should be: <case_name> <realization_number>")

odt_path = os.environ.get("ODT_PATH")
rlzStr   = f"{rlzN:05d}"
dataDir  = f"{odt_path}/data/{caseN}/data/data_{rlzStr}/"

# instantaneous data
fInit    = os.path.join(dataDir, "odt_init.dat")
fDmp0    = os.path.join(dataDir, "dmp_00000.dat")
if not(os.path.exists(fDmp0)):
    print(f"\nFile '{fInit}' copied as new file '{fDmp0}'")
    shutil.copy2(fInit, fDmp0)


# statistics data
fInit    = os.path.join(dataDir, "statistics/stat_odt_init.dat")
fDmp0    = os.path.join(dataDir, "statistics/stat_dmp_00000.dat")
if not(os.path.exists(fDmp0)):
    print(f"\nFile '{fInit}' copied as new file '{fDmp0}'")
    shutil.copy2(fInit, fDmp0)


# state data
fInit    = os.path.join(dataDir, "state/state_odt_init.dat")
fDmp0    = os.path.join(dataDir, "state/state_dmp_00000.dat")
if not(os.path.exists(fDmp0)):
    print(f"\nFile '{fInit}' copied as new file '{fDmp0}'")
    shutil.copy2(fInit, fDmp0)