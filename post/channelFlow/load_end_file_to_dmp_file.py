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
fEnd        = os.path.join(dataDir, "odt_end.dat")
fDmpAll     = glob.glob(os.path.join(dataDir, "dmp_*.dat"))
fDmpLast    = sorted(fDmpAll)[-1]
timeEnd     = get_time(fEnd)
timeDmpLast = get_time(fDmpLast)
if timeDmpLast < timeEnd:
    # copy odt_end.dat data into next dmp file, with next dmp file idx (=length of fInstDmpAll, because indexing begins with 0)
    dmpNextIdx = len(fDmpAll)
    dmpNextStr = f"{dmpNextIdx:05d}"
    fDmpNext   = os.path.join(dataDir, f"dmp_{dmpNextStr}.dat")
    print(f"\nFile '{fEnd}' copied as new file '{fDmpNext}'")
    shutil.copy2(fEnd, fDmpNext)


# statistics data
fEnd        = os.path.join(dataDir, "statistics/stat_odt_end.dat")
fDmpAll     = glob.glob(os.path.join(dataDir, "statistics/stat_dmp_*.dat"))
fDmpLast    = sorted(fDmpAll)[-1]
timeEnd     = get_time(fEnd)
timeDmpLast = get_time(fDmpLast)
if timeDmpLast < timeEnd:
    # copy odt_end.dat data into next dmp file, with next dmp file idx (=length of fInstDmpAll, because indexing begins with 0)
    dmpNextIdx = len(fDmpAll)
    dmpNextStr = f"{dmpNextIdx:05d}"
    fDmpNext   = os.path.join(dataDir, f"statistics/stat_dmp_{dmpNextStr}.dat")
    print(f"\nFile '{fEnd}' copied as new file '{fDmpNext}'")
    shutil.copy2(fEnd, fDmpNext)


# state data
fEnd        = os.path.join(dataDir, "state/state_odt_end.dat")
fDmpAll     = glob.glob(os.path.join(dataDir, "state/state_dmp_*.dat"))
fDmpLast    = sorted(fDmpAll)[-1]
timeEnd     = get_time(fEnd)
timeDmpLast = get_time(fDmpLast)
if timeDmpLast < timeEnd:
    # copy odt_end.dat data into next dmp file, with next dmp file idx (=length of fInstDmpAll, because indexing begins with 0)
    dmpNextIdx = len(fDmpAll)
    dmpNextStr = f"{dmpNextIdx:05d}"
    fDmpNext   = os.path.join(dataDir, f"state/state_dmp_{dmpNextStr}.dat")
    print(f"\nFile '{fEnd}' copied as new file '{fDmpNext}'")
    shutil.copy2(fEnd, fDmpNext)