#!/bin/bash

# INSTRUCTIONS:
#
# Run as (e.g.):
# ./main.sh channel180_2_5_2024 0 180 10 1 50
#
# Arguments:
# - 1_case_name 
# - 2_realization_number 
# - 3_reynolds_number 
# - 4_delta_time_stats 
# - 5_delta_time_stats_anisotropy_gifs 
# - 6_time_begin_averaging 
# - 7_time_end_averaging 

# Check if the correct number of arguments is provided
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <1_case_name> <2_realization_number> <3_reynolds_number> <4_delta_time_stats> <5_delta_time_stats_anisotropy_gifs> <6_time_begin_averaging> <7_time_end_averaging>"
    exit 1
else
    echo "- Case name: $1"
    echo "- Realization number: $2"
    echo "- Reynolds number: $3"
    echo "- dt statistics: $4"
    echo "- dt statistics anisotropy gifs: $5"
    echo "- Time begin averaging: $6"
    echo "- Time end averaging: $7"
fi

### # Generate dmp file for odt_end.dat file (idem. for statistics & state) if last dmp file time < end file time
### echo -e "\n\n\n----------------------------------------------------------------"
### echo -e "----------------- load_end_file_to_dmp_file.py -----------------"
### echo -e "----------------------------------------------------------------\n\n\n"
### python3 load_end_file_to_dmp_file.py "$1" "$2"

### # Execute multiple post-processing python3 scripts, with corresponding call arguments
### echo -e "\n\n\n----------------------------------------------------------------"
### echo -e "--------------------- stats_odt_vs_dns.py ----------------------"
### echo -e "----------------------------------------------------------------\n\n\n"
### python3 stats_odt_vs_dns.py "$1" "$2" "$3" "$6" "$7"
 
echo -e "\n\n\n----------------------------------------------------------------"
echo -e "--------------------- stats_odt_vs_odtReference.py ----------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 stats_odt_vs_odtReference.py "$1" "$2" "$3" "$6" "$7"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------------ stats_odt_convergence.py --------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 stats_odt_convergence.py "$1" "$2" "$3" "$4" "$6" "$7"

### echo -e "\n\n\n----------------------------------------------------------------"
### echo -e "---------------- anisotropy_tensor_odt_vs_dns.py ---------------"
### echo -e "----------------------------------------------------------------\n\n\n"
### python3 anisotropy_tensor_odt_vs_dns.py "$1" "$2" "$3" "$6" "$7"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "---------------- anisotropy_tensor_odt_vs_odtReference.py ---------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 anisotropy_tensor_odt_vs_odtReference.py "$1" "$2" "$3" "$6" "$7"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------- anisotropy_tensor_odt_convergence.py -------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 anisotropy_tensor_odt_convergence.py "$1" "$2" "$3" "$5" "$6" "$7"