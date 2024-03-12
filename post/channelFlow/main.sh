#!/bin/bash

# Run as:
# ./main.sh channel590 0 590 25 1
#./main.sh channel180_2_5_2024_longSimulation 0 180 50 1

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <case_name> <realization_number> <reynolds_number> <delta_time_stats_stats> <delta_time_stats_anisotropy>"
    exit 1
fi

# Execute multiple post-processing python3 scripts, with corresponding call arguments
echo -e "\n\n\n----------------------------------------------------------------"
echo -e "--------------------- stats_odt_vs_dns.py ----------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 stats_odt_vs_dns.py "$1" "$2" "$3"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------------ stats_odt_convergence.py --------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 stats_odt_convergence.py "$1" "$2" "$3" "$4"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "---------------- anisotropy_tensor_odt_vs_dns.py ---------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 anisotropy_tensor_odt_vs_dns.py "$1" "$2" "$3"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------- anisotropy_tensor_odt_convergence.py -------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 anisotropy_tensor_odt_convergence.py "$1" "$2" "$3" "$5"