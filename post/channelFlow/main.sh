#!/bin/bash

# Run as:
# ./main.sh channel590 590 25 1

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <case_name> <reynolds_number> <delta_time_stats_stats> <delta_time_stats_anisotropy"
    exit 1
fi

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "--------------------- stats_odt_vs_dns.py ----------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 stats_odt_vs_dns.py "$1" "$2"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------------ stats_odt_convergence.py --------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 stats_odt_convergence.py "$1" "$2" "$3"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "---------------- anisotropy_tensor_odt_vs_dns.py ---------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 anisotropy_tensor_odt_vs_dns.py "$1" "$2"

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------- anisotropy_tensor_odt_convergence.py -------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 anisotropy_tensor_odt_convergence.py "$1" "$2" "$4"
