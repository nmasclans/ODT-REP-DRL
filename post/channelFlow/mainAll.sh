#!/bin/bash

# INSTRUCTIONS
# This bash file will efectivelly run 'mainRL.sh' for multiple realizations comparison, and 'main.sh' for each realization,
# with the required arguments for each bash file.
#
# Run as (e.g.):
# ./mainAll.sh 180 channel180_2_5_2024_longSimulation 0 channel180_2_5_2024_run1710856617 0 4 1 101 111 1000 1 0.1
# 
# Arguments (for mainAll.sh):
# - 1_Re_tau                             (for mainRL.sh)   
# - 2_case_name_nonRL                    (for mainRL.sh)           
# - 3_realization_number_nonRL           (for mainRL.sh)                   
# - 4_case_name_RL                       (for mainRL.sh & main.sh)       
# - 5_realization_number_min_RL          (for mainRL.sh & main.sh)                   
# - 6_realization_number_max_RL          (for mainRL.sh & main.sh)                   
# - 7_realization_number_step_RL         (for mainRL.sh & main.sh)                       
# - 8_time_begin_averaging               (for mainRL.sh & main.sh)
# - 9_time_end_averaging_non_converged   (for mainRL.sh & main.sh)                           
# - 10_time_end_averaging_converged      (for mainRL.sh)                       
# - 11_delta_time_stats                  (for main.sh)
# - 12_delta_time_stats_anisotropy_gifs  (for main.sh)
# 
# where:
# A. Arguments (for mainRL.sh):
#    - 1_Re_tau 
#    - 2_case_name_nonRL
#    - 3_realization_number_nonRL
#    - 4_case_name_RL
#    - 5_realization_number_min_RL
#    - 6_realization_number_max_RL
#    - 7_realization_number_step_RL
#    - 8_time_begin_averaging_non_converged
#    - 9_time_end_averaging_non_converged
#    - 10_time_end_averaging_converged
# B. Arguments (for main.sh):
#    - 1_case_name 
#    - 2_realization_number 
#    - 3_reynolds_number 
#    - 4_delta_time_stats 
#    - 5_delta_time_stats_anisotropy_gifs 
#    - 6_time_begin_averaging 
#    - 7_time_end_averaging 

echo "$#"
if [ "$#" -ne 12 ]; then
    echo "Usage: $0 <1_Re_tau> <2_case_name_nonRL> <3_realization_number_nonRL> <4_case_name_RL> <5_realization_number_min_RL> <6_realization_number_max_RL> <7_realization_number_step_RL> <8_time_begin_averaging> <9_time_end_averaging_non_converged> <10_time_end_averaging_converged> <11_delta_time_stats> <12_delta_time_stats_anisotropy_gifs>"
    exit 1
else
    echo -e "\n\n\n****************************************************************"
    echo -e "************************** mainAll.sh **************************"
    echo -e "****************************************************************"
    echo "- Reynolds number: $1"
    echo "- Case name non-RL: $2"
    echo "- Realization number non-RL: $3"
    echo "- Case name RL: $4"
    echo "- Realization number min RL: $5"
    echo "- Realization number max RL: $6"
    echo "- Realization number step RL: $7"
    echo "- Time begin averaging: $8"
    echo "- Time end averaging non-converged: $9"
    echo "- Time end averaging converged: ${10}"
    echo "- dt statistics: ${11}"
    echo "- dt statistics anisotropy gifs: ${12}"
fi

# Run main.sh for each realization
for ((rlzNum = $5; rlzNum <= $6; rlzNum+=$7)); do
    echo -e "\n\n\n****************************************************************"
    echo -e "****************** main.sh for realization #$rlzNum ******************"
    echo -e "****************************************************************"
    ./main.sh "$4" "$rlzNum" "$1" "${11}" "${12}" "$8" "$9"
done

# Run mainRL.sh for multiple realizations
echo -e "\n\n\n****************************************************************"
echo -e "************************** mainRL.sh ***************************"
echo -e "****************************************************************"
./mainRL.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}"