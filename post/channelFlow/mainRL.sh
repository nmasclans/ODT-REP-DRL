#!/bin/bash

# INSTRUCTIONS
# 
# Run as (e.g.):
# ./mainRL.sh 180 channel180_2_5_2024 0 0 1 101 111 1000
# 
# Arguments:
#    - 1_Re_tau 
#    - 2_case_name_RL
#    - 3_realization_number_min_RL
#    - 4_realization_number_max_RL
#    - 5_realization_number_step_RL
#    - 6_time_begin_averaging_non_converged
#    - 7_time_end_averaging_non_converged
#    - 8_time_end_averaging_converged
#    - 9_dt_statistics
#    - 10_dt_statistics_RL_gifs
#    - 11_nohup_filename
#    - 12_actions_avg_freq 

# Check if the correct number of arguments is provided
if [ "$#" -ne 13 ]; then
    echo "Usage: $0 <1_Re_tau> <2_case_name_RL> <3_realization_number_min_RL> <4_realization_number_max_RL> <5_realization_number_step_RL> <6_time_begin_averaging_non_converged> <7_time_end_averaging_non_converged> <8_time_end_averaging_converged> <9_dt_statistics_nonRL> <10_dt_statistics_RL> <11_dt_statistics_RL_gifs> <12_nohup_filename> <13_actions_avg_freq>"
    exit 1
else
    echo -e "\n\n\n****************************************************************"
    echo -e "************************** mainRL.sh ***************************"
    echo -e "****************************************************************"
    echo "- Reynolds number: $1"
    echo "- Case name RL: $2"
    echo "- Realization number min RL: $3"
    echo "- Realization number max RL: $4"
    echo "- Realization number step RL: $5"
    echo "- Time begin averaging: $6"
    echo "- Time end averaging non-converged: $7"
    echo "- Time end averaging converged: $8"
    echo "- dt statistics non-RL: $9"
    echo "- dt statistics RL: ${10}"
    echo "- dt statistics RL gifs: ${11}"
    echo "- Nohup filename for RL training run: ${12}"
    echo "- Actions averaging frequency, number of simulation steps for averaged actions kde: ${13}"
fi

### # For each realization, if needed:
### # Generate dmp file for odt_end.dat file (idem. for statistics & state) if last dmp file time < end file time
### echo -e "\n\n\n----------------------------------------------------------------"
### echo -e "----------------- load_end_file_to_dmp_file.py -----------------"
### echo -e "----------------------------------------------------------------\n\n\n"
### python3 load_end_file_to_dmp_file.py "$2" "$3"
### for ((rlzNum = $3; rlzNum <= $4; rlzNum+=$5)); do
###     python3 load_end_file_to_dmp_file.py "$2" "$rlzNum"
### done

echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------------ RL_realization_results.py -------------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 RL_realization_results.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"

### echo -e "\n\n\n----------------------------------------------------------------"
### echo -e "-------------- RL_realization_convergence_gifs.py --------------"
### echo -e "----------------------------------------------------------------\n\n\n"
### python3 RL_realization_convergence_gifs.py "$1" "$2" "$3" "$4" "$5" "$6" "${11}"

### echo -e "\n\n\n----------------------------------------------------------------"
### echo -e "------------------ RL_run_actions_rewards.py -------------------"
### echo -e "----------------------------------------------------------------\n\n\n"
### python3 RL_run_actions_rewards.py "$2" "$3" "$4" "$5" "$6" "$7"
### 
echo -e "\n\n\n----------------------------------------------------------------"
echo -e "------------- RL_run_actions_rewards_from_nohup.py -------------"
echo -e "----------------------------------------------------------------\n\n\n"
python3 RL_run_actions_rewards_from_nohup.py "$2" "${12}" "${13}"
