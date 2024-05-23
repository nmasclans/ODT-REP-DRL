#!/bin/bash

# INSTRUCTIONS
# This bash file will efectivelly run 'mainRL.sh' for multiple realizations comparison, and 'main.sh' for each realization,
# with the required arguments for each bash file.
#
# Run as (e.g.):
# ./mainAll.sh 180 channel180_2_5_2024_run1710856617 0 4 1 101 111 1000 1 0.1
# 
# Arguments (for mainAll.sh):
# - 1_Re_tau                             (for mainRL.sh)   
# - 2_case_name_RL                       (for mainRL.sh & main.sh)       
# - 3_realization_number_min_RL          (for mainRL.sh & main.sh)                   
# - 4_realization_number_max_RL          (for mainRL.sh & main.sh)                   
# - 5_realization_number_step_RL         (for mainRL.sh & main.sh)                       
# - 6_time_begin_averaging               (for mainRL.sh & main.sh)
# - 7_time_end_averaging_non_converged   (for mainRL.sh & main.sh)                           
# - 8_time_end_averaging_converged       (for mainRL.sh)                       
# - 9_delta_time_stats                   (for main.sh)
# - 10_delta_time_stats_anisotropy_gifs  (for main.sh)
# - 11_nohup_filename                    (for mainRL.sh)
# - 12_actions_avg_freq                  (for mainRL.sh)
# 
# where:
# A. Arguments (for mainRL.sh):
#    - 1_Re_tau 
#    - 2_case_name_RL
#    - 3_realization_number_min_RL
#    - 4_realization_number_max_RL
#    - 5_realization_number_step_RL
#    - 6_time_begin_averaging_non_converged
#    - 7_time_end_averaging_non_converged
#    - 8_time_end_averaging_converged
#    - 9_dt_statistics
#    - 10_nohup_filename
#    - 11_actions_avg_freq 
# B. Arguments (for main.sh):
#    - 1_case_name 
#    - 2_realization_number 
#    - 3_reynolds_number 
#    - 4_delta_time_stats 
#    - 5_delta_time_stats_anisotropy_gifs 
#    - 6_time_begin_averaging 
#    - 7_time_end_averaging 


echo "$#"
if [ "$#" -ne 13 ]; then
    echo "Usage: $0 <1_Re_tau> <2_case_name_RL> <3_realization_number_min_RL> <4_realization_number_max_RL> <5_realization_number_step_RL> <6_time_begin_averaging> <7_time_end_averaging_non_converged> <8_time_end_averaging_converged> <9_delta_time_stats_nonRL> <10_delta_time_stats_RL> <11_delta_time_stats_anisotropy_gifs> <12_nohup_filename> <13_actions_avg_freq>"
    exit 1
else
    echo -e "\n\n\n****************************************************************"
    echo -e "************************** mainAll.sh **************************"
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
    echo "- dt statistics anisotropy gifs: ${11}"
    echo "- Nohup filename for RL training run: ${12}"
    echo "- Actions averaging frequency, number of simulation steps for averaged actions kde: ${13}"
fi

# For each realization, if needed:
# Generate dmp file for odt_init.dat file (idem. for statistics & state) if first dmp file id 00000 does not exist
echo -e "\n\n\n----------------------------------------------------------------"
echo -e "----------------- load_init_file_to_dmp_file.py ----------------"
echo -e "----------------------------------------------------------------\n\n\n"
for ((rlzNum = $3; rlzNum <= $4; rlzNum+=$5)); do
    python3 load_init_file_to_dmp_file.py "$2" "$rlzNum"
done

# For each realization, if needed:
# Generate dmp file for odt_end.dat file (idem. for statistics & state) if last dmp file time < end file time
echo -e "\n\n\n----------------------------------------------------------------"
echo -e "----------------- load_end_file_to_dmp_file.py -----------------"
echo -e "----------------------------------------------------------------\n\n\n"
for ((rlzNum = $3; rlzNum <= $4; rlzNum+=$5)); do
    python3 load_end_file_to_dmp_file.py "$2" "$rlzNum"
done

# Run main.sh for each realization
### for ((rlzNum = $3; rlzNum <= $4; rlzNum+=$5)); do
###     echo -e "\n\n\n****************************************************************"
###     echo -e "****************** main.sh for realization #$rlzNum ******************"
###     echo -e "****************************************************************"
###     ./main.sh "$2" "$rlzNum" "$1" "$9" "${11}" "$6" "$7"
### done

# Run mainRL.sh for multiple realizations
echo -e "\n\n\n****************************************************************"
echo -e "************************** mainRL.sh ***************************"
echo -e "****************************************************************"
./mainRL.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${12}" "${13}"