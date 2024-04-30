#!/bin/bash

# Check if both minimum and maximum RL numbers are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <min_rl_id_number> <max_rl_id_number>"
    exit 1
fi

# Extract minimum and maximum RL numbers from arguments
min_rl="$1"
max_rl="$2"

# Loop from min_rl to max_rl
for ((RL=min_rl; RL<=max_rl; RL++)); do
    # Construct the scp command with the current RL number
    echo "\n\n--------- RL #"$RL"---------"  
    scp -r jofre@triton.eebe.upc.edu:/home/jofre/Nuria/repositories/ODT/data/channel180_RL"$RL"/ /home/eckert/Documents/Nuria_Masclans/repositories/ODT/data/
    scp jofre@triton.eebe.upc.edu:/home/jofre/Nuria/repositories/MARL-Statistics-Convergence/src/nohup_RL"$RL".out /home/eckert/Documents/Nuria_Masclans/repositories/ODT/data/channel180_RL"$RL"/nohup/
done
