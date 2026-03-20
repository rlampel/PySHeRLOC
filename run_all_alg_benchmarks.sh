#!/bin/bash

# Define the file containing the benchmark names
# INPUT_FILE="benchmark_problems.txt"
INPUT_FILE="oed_problems.txt"

# Check if the file exists before starting
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found."
    exit 1
fi

# Arrays for the boolean combinations
BOOLS=("n" "y")

# Nested loops for all 8 combinations (-fs, -hess, -cond)
for hess_val in "${BOOLS[@]}"; do
    for cond_val in "${BOOLS[@]}"; do
        for fs_val in "${BOOLS[@]}"; do
            
            echo "--- Starting batch with FS=$fs_val, HESS=$hess_val, COND=$cond_val ---"

            # Loop through each line of the file
            while IFS= read -r name || [[ -n "$name" ]]; do
                # Skip empty lines
                [[ -z "$name" ]] && continue

                echo "Running benchmark for: $name"
                
                # Execute the python script with the new boolean arguments
                python3 benchmark_algs.py -n "$name" -hess $hess_val -cond $cond_val -fs $fs_val 

            done < "$INPUT_FILE"
            
        done
    done
done

echo "All benchmark combinations completed."
