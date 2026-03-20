
#!/bin/bash

# Define the file containing the benchmark names
INPUT_FILE="oed_problems.txt"

# Check if the file exists before starting
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found."
    exit 1
fi

# Arrays for the boolean combinations
SOLVERS=("blockSQP2" "IPOPT")
BOOLS=("n" "y")

# Nested loops for all 8 combinations (-fs, -hess, -cond)
for solver in "${SOLVERS[@]}"; do
    for hess_val in "${BOOLS[@]}"; do
	echo "--- Starting batch with SOLVER=$solver, HESS=$hess_val ---"

        # Loop through each line of the file
        while IFS= read -r name || [[ -n "$name" ]]; do
            # Skip empty lines
            [[ -z "$name" ]] && continue

            echo "Running benchmark for: $name"
                
            # Execute the python script with the new boolean arguments
            python3 benchmark_def.py -n "$name" -hess $hess_val -solver $solver

        done < "$INPUT_FILE"
    done
done

echo "All benchmark combinations completed."
