#!/bin/bash

# Stop if no simulation files are found
shopt -s nullglob
FILES=(FoilSimulations/Main_*.txt)

TOTAL=${#FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "No simulation files found!"
    exit 1
fi

# Create data directory if necessary
mkdir -p data

# Count completed simulations
N_COMPLETED=0

for FILE in "${FILES[@]}"; do
    ID=$(basename "$FILE" .txt)
    ID=${ID#Main_}

    if [ -f "data/Completed_${ID}.done" ]; then
        ((N_COMPLETED++))
    fi
done

echo "======================================"
echo "Simulation files:       $TOTAL"
echo "Already completed:      $N_COMPLETED"
echo "Remaining simulations:  $((TOTAL - N_COMPLETED))"
echo "======================================"

# Run simulations
for FILE in "${FILES[@]}"
do
    ID=$(basename "$FILE" .txt)
    ID=${ID#Main_}

    # Skip successfully completed simulations
    if [ -f "data/Completed_${ID}.done" ]; then
        echo "Simulation $ID already completed. Skipping."
        continue
    fi

    echo ""
    echo "======================================"
    echo "Running simulation $ID"
    echo "File: $FILE"
    echo "======================================"

    # Run TOPAS
    topas "$FILE"

    # Check whether TOPAS succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "TOPAS failed or was stopped at simulation $ID."
        echo "Stopping batch."
        exit 1
    fi

    # Mark simulation as successfully completed
    touch "data/Completed_${ID}.done"

    echo "Finished simulation $ID"
done

echo ""
echo "======================================"
echo "All available simulations completed!"
echo "======================================"
