#!/bin/bash

TOTAL=$(ls FoilSimulations/Main_*.txt | wc -l)

# Count existing PDD output files
N_OUTPUT=$(ls data/PDD_*.csv 2>/dev/null | wc -l)

# TOPAS creates the current output before it has finished,
# so assume the final existing file may be incomplete.
START=$N_OUTPUT

# Simulation IDs start at 1
if [ "$START" -lt 1 ]; then
    START=1
fi

echo "======================================"
echo "Total simulations: $TOTAL"
echo "Existing outputs:  $N_OUTPUT"
echo "Starting from:      $START"
echo "======================================"

for ((i=START; i<=TOTAL; i++))
do
    ID=$(printf "%04d" "$i")
    FILE="FoilSimulations/Main_${ID}.txt"

    echo ""
    echo "======================================"
    echo "Simulation $i / $TOTAL"
    echo "Running: $FILE"
    echo "======================================"

    topas "$FILE"

    # Check whether TOPAS succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "TOPAS failed or was stopped at simulation $ID."
        echo "Stopping batch."
        exit 1
    fi

    echo "Finished simulation $ID"
done

echo ""
echo "======================================"
echo "All $TOTAL simulations finished!"
echo "======================================"
