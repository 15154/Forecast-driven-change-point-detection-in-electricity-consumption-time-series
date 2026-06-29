#!/bin/bash

# """
# Author: Mathias de Schietere
# Organization: UCLouvain
# GitHub: https://github.com/15154
# Created: 2026-03-12
# """

file="ECOS2026/jobs.txt"
i=0

while IFS= read -r cmd
do
    echo "Submitting job $((i+1)) at $(date)"

    sleep 1

    # send "y" automatically
    printf "y\n" | eval "$cmd"

    if [ $i -eq 0 ]; then
        sleep 1
    else
        sleep 360
    fi

    ((i++))

done < "$file"

echo "All jobs submitted."
