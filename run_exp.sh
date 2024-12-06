#!/bin/bash
folder_path=./data

# mode="run"
mode="harvest"

date_str=20241206_062303
work_dir=/scratch/hanyu

while IFS= read -r c_file; do
    base_name=$(basename "$c_file" .c)
    echo "Found: $base_name"
    if [ "$mode" == "run" ]; then
        echo "Running $base_name"
        python3 main.py --benchmark "$base_name" &
    fi
    if [ "$mode" == "harvest" ]; then
        echo "Checking if work_${base_name}_${date_str} exists in /sratch/hanyu"
        if [ -d "${work_dir}/work_${base_name}_${date_str}" ]; then
            echo "work_${base_name}_${date_str} exists in /scratch/hanyu"
            cp ${work_dir}/work_${base_name}_${date_str}/results.csv ./results/${base_name}.csv
            cp ${work_dir}/work_${base_name}_${date_str}/openai.log ./results/${base_name}.txt
        else
            echo "work_${base_name}_${date_str} does not exist in /scratch/hanyu"
        fi
    fi
done < <(find "$folder_path" -type f -name "*.c")