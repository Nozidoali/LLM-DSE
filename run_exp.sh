#!/bin/bash
folder_path=./data/test
result_path=./results0105

mode="run"
# mode="harvest"

work_dir=/home/s41/project/cs259-llm-dse/
date_str=20250105_045636 # for harvest

while IFS= read -r c_file; do
    base_name=$(basename "$c_file" .c)
    echo "Found: $base_name"
    if [ "$mode" == "run" ]; then
        echo "Running $base_name"
        python3 main.py --benchmark "$base_name" --folder "$folder_path" &
    fi
    if [ "$mode" == "harvest" ]; then
        echo "Checking if work_${base_name}_${date_str} exists in /sratch/hanyu"
        if [ -d "${work_dir}/work_${base_name}_${date_str}" ]; then
            echo "work_${base_name}_${date_str} exists in /scratch/hanyu"
            cp ${work_dir}/work_${base_name}_${date_str}/results.csv ./${result_path}/${base_name}.csv
            cp ${work_dir}/work_${base_name}_${date_str}/openai.log ./${result_path}/${base_name}.txt
        else
            echo "work_${base_name}_${date_str} does not exist in /scratch/hanyu"
        fi
    fi
done < <(find "$folder_path" -type f -name "*.c")