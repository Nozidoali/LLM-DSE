#!/bin/bash
folder_path=./data/pack2
result_path=./results0117

# mode="run"
mode="harvest"

work_dir=/scratch/hanyu
date_str=20250116_152048 # for harvest

while IFS= read -r c_file; do
    base_name=$(basename "$c_file" .c)
    echo "Found: $base_name"
    if [ "$mode" == "run" ]; then
        echo "Running $base_name"
        python3 main.py --benchmark "$base_name" --folder "$folder_path" &
    fi
    if [ "$mode" == "harvest" ]; then
        echo "Checking if work_${base_name}_${date_str} exists in ${work_dir}"
        if [ -d "${work_dir}/work_${base_name}_${date_str}" ]; then
            echo "work_${base_name}_${date_str} exists in ${work_dir}"
            cp ${work_dir}/work_${base_name}_${date_str}/results.csv ./${result_path}/${base_name}.csv
            cp ${work_dir}/work_${base_name}_${date_str}/openai.log ./${result_path}/${base_name}.txt
            cp ${work_dir}/work_${base_name}_${date_str}/time.log ./${result_path}/${base_name}.log
        else
            echo "work_${base_name}_${date_str} does not exist in ${work_dir}"
        fi
    fi
done < <(find "$folder_path" -type f -name "*.c")