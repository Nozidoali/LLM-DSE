#!/bin/bash
work_dir=/scratch/hanyu
folder_path=./data/lad25

harvest_from=20250129_224833  # for harvest
harvest_to=./exp_0130_2_oab

if [ -d $harvest_from ]; then
    echo "Result path $harvest_from already exists. Please remove it first."
    exit 1
fi

if [ ! -d "$harvest_from" ]; then
    mkdir -p "$harvest_from"
fi

while IFS= read -r c_file; do
    base_name=$(basename "$c_file" .c)
    echo "Found: $base_name"

    echo "Checking if work_${base_name}_${date_str} exists in ${work_dir}"
    if [ -d "${work_dir}/work_${base_name}_${date_str}" ]; then
        echo "work_${base_name}_${date_str} exists in ${work_dir}"
        cp ${work_dir}/work_${base_name}_${date_str}/results.csv ./${result_path}/${base_name}.csv
        cp ${work_dir}/work_${base_name}_${date_str}/openai.log  ./${result_path}/${base_name}.txt
        cp ${work_dir}/work_${base_name}_${date_str}/config.json ./${result_path}/${base_name}.json
    else
        echo "work_${base_name}_${date_str} does not exist in ${work_dir}"
    fi

done < <(find "$folder_path" -type f -name "*.c")

python3 scripts/plot_exp.py $result_path
