#!/bin/bash
folder_path=./data/lad25
result_path=./exp_opt_arb_besa_ref_3

if [ -d $result_path ]; then
    echo "Result path $result_path already exists. Please remove it first."
    exit 1
fi

if [ ! -d "$result_path" ]; then
    mkdir -p "$result_path"
fi

# mode="run"
mode="harvest"

work_dir=/scratch/hanyu
date_str=20250128_010029  # for harvest

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
            cp ${work_dir}/work_${base_name}_${date_str}/config.json ./${result_path}/${base_name}.json
            # cp ${work_dir}/work_${base_name}_${date_str}/time.log ./${result_path}/${base_name}.log
        else
            echo "work_${base_name}_${date_str} does not exist in ${work_dir}"
        fi
    fi
done < <(find "$folder_path" -type f -name "*.c")

if [ "$mode" == "run" ]; then
    wait
    echo "All done"
fi


if [ "$mode" == "harvest" ]; then
    python3 plot_exp.py $result_path
fi