#!/bin/bash
folder_path=./data/lad25
result_path=./exp_opt_arb_bes_2

while IFS= read -r c_file; do
    base_name=$(basename "$c_file" .c)
    python3 analyze_log.py --benchmark "$base_name"
done < <(find "$folder_path" -type f -name "*.c")