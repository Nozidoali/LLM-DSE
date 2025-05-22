#!/bin/bash
work_dir=/scratch/hanyu
folder_path=./data/lad25


while IFS= read -r c_file; do
    base_name=$(basename "$c_file" .c)
    
    echo "Running $base_name"
    python3 src/main.py --benchmark "$base_name" --folder "$folder_path" &

done < <(find "$folder_path" -type f -name "*.c")

wait
echo "All done"


