#!/bin/bash
folder_path=./data
while IFS= read -r c_file; do
    base_name=$(basename "$c_file" .c)
    echo "Found: $base_name"
    python3 main.py --benchmark "$base_name" &
done < <(find "$folder_path" -type f -name "*.c")