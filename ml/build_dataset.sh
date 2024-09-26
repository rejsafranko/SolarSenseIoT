#!/bin/bash
mkdir -p data/merged/{clean,dirty}

main_dir="data"

for dataset_dir in "$main_dir"/dataset*; do
  if [ -d "$dataset_dir" ]; then
    if [ -d "$dataset_dir/clean" ]; then
      cp "$dataset_dir/clean"/* "$main_dir/merged/clean/"
    fi
    
    if [ -d "$dataset_dir/dirty" ]; then
      cp "$dataset_dir/dirty"/* "$main_dir/merged/dirty/"
    fi
  fi
done

echo "Images copied to merged directories successfully."

python src/split_data.py --data_dir data/