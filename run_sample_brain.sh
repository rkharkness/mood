#!/bin/bash

# Check if the input arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input_folder output_folder"
  exit 1
fi

input_folder="$1"
output_folder="$2"

python3 /workspace/scripts/ood-llr-swindow.py --iw_samples_elbo 3 \
--n_latents_skip 2 \
--model_dir "/workspace/models/BrainMOOD128x128Dequantized-2023-08-28-20-17-37.529063/" \
--scan "brain" --save_dir $output_folder --source_dir $input_folder

