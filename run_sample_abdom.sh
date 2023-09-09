#!/bin/bash

# Check if the input arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input_folder output_folder"
  exit 1
fi

input_folder="$1"
output_folder="$2"

python /workspace/scripts/ood-llr-swindow.py --iw_samples_elbo 3 \
--n_latents_skip 2 \
--model_dir "/workspace/models/AbdomMOOD256x256Dequantized-2023-08-29-15-24-14.597920/" \
--scan "abdom" \
--save_dir $output_folder \
--source_dir $input_folder \

