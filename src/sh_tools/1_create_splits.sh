#!/bin/bash

# Set the Python command for creating splits
python create_splits_seq.py \
    --task Eye_9tasks \
    --seed 1 \
    --k 10

echo "Data splits creation completed!" 