#!/bin/bash

TOKENIZERS_PARALLELISM=false python3 test.py --context_file=$1 --test_data=$2 --output_file=$3