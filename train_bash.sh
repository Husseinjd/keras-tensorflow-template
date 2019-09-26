#!/bin/bash

rm -r experiments

python -W ignore train.py -cd configs/test_configs

echo '--------------------------------'
echo '--------------------------------'
echo '--------------------------------'
echo '--------------------------------'
echo 'Evaluation Results: '
python evaluate.py -d experiments/2019-09-26/