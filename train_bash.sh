#!/bin/bash

rm -r experiments

python -W ignore train.py -cd configs/test_configs -e t

