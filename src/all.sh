#!/bin/bash
#./generar_dataset.sh $1
#./generar_dataset_pro.sh $1
python3 image_clustering.py
python3 evaluate.py
