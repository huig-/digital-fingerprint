#!/bin/bash
#./generar_dataset.sh $1
#./generar_dataset_pro.sh $1
./generar_dataset_vid.sh $1
#python3 image_clutering.py
python3 video_clustering.py
python3 evaluate.py
