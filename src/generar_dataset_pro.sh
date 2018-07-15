#!/bin/bash

DIR_DATASET="/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Dataset-pro"
DIR_CANON=${DIR_DATASET}/"Canon_60D"
DIR_NIKON_D90=${DIR_DATASET}/"Nikon_D90"
DIR_NIKON_D70="${DIR_DATASET}/Nikon_D7000"
DIR_SONY_A57="${DIR_DATASET}/Sony_A57"

SYMMETRIC=1

DIR_PRUEBAS="/Users/Gago/Desktop/Universidad/Master/TFM/pruebas/"
FILENAME_GROUPS="groups.txt"

DIRS=($DIR_CANON $DIR_NIKON_D90 $DIR_NIKON_D70 $DIR_SONY_A57)

NEW_DIR=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR=$DIR_PRUEBAS$NEW_DIR

mkdir -p ${OUTPUT_DIR}/images #creamos el directorio en el que se van a guardar las imagenes

for i in "${DIRS[@]}";
do
    echo $(basename $i) >> ${OUTPUT_DIR}/${FILENAME_GROUPS}
    if [ $SYMMETRIC -eq 1 ]
    then
	DELTA=0
    else
	DELTA=$(((RANDOM % 3) + 1))
    fi
    NUM_IMAGES=$(expr $1 + $DELTA)
    FILES=$(ls -1 $i | python3 -c "import sys; import random; print(''.join(random.sample(sys.stdin.readlines(), int(sys.argv[1]))).rstrip())" $NUM_IMAGES)
    for file in $FILES;
    do
        FILENAME=$(basename $file)
	cp ${i}/${file} ${OUTPUT_DIR}/images/$(basename $i)_${FILENAME}
    done
done

echo "Directory: " $NEW_DIR
echo $NEW_DIR > cluster.info
