#!/bin/bash

DIR_DATASET="/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Video"
DIR_SAMSUNG=${DIR_DATASET}/"samsung_g6"
DIR_IPHONE_ROB=${DIR_DATASET}/"iphone7_roberto"
DIR_BQ_AQUARIS=${DIR_DATASET}/"bq_aquaris_e5"
DIR_IPHONE_8=${DIR_DATASET}/"iphone_8plus"
DIR_HUAWEI=${DIR_DATASET}/"huawei_y635_l01"
DIR_XIOMI=${DIR_DATASET}/"xiomi_m3"
DIR_MOTO=${DIR_DATASET}/"nexus"

SYMMETRIC=1

DIR_PRUEBAS="/Users/Gago/Desktop/Universidad/Master/TFM/pruebas/"
FILENAME_GROUPS="groups.txt"

DIRS=($DIR_SAMSUNG $DIR_BQ_AQUARIS $DIR_XIOMI $DIR_IPHONE_ROB $DIR_MOTO)

NEW_DIR=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR=$DIR_PRUEBAS$NEW_DIR

mkdir -p ${OUTPUT_DIR}/vids #creamos el directorio en el que se van a guardar las imagenes
mkdir -p ${OUTPUT_DIR}/noise

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
	ln -s ${i}/${file} ${OUTPUT_DIR}/vids/$(basename $i)_${FILENAME}
    done
done

echo "Directory: " $NEW_DIR
echo $NEW_DIR > cluster.info
