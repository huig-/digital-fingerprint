#!/bin/bash

DIR_DATASET="/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Dataset-pro"
DIR_CANON=${DIR_DATASET}/"Canon_60D"
DIR_NIKON_D90=${DIR_DATASET}/"Nikon_D90"
DIR_NIKON_D70="${DIR_DATASET}/Nikon_D7000"
DIR_SONY_A57="${DIR_DATASET}/Sony_A57"

DIR_PRUEBAS="/Users/Gago/Desktop/Universidad/Master/TFM/pruebas/"
FILENAME_GROUPS="groups.txt"

DIRS=($DIR_CANON $DIR_NIKON_D90 $DIR_NIKON_D70 $DIR_SONY_A57)

DIR_100="pro100"
DIR_90="pro90"
DIR_85="pro85"
DIR_80="pro80"
OUTPUT_DIR_100=$DIR_PRUEBAS$DIR_100
OUTPUT_DIR_90=$DIR_PRUEBAS$DIR_90
OUTPUT_DIR_85=$DIR_PRUEBAS$DIR_85
OUTPUT_DIR_80=$DIR_PRUEBAS$DIR_80

mkdir -p ${OUTPUT_DIR_100}/images #creamos el directorio en el que se van a guardar las imagenes
mkdir -p ${OUTPUT_DIR_90}/images #creamos el directorio en el que se van a guardar las imagenes
mkdir -p ${OUTPUT_DIR_85}/images #creamos el directorio en el que se van a guardar las imagenes
mkdir -p ${OUTPUT_DIR_80}/images #creamos el directorio en el que se van a guardar las imagenes

for i in "${DIRS[@]}";
do
    FILES=$(ls -p $i | grep -v / | python3 -c "import sys; import random; print(''.join(random.sample(sys.stdin.readlines(), int(sys.argv[1]))).rstrip())" $1)
    echo $FILES
    for file in $FILES;
    do
        FILENAME=$(basename $file)
        FILENAME_NOEXT=${FILENAME%%.*}
	    cp ${i}/${file} ${OUTPUT_DIR_100}/images/$(basename $i)_${FILENAME}
        cp ${i}/j90/${FILENAME_NOEXT}.jpeg ${OUTPUT_DIR_90}/images/$(basename $i)_${FILENAME_NOEXT}.jpeg
        cp ${i}/j85/${FILENAME_NOEXT}.jpeg ${OUTPUT_DIR_85}/images/$(basename $i)_${FILENAME_NOEXT}.jpeg
        cp ${i}/j80/${FILENAME_NOEXT}.jpeg ${OUTPUT_DIR_80}/images/$(basename $i)_${FILENAME_NOEXT}.jpeg
    done
done
