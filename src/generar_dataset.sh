#!/bin/bash

DIR_APPLE="/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Dataset-2/apple"
DIR_HUAWEI="/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Dataset-2/huawei"
DIR_LG="/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Dataset-2/lg"
DIR_PRUEBAS="/Users/Gago/Desktop/Universidad/Master/TFM/pruebas/"
FILENAME_GROUPS="groups.txt"

SYMMETRIC=1

DIRS=()

for dir in $(ls -d $DIR_APPLE/*/);
do
    DIRS+="$dir,"
done

for dir in $(ls -d $DIR_HUAWEI/*/);
do
    DIRS+="$dir,"
done

for dir in $(ls -d $DIR_LG/*/);
do
    DIRS+="$dir,"
done

DIRS=($(echo $DIRS | sed 's/,$//'))

DIRS=($(echo $DIRS | tr ',' '\n' | awk 'BEGIN { srand() } { print rand() "\t" $0 }' | sort -n | cut -f2- | tr '\n' ',' | sed 's/,$//' | tr ',' ' ' )) #shuffle array

NEW_DIR=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR=$DIR_PRUEBAS$NEW_DIR

mkdir -p ${OUTPUT_DIR}/images #creamos el directorio en el que se van a guardar las imagenes


NUM_CAMERAS=4
IT=0
for i in "${DIRS[@]}";
do
    if [ $IT -ge $NUM_CAMERAS ]
    then 
	break
    fi
    COUNTER=0
    echo $(basename $i) >> ${OUTPUT_DIR}/${FILENAME_GROUPS}
    for file in ${i}test/*;
    do
	if [ $SYMMETRIC -eq 1 ]
	then
	    DELTA=0
	else
	    DELTA=$(((RANDOM % 5)+1))
	fi

	if [ $(expr $COUNTER + $DELTA) -lt $1 ]
	then
	    CAMERANAME=$(basename $i)
	    FILENAME=$(basename $file)
	    cp $file ${OUTPUT_DIR}/images/${CAMERANAME}_${FILENAME}
	    COUNTER=$(expr $COUNTER + 1)
	else
	    break 1
	fi
    done
    IT=$(expr $IT + 1)
done

echo "Directory: " $NEW_DIR
#sed -i.bak "4 s/.*/cluster="$NEW_DIR"/" evaluate.py 
#rm evaluate.py.bak
#sed -i.bak "252 s/.*/attempt="$NEW_DIR"/" image_clustering.py
#rm image_clustering.py.bak

echo $NEW_DIR > cluster.info
