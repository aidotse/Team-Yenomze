#!/bin/bash

IMAGE_NAME=yenomze:adipocyte
YENOMZE_HOME=/home/group4
DATA=$YENOMZE_HOME/astra_data_readonly
SRC_DIR=$YENOMZE_HOME/workspace

docker run -t \
	-v $DATA:/data \
	-v $SRC_DIR:/workspace \
	-p 3311:8888 \
	-p 1001:6006 \
	yenomze:adipocyte nohup jupyter-lab --ip 0.0.0.0 > docker.log & 
