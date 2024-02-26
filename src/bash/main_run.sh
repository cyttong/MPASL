#!/bin/bash


model_type=$1
dataset=$2
gpu=$3

if [ $model_type = "MPASL" ]
then 
    if [ $dataset = "synthetic" ]
	then
	bash mpasl_synthetic.sh $gpu
    else
	echo "Invalid dataset! Dataset should be 'synthetic'."
    fi
else 
    echo "Invalid model!"
    exit 1
fi
