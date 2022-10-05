#!/bin/bash

DEVICE_TYPE='gpu'
GPU_NUMBER=0
MODEL_INDEX='1'

MODEL_PATH='models/'
DATA_FOLDER='sample_data/images'
INITIAL_EXAM_LIST_PATH='sample_data/exam_list_before_cropping.pkl'
CROPPED_IMAGE_PATH='sample_output/cropped_images'
CROPPED_EXAM_LIST_PATH='sample_output/cropped_images/cropped_exam_list.pkl'
EXAM_LIST_PATH='sample_output/data.pkl'
OUTPUT_PATH='sample_output'
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ ! -d $DATA_FOLDER ]]; then
  echo 'Image folder not found; please download and untar the sample data as described in README'
  exit 1
fi

if [[ -d $CROPPED_IMAGE_PATH ]]; then
  echo 'Cropped image path already exists. Please remove the previous output folder when re-running the entire pipeline.'
  exit 1
fi

echo 'Stage 1: Crop Mammograms'
python3 src/cropping/crop_mammogram.py \
    --input-data-folder $DATA_FOLDER \
    --output-data-folder $CROPPED_IMAGE_PATH \
    --exam-list-path $INITIAL_EXAM_LIST_PATH  \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH

echo 'Stage 2: Extract Centers'
python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --data-prefix $CROPPED_IMAGE_PATH \
    --output-exam-list-path $EXAM_LIST_PATH

echo 'Stage 3: Run Classifier'
python3 src/scripts/run_model.py \
    --model-path $MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $OUTPUT_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --model-index $MODEL_INDEX \
    #--visualization-flag

