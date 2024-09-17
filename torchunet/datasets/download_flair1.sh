#!/bin/bash

# Define the base URL for the FLAIR #1 datasets
BASE_URL="https://ignf.github.io/FLAIR/datasets"

# Define the filenames for the training and testing datasets
TRAINING_DATASET="FLAIR1_train.zip"
TESTING_DATASET="FLAIR1_test.zip"

# Define the directory where the datasets will be saved
DATA_DIR="./FLAIR_Datasets"

# Create the directory if it doesn't exist
mkdir -p $DATA_DIR

# Download the training dataset
echo "Downloading the FLAIR #1 training dataset..."
wget -c "$BASE_URL/$TRAINING_DATASET" -O "$DATA_DIR/$TRAINING_DATASET"

# Download the testing dataset
echo "Downloading the FLAIR #1 testing dataset..."
wget -c "$BASE_URL/$TESTING_DATASET" -O "$DATA_DIR/$TESTING_DATASET"

# Unzip the training dataset
echo "Unzipping the training dataset..."
unzip "$DATA_DIR/$TRAINING_DATASET" -d $DATA_DIR

# Unzip the testing dataset
echo "Unzipping the testing dataset..."
unzip "$DATA_DIR/$TESTING_DATASET" -d $DATA_DIR

echo "Download and extraction complete!"
