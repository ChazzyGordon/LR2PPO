#!/bin/bash

# Validate input parameters
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <WEB10K_DIR> <MQ2008_DIR> <OUTPUT_DIR>"
    exit 1
fi

WEB10K_DIR=$1
MQ2008_DIR=$2
OUTPUT_DIR=$3

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Merge training data
cat "$WEB10K_DIR/train.tsv" "$MQ2008_DIR/train.tsv" > "$OUTPUT_DIR/train.tsv"

# Copy test data from target domain (MQ2008)
cp "$MQ2008_DIR/test.tsv" "$OUTPUT_DIR"

echo "Merged files:"
echo "- Training data: $OUTPUT_DIR/train.tsv"
echo "- Test data:    $OUTPUT_DIR/test.tsv"