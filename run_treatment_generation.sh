#!/bin/bash
# Batch processing script for generating treatment plans
# This script processes the dataset in batches with checkpoint support

set -e  # Exit on error

# Configuration
INPUT_FILE="/blue/gtyson.fsu/tp22o.fsu/medgemma/cxr_schema_dataset/dataset_cxr_primary.json"
OUTPUT_FILE="/blue/gtyson.fsu/tp22o.fsu/medgemma/cxr_schema_dataset/dataset_cxr_with_treatment.json"
CHECKPOINT_FILE="treatment_checkpoint.json"
BATCH_SIZE=50  # Process 50 cases at a time

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Treatment Plan Generation - Batch Mode${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${GREEN}Input:${NC} $INPUT_FILE"
echo -e "${GREEN}Output:${NC} $OUTPUT_FILE"
echo -e "${GREEN}Checkpoint:${NC} $CHECKPOINT_FILE"
echo -e "${GREEN}Batch size:${NC} $BATCH_SIZE cases"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Check if Python script exists
if [ ! -f "generate_treatment_plans.py" ]; then
    echo -e "${RED}Error: generate_treatment_plans.py not found${NC}"
    exit 1
fi

# Ask for confirmation
read -p "Start processing? This will use GPU resources. (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Aborted by user${NC}"
    exit 0
fi

# Run with batch processing
echo -e "\n${BLUE}Starting batch processing...${NC}\n"

python3 generate_treatment_plans.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --checkpoint "$CHECKPOINT_FILE" \
    --max-cases "$BATCH_SIZE"

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Processing completed successfully!${NC}"
    echo -e "${GREEN}Output saved to: $OUTPUT_FILE${NC}"
    
    # Show file size
    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo -e "${GREEN}File size: $SIZE${NC}"
    fi
else
    echo -e "\n${RED}✗ Processing failed!${NC}"
    echo -e "${YELLOW}Check the log file: treatment_generation.log${NC}"
    exit 1
fi
