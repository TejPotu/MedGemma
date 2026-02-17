#!/usr/bin/env python3
"""
Collect all chest X-rays from MultiCaRe dataset and organize them.
Creates a structured dataset with JSON metadata and organized images.
"""

import json
import os
import shutil
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Paths
DATASET_DIR = Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/whole_multicare_dataset")
OUTPUT_DIR = Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/chest_xray_dataset_organized")
OUTPUT_JSON_DIR = OUTPUT_DIR / "json_files"
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"

# Create directories
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset files...")

# Load the captions and labels data
try:
    captions_df = pd.read_csv(DATASET_DIR / "captions_and_labels.csv")
    print(f"Loaded {len(captions_df)} image records")
except Exception as e:
    print(f"Error loading captions: {e}")
    exit(1)

# Load case_images to get more metadata
try:
    case_images = pq.read_table(DATASET_DIR / "case_images.parquet").to_pandas()
    print(f"Loaded {len(case_images)} case image records")
except Exception as e:
    print(f"Warning: Could not load case_images.parquet: {e}")
    case_images = None

# Load cases.parquet for article metadata
try:
    cases_df = pq.read_table(DATASET_DIR / "cases.parquet").to_pandas()
    print(f"Loaded {len(cases_df)} case records")
except Exception as e:
    print(f"Warning: Could not load cases.parquet: {e}")
    cases_df = None

# Strategy: Find all X-ray images
xray_mask = captions_df['image_subtype'] == 'x_ray'
all_xrays = captions_df[xray_mask].copy()
print(f"\n✓ Found {len(all_xrays)} total X-ray images")

# Filter for chest X-rays by examining captions
def is_chest_xray(row):
    """Check if an X-ray is a chest X-ray by examining caption and other fields"""
    caption = str(row.get('caption', '')).lower()
    chest_keywords = ['chest', 'thorax', 'thoracic', 'lung', 'pulmonary', 'pneumonia', 
                      'atelectasis', 'infiltrate', 'edema', 'ptx', 'cxr']
    return any(kw in caption for kw in chest_keywords)

# Apply chest X-ray filter
all_xrays['is_chest'] = all_xrays.apply(is_chest_xray, axis=1)
chest_xrays = all_xrays[all_xrays['is_chest']].copy()

print(f"✓ Chest X-rays identified: {len(chest_xrays)} images")

# Also check radiology_view if available
print(f"\nRadiology views in X-rays:")
print(all_xrays['radiology_view'].value_counts())

print(f"\nRadiology regions in X-rays:")
print(all_xrays['radiology_region'].value_counts())

# Extract PMC IDs
pmc_ids = chest_xrays['patient_id'].str.extract(r'(PMC\d+)', expand=False).unique()
print(f"\n✓ Chest X-rays come from {len(pmc_ids)} unique PMC articles")

print("\nSample chest X-ray records:")
print(chest_xrays[['file', 'patient_id', 'radiology_view', 'radiology_region', 'caption']].head(10))

# Save detailed overview
overview = {
    "total_xrays": len(all_xrays),
    "total_chest_xrays": len(chest_xrays),
    "unique_articles_with_chest_xrays": len(pmc_ids),
    "views": chest_xrays['radiology_view'].dropna().unique().tolist()[:20],
    "sample_records": chest_xrays[['file', 'patient_id', 'radiology_view', 'caption']].head(5).to_dict(orient='records')
}

print("\n" + "="*80)
print(f"SUMMARY: Found {len(chest_xrays)} chest X-ray images from {len(pmc_ids)} PMC articles")
print("="*80)

with open("chest_xrays_overview.json", "w") as f:
    json.dump(overview, f, indent=2)

print("Overview saved to chest_xrays_overview.json")

# Save the filtered dataframe for next step
chest_xrays.to_csv("chest_xrays_filtered.csv", index=False)
print("Filtered chest X-rays saved to chest_xrays_filtered.csv")
