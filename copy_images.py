#!/usr/bin/env python3
"""
Copy images for the chest X-ray dataset.
"""

import shutil
import pandas as pd
from pathlib import Path

# Paths
DATASET_DIR = Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/whole_multicare_dataset")
OUTPUT_IMG_DIR = Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/chest_xray_dataset_final/images")

print("="*80)
print("COPYING CHEST X-RAY IMAGES")
print("="*80)

# Load filtered chest X-rays
print("\nLoading chest X-ray list...")
chest_xrays = pd.read_csv("chest_xrays_filtered.csv")
chest_xrays['pmc_id'] = chest_xrays['patient_id'].str.extract(r'(PMC\d+)', expand=False)

# Group by PMC ID
grouped = chest_xrays.groupby('pmc_id')

print(f"Found {len(grouped)} PMC articles to process\n")

images_copied = 0
images_failed = 0
errors = []

for i, (pmc_id, group) in enumerate(grouped, 1):
    if i % 100 == 0:
        print(f"Processing article {i}/{len(grouped)}... ({images_copied} images copied)")
    
    # Create image directory for this article
    img_article_dir = OUTPUT_IMG_DIR / pmc_id
    img_article_dir.mkdir(parents=True, exist_ok=True)
    
    for _, img_row in group.iterrows():
        filename = img_row['file']
        # Construct source path: f"{file[:4]}/{file[:5]}/{file}"
        src_path = DATASET_DIR / filename[:4] / filename[:5] / filename
        dst_path = img_article_dir / filename
        
        try:
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                images_copied += 1
            else:
                errors.append(f"Source not found: {src_path}")
                images_failed += 1
        except Exception as e:
            errors.append(f"Error copying {filename}: {str(e)}")
            images_failed += 1

# Print summary
print("\n" + "="*80)
print("COPY SUMMARY")
print("="*80)
print(f"✓ Images successfully copied: {images_copied}")
print(f"✗ Images failed: {images_failed}")

if errors:
    print(f"\nShowing first 10 errors:")
    for err in errors[:10]:
        print(f"  - {err}")

print("\n✓ Dataset image organization complete!")
