#!/usr/bin/env python3
"""
Generate JSON metadata files for the chest X-ray dataset.
One JSON file per PMC article with all images and metadata.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
DATASET_DIR = Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/whole_multicare_dataset")
OUTPUT_DIR = Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/chest_xray_dataset_final")
OUTPUT_JSON_DIR = OUTPUT_DIR / "json_files"
OUTPUT_JSON_DIR.mkdir(exist_ok=True)

print("="*80)
print("GENERATING JSON METADATA FILES")
print("="*80)

# Load filtered chest X-rays
print("\nLoading chest X-ray data...")
chest_xrays = pd.read_csv("chest_xrays_filtered.csv")
chest_xrays['pmc_id'] = chest_xrays['patient_id'].str.extract(r'(PMC\d+)', expand=False)

# Group by PMC article
grouped = chest_xrays.groupby('pmc_id')
print(f"✓ Found {len(grouped)} unique PMC articles")

# Create JSON for each article
print("\nGenerating JSON files...")
for i, (pmc_id, group) in enumerate(grouped, 1):
    if i % 200 == 0:
        print(f"Processing article {i}/{len(grouped)}...")
    
    # Create article JSON
    article_json = {
        "pmc_id": pmc_id,
        "article_metadata": {
            "title": "",
            "journal": "",
            "year": "",
            "doi": "",
            "authors": []
        },
        "total_chest_xrays": len(group),
        "chest_xray_images": []
    }
    
    # Add each image
    for idx, (_, img_row) in enumerate(group.iterrows(), 1):
        img_info = {
            "index": idx,
            "filename": img_row['file'],
            "file_id": img_row['file_id'],
            "caption": img_row['caption'],
            "patient_id_article": img_row['patient_id'],
            "radiology_view": img_row['radiology_view'] if pd.notna(img_row['radiology_view']) else "",
            "radiology_region": img_row['radiology_region'] if pd.notna(img_row['radiology_region']) else "",
            "image_type": img_row['image_type'] if pd.notna(img_row['image_type']) else "",
            "image_subtype": img_row['image_subtype'] if pd.notna(img_row['image_subtype']) else "",
            "license": img_row['license'] if pd.notna(img_row['license']) else "CC BY",
            "local_path":f"images/{pmc_id}/{img_row['file']}"
        }
        article_json["chest_xray_images"].append(img_info)
    
    # Save JSON file
    json_path = OUTPUT_JSON_DIR / f"{pmc_id}.json"
    with open(json_path, 'w') as f:
        json.dump(article_json, f, indent=2)

print(f"\n✓ Generated {len(grouped)} JSON files")

# Create/update master index
print("\nUpdating master index...")
master_index = {
    "dataset_name": "Chest X-ray Dataset from MultiCaRe",
    "description": "Organized collection of chest X-rays from PubMed Central articles with metadata",
    "dataset_version": "1.0",
    "total_articles": len(grouped),
    "total_chest_xray_images": len(chest_xrays),
    "average_images_per_article": round(len(chest_xrays) / len(grouped), 2),
    "creation_date": "2026-02-16",
    "structure": {
        "json_files": "Directory containing one JSON file per PMC article (PMCXXXXXX.json)",
        "images": "Directory organized as images/PMCXXXXXX/ with all chest X-rays for each article"
    },
    "articles": []
}

# Add article metadata to master index
for pmc_id, group in grouped:
    master_index["articles"].append({
        "pmc_id": pmc_id,
        "total_chest_xrays": len(group),
        "json_file": f"json_files/{pmc_id}.json",
        "images_directory": f"images/{pmc_id}"
    })

# Save master index
index_path = OUTPUT_DIR / "dataset_index.json"
with open(index_path, 'w') as f:
    json.dump(master_index, f, indent=2)

print(f"✓ Master index updated: {index_path}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Total PMC articles: {len(grouped)}")
print(f"✓ Total chest X-ray images: {len(chest_xrays)}")
print(f"✓ Average images per article: {len(chest_xrays) / len(grouped):.2f}")
print(f"✓ JSON metadata files created: {len(list(OUTPUT_JSON_DIR.glob('*.json')))}")
print(f"\nDataset location: {OUTPUT_DIR}")
