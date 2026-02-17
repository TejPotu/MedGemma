#!/usr/bin/env python3
"""
Build complete chest X-ray dataset with organized structure.
Groups images by PMC article, creates JSON metadata, and organizes files.
"""

import json
import shutil
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import os

# Paths
DATASET_DIR = Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/whole_multicare_dataset")
OUTPUT_DIR = Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/chest_xray_dataset_final")
OUTPUT_JSON_DIR = OUTPUT_DIR / "json_files"
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("BUILDING CHEST X-RAY DATASET")
print("="*80)

# Load filtered chest X-rays
print("\nLoading filtered chest X-ray data...")
chest_xrays = pd.read_csv("chest_xrays_filtered.csv")
print(f"✓ Loaded {len(chest_xrays)} chest X-ray images")

# Load article metadata
print("Loading article metadata...")
cases_df = pq.read_table(DATASET_DIR / "cases.parquet").to_pandas()
print(f"✓ Loaded metadata for {len(cases_df)} articles")

# Group by PMC ID
print("\nGrouping images by PMC article...")
chest_xrays['pmc_id'] = chest_xrays['patient_id'].str.extract(r'(PMC\d+)', expand=False)
grouped = chest_xrays.groupby('pmc_id')
pmc_ids_with_chest = list(grouped.groups.keys())
print(f"✓ Found {len(pmc_ids_with_chest)} unique PMC articles")

# Build master index
master_index = {
    "dataset_name": "Chest X-ray Dataset from MultiCaRe",
    "description": "Organized collection of chest X-rays from PubMed Central articles",
    "total_articles": len(pmc_ids_with_chest),
    "total_images": len(chest_xrays),
    "creation_date": datetime.now().isoformat(),
    "articles": []
}

# Process each PMC article
print("\n" + "="*80)
print("PROCESSING ARTICLES AND CREATING JSON FILES")
print("="*80)

images_copied = 0
errors = []

for i, (pmc_id, group) in enumerate(grouped, 1):
    if i % 100 == 0:
        print(f"Processing article {i}/{len(pmc_ids_with_chest)}...")
    
    try:
        # Get article metadata
        article_data = cases_df[cases_df['case_id'] == pmc_id]
        
        # Create article JSON
        article_json = {
            "pmc_id": pmc_id,
            "total_chest_xrays": len(group),
            "chest_xray_images": [],
            "metadata": {}
        }
        
        # Add article metadata if available
        if len(article_data) > 0:
            row = article_data.iloc[0]
            article_json["metadata"] = {
                "article_title": row.get('case_title', ''),
                "published_date": str(row.get('publish_time', '')),
                "journal": row.get('journal', ''),
                "doi": row.get('doi', ''),
                "authors": row.get('authors', ''),
            }
        
        # Add image info
        for idx, (_, img_row) in enumerate(group.iterrows(), 1):
            img_info = {
                "index": idx,
                "filename": img_row['file'],
                "file_id": img_row['file_id'],
                "caption": img_row['caption'],
                "patient_id": img_row['patient_id'],
                "radiology_view": img_row['radiology_view'],
                "radiology_region": img_row['radiology_region'],
                "license": img_row.get('license', 'CC BY')
            }
            article_json["chest_xray_images"].append(img_info)
        
        # Create image directory for this article
        img_article_dir = OUTPUT_IMG_DIR / pmc_id
        img_article_dir.mkdir(exist_ok=True)
        
        # Copy images and build source paths
        for img_info in article_json["chest_xray_images"]:
            filename = img_info['filename']
            # Construct source path: f"{file[:4]}/{file[:5]}/{file}"
            src_path = DATASET_DIR / filename[:4] / filename[:5] / filename
            dst_path = img_article_dir / filename
            
            try:
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    img_info['local_path'] = f"images/{pmc_id}/{filename}"
                    images_copied += 1
                else:
                    img_info['local_path'] = f"images/{pmc_id}/{filename}"
                    errors.append(f"Source not found: {src_path}")
            except Exception as e:
                errors.append(f"Error copying {filename}: {str(e)}")
                img_info['local_path'] = f"images/{pmc_id}/{filename}"
        
        # Save article JSON
        json_path = OUTPUT_JSON_DIR / f"{pmc_id}.json"
        with open(json_path, 'w') as f:
            json.dump(article_json, f, indent=2)
        
        # Add to master index
        master_index["articles"].append({
            "pmc_id": pmc_id,
            "chest_xray_count": len(group),
            "json_file": f"json_files/{pmc_id}.json",
            "images_directory": f"images/{pmc_id}",
            "article_title": article_json["metadata"].get("article_title", ""),
        })
        
    except Exception as e:
        errors.append(f"Error processing {pmc_id}: {str(e)}")

# Save master index
print("\nSaving master index...")
master_index_path = OUTPUT_DIR / "dataset_index.json"
with open(master_index_path, 'w') as f:
    json.dump(master_index, f, indent=2)

# Create README
readme_content = f"""# Chest X-ray Dataset from MultiCaRe

## Overview
This dataset contains **{len(chest_xrays)}** chest X-ray images from **{len(pmc_ids_with_chest)}** PubMed Central articles.

## Dataset Structure
```
chest_xray_dataset_final/
├── dataset_index.json          # Master index of all articles
├── README.md                   # This file
├── json_files/
│   ├── PMC10009052.json       # Metadata for each PMC article
│   ├── PMC10010120.json
│   └── ...
└── images/
    ├── PMC10009052/           # Images organized by PMC ID
    ├── PMC10010120/
    └── ...
```

## File Formats
- **JSON files**: Each PMC article has a JSON file containing:
  - Article metadata (title, journal, authors, DOI)
  - List of chest X-rays with metadata
  - Image file paths and captions
  - Radiology information (view, region)

- **Images**: Chest X-ray images in WebP format (.webp)

## Sample JSON Structure
```json
{{
  "pmc_id": "PMC10009052",
  "total_chest_xrays": 2,
  "metadata": {{
    "article_title": "Acute respiratory distress syndrome",
    "journal": "Example Journal",
    "published_date": "2023-01-15"
  }},
  "chest_xray_images": [
    {{
      "filename": "PMC10009052_gr1_undivided_1_1.webp",
      "caption": "Chest X-ray posterior-anterior view...",
      "radiology_view": "frontal",
      "radiology_region": "thorax",
      "local_path": "images/PMC10009052/PMC10009052_gr1_undivided_1_1.webp"
    }}
  ]
}}
```

## Usage
1. Load `dataset_index.json` for a quick overview of all articles
2. Access individual article metadata via JSON files in `json_files/`
3. Load images from corresponding `images/PMCXXXX/` directories
4. Use captions and metadata for image-text pairing tasks

## Dataset Statistics
- Total Articles: {len(pmc_ids_with_chest)}
- Total X-ray Images: {len(chest_xrays)}
- Average Images per Article: {len(chest_xrays) / len(pmc_ids_with_chest):.2f}
- Images Successfully Copied: {images_copied}
- Common X-ray Views: frontal, sagittal, oblique

## Image Information
Each chest X-ray includes:
- Patient ID (pseudonymized within article)
- Radiology view (e.g., frontal, sagittal, oblique)
- Radiology region (thorax, other body regions)
- Image caption/description from the article
- License information (typically CC BY)

## Creation Date
{datetime.now().isoformat()}

## License
Images are from PubMed Central articles and follow the source article licenses (typically CC BY).
"""

with open(OUTPUT_DIR / "README.md", 'w') as f:
    f.write(readme_content)

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Dataset saved to: {OUTPUT_DIR}")
print(f"✓ JSON files saved: {len(pmc_ids_with_chest)}")
print(f"✓ Images copied: {images_copied}/{len(chest_xrays)}")
print(f"✓ Master index: {master_index_path}")
print(f"✓ README: {OUTPUT_DIR / 'README.md'}")

if errors:
    print(f"\n⚠ Warnings/Errors ({len(errors)}):")
    for err in errors[:10]:
        print(f"  - {err}")

print("\n" + "="*80)
print("DATASET READY FOR USE!")
print("="*80)
