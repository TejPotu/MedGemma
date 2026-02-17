# Quick Start Guide - Chest X-Ray Dataset

## 🚀 Getting Started

Your chest X-ray dataset is ready at:
```
/blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/chest_xray_dataset_final/
```

## 📋 What You Have

- **2,235 chest X-ray images** from **1,553 unique PMC articles**
- **1,553 JSON metadata files** - one per article with complete image information
- **Master index** - quick lookup of all articles
- **Organized folder structure** - images grouped by PMC article ID

## 🔍 Quick Exploration

### View the Master Index
```bash
cd /blue/gtyson.fsu/tp22o.fsu/medgemma/medical_datasets/chest_xray_dataset_final
cat dataset_index.json | head -30
```

### List an Article's Metadata
```bash
cat json_files/PMC10009052.json | python -m json.tool
```

### See Available Images
```bash
ls -lh images/PMC10009052/
```

## 💻 Python Usage Examples

### 1. Load All Articles
```python
import json

# Load master index
with open('dataset_index.json', 'r') as f:
    data = json.load(f)

print(f"Total articles: {data['total_articles']}")
print(f"Total images: {data['total_chest_xray_images']}")

# List first 5 articles
for article in data['articles'][:5]:
    print(f"{article['pmc_id']}: {article['total_chest_xrays']} X-rays")
```

### 2. Load a Specific Article
```python
import json

pmc_id = "PMC10009052"

with open(f'json_files/{pmc_id}.json', 'r') as f:
    article = json.load(f)

print(f"Article: {article['pmc_id']}")
print(f"Chest X-rays: {article['total_chest_xrays']}")

for xray in article['chest_xray_images']:
    print(f"\n  📷 {xray['filename']}")
    print(f"     Caption: {xray['caption'][:80]}...")
    print(f"     View: {xray['radiology_view']}")
    print(f"     Region: {xray['radiology_region']}")
```

### 3. Load Images with PIL
```python
from PIL import Image
import json

pmc_id = "PMC10009052"

# Get metadata
with open(f'json_files/{pmc_id}.json', 'r') as f:
    article = json.load(f)

# Load and display images
for xray in article['chest_xray_images']:
    img_path = xray['local_path']
    img = Image.open(img_path)
    print(f"Loaded {xray['filename']}: {img.size}")
```

### 4. Create a DataFrame of All Images
```python
import json
import pandas as pd
from pathlib import Path

# Collect all image data
all_images = []

for json_file in Path('json_files').glob('*.json'):
    with open(json_file) as f:
        article = json.load(f)
    
    for xray in article['chest_xray_images']:
        all_images.append({
            'pmc_id': article['pmc_id'],
            'filename': xray['filename'],
            'caption': xray['caption'],
            'view': xray['radiology_view'],
            'region': xray['radiology_region'],
            'path': xray['local_path']
        })

df = pd.DataFrame(all_images)

# Analyze the data
print(f"Total images: {len(df)}")
print(f"\nMost common views:")
print(df['view'].value_counts())
print(f"\nImages per article:")
print(df.groupby('pmc_id').size().describe())
```

### 5. Batch Image Processing
```python
from PIL import Image
import json
from pathlib import Path

# Process all images
for json_file in Path('json_files').glob('*.json'):
    with open(json_file) as f:
        article = json.load(f)
    
    pmc_id = article['pmc_id']
    
    for xray in article['chest_xray_images']:
        img_path = xray['local_path']
        
        # Your processing here
        try:
            img = Image.open(img_path)
            # Resize, normalize, augment, etc.
            processed = img.resize((256, 256))
            
            print(f"✓ Processed {pmc_id}/{xray['filename']}")
        except Exception as e:
            print(f"✗ Error processing {img_path}: {e}")
```

## 📊 Dataset Statistics

Check view distribution:
```python
import json
from pathlib import Path
from collections import Counter

views = []
regions = []

for json_file in Path('json_files').glob('*.json'):
    with open(json_file) as f:
        article = json.load(f)
    
    for xray in article['chest_xray_images']:
        if xray['radiology_view']:
            views.append(xray['radiology_view'])
        if xray['radiology_region']:
            regions.append(xray['radiology_region'])

print("X-ray Views:")
for view, count in Counter(views).most_common():
    print(f"  {view}: {count}")

print("\nRadiology Regions:")
for region, count in Counter(regions).most_common():
    print(f"  {region}: {count}")
```

## 🎯 Common Tasks

### Extract All Captions
```python
import json
from pathlib import Path

captions = []
for json_file in Path('json_files').glob('*.json'):
    with open(json_file) as f:
        article = json.load(f)
    for xray in article['chest_xray_images']:
        captions.append(xray['caption'])

# Save to file
with open('all_captions.txt', 'w') as f:
    for caption in captions:
        f.write(caption + '\n')
```

### Filter Specific Views
```python
import json
from pathlib import Path

frontal_xrays = []

for json_file in Path('json_files').glob('*.json'):
    with open(json_file) as f:
        article = json.load(f)
    
    for xray in article['chest_xray_images']:
        if xray['radiology_view'] == 'frontal':
            frontal_xrays.append({
                'pmc_id': article['pmc_id'],
                'filename': xray['filename'],
                'path': xray['local_path']
            })

print(f"Found {len(frontal_xrays)} frontal chest X-rays")
```

### Create Training/Test Split
```python
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Collect all image paths
all_images = []

for json_file in Path('json_files').glob('*.json'):
    with open(json_file) as f:
        article = json.load(f)
    
    for xray in article['chest_xray_images']:
        all_images.append(xray['local_path'])

# Split 80/20
train, test = train_test_split(all_images, test_size=0.2, random_state=42)

print(f"Training: {len(train)}")
print(f"Testing: {len(test)}")

# Save splits
with open('train_paths.txt', 'w') as f:
    f.write('\n'.join(train))

with open('test_paths.txt', 'w') as f:
    f.write('\n'.join(test))
```

## 📁 File Organization

```
chest_xray_dataset_final/
├── dataset_index.json           # Master index
├── README.md                    # Full documentation
├── json_files/
│   ├── PMC10009052.json        # Article 1
│   ├── PMC10010120.json        # Article 2
│   └── PMC10012974.json        # Article 3
└── images/
    ├── PMC10009052/
    │   └── PMC10009052_gr1_undivided_1_1.webp
    ├── PMC10010120/
    │   ├── image1.webp
    │   └── image2.webp
    └── PMC10012974/
        └── image.webp
```

## ❓ Troubleshooting

**Issue**: Images not found
- Check the image path in the JSON file matches the actual file location
- Use `Path('images/PMC10009052').exists()` to verify the directory exists

**Issue**: JSON parsing errors
- Ensure you're opening JSON files in the correct directory
- Use `json.JSONDecodeError` exception handling

**Issue**: Memory issues with large batch processing
- Process images one article at a time instead of loading all
- Use generators instead of loading all into memory

## 🔗 Next Steps

1. **Explore the data** - Start with a few sample articles
2. **Preprocess images** - Normalize, resize, augment as needed
3. **Build models** - Use for classification, detection, segmentation
4. **Share results** - Document your findings and results

## 📚 Related Files

- Main documentation: `medical_datasets/chest_xray_dataset_final/README.md`
- Dataset summary: `DATASET_SUMMARY.md`
- Source data: `medical_datasets/whole_multicare_dataset/`

---

**Ready to get started? Load the first article and explore!** 🎉
