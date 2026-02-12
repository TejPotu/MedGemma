# MedGemma Medical Image Analysis

A medical imaging AI project using Google's MedGemma model for analyzing brain MRI scans and predicting diagnoses from the MultiCaRe dataset.

## Overview

This project demonstrates the use of MedGemma (google/medgemma-1.5-4b-it), a medical vision-language model, to:
- Analyze multiple brain MRI images from a single patient case
- Generate differential diagnoses based on radiological findings
- Compare AI-generated diagnoses with ground truth from case reports
- Evaluate model performance on medical image interpretation tasks

## Project Structure

```
medgemma/
├── README.md
├── medgemma.ipynb              # Main MedGemma experimentation notebook
├── multicare_dataset.ipynb     # MultiCaRe dataset analysis with MedGemma
└── medical_datasets/
    ├── pxa_test_set/           # Filtered dataset for PXA/glioma cases
    │   ├── cases.csv           # Patient case information
    │   ├── image_metadata.json # Image metadata and file paths
    │   ├── article_metadata.json
    │   ├── case_report_citations.json
    │   ├── readme.txt
    │   └── images/             # MRI images organized by PMC ID
    │       └── PMC*/
    └── whole_multicare_dataset/
        ├── captions_and_labels.csv
        ├── data_dictionary.csv
        └── PMC*/               # Full dataset images
```

## Features

### Multi-Image Analysis
The system analyzes **all available images** for a single patient case, providing:
- Image-by-image findings
- Integrated analysis across all images
- Key radiological features (signal characteristics, lesion location, mass effect, etc.)
- Differential diagnosis ranked by probability
- Recommended next steps

### Model Self-Evaluation
Automated comparison of AI predictions against ground truth case reports, evaluating:
- Diagnosis accuracy
- Radiological findings comparison
- Clinical reasoning assessment
- Overall score with justification

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- PIL (Pillow)
- Pandas
- Matplotlib
- multiversity (for MultiCaRe dataset creation)

## Installation

```bash
pip install transformers torch pillow pandas matplotlib
pip install multiversity  # For MultiCaRe dataset handling
```

## Usage

### 1. Create a Filtered Dataset

```python
from multiversity.multicare_dataset import MedicalDatasetCreator

mdc = MedicalDatasetCreator(directory='medical_datasets')
filters = [
    {'field': 'case_strings', 'string_list': ['PXA', 'xanthoastrocytoma', 'glioma'], 'operator': 'any'},
    {'field': 'label', 'string_list': ['mri', 'head']}
]
mdc.create_dataset(dataset_name='pxa_test_set', filter_list=filters, dataset_type='multimodal')
```

### 2. Load MedGemma Model

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
```

### 3. Run Multi-Image Diagnosis

```python
# Load all images for a case
loaded_images = [...]  # List of {'image': PIL.Image, 'caption': str, 'view': str}

# Generate diagnosis
prediction = predict_diagnosis_multi_image(loaded_images, patient_info)

# Compare with ground truth
comparison = compare_diagnosis_with_gt(prediction, case_info['case_text'], patient_info)
```

## Dataset

This project uses the **MultiCaRe** (Multimodal Case Reports) dataset, which contains:
- Medical case reports with patient information
- Associated medical images (MRI, CT, X-ray, etc.)
- Image captions and metadata
- Ground truth diagnoses from published case reports

The `pxa_test_set` is a filtered subset focusing on:
- Pleomorphic Xanthoastrocytoma (PXA)
- Glioma-related cases
- Head/brain MRI images

## Model

**MedGemma 1.5 4B-IT** is a medical vision-language model from Google designed for:
- Medical image understanding
- Clinical reasoning
- Diagnostic assistance

> ⚠️ **Disclaimer**: This model is for research purposes only and should not be used for actual clinical diagnosis.

## Notebooks

### `multicare_dataset.ipynb`
Main analysis notebook that:
1. Loads and filters the MultiCaRe dataset
2. Selects brain MRI cases
3. Loads all images for a patient case
4. Runs multi-image diagnosis prediction
5. Compares predictions with ground truth
6. Evaluates model performance

### `medgemma.ipynb`
Additional experimentation with the MedGemma model.

## License

This project is for research and educational purposes. Please refer to:
- [MedGemma Model License](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MultiCaRe Dataset License](https://github.com/mauro-nievoff/MultiCaRe_Dataset)

## Acknowledgments

- Google Health AI for the MedGemma model
- MultiCaRe dataset creators
- PubMed Central for open-access medical case reports
