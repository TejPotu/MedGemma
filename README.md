# MedGemma Medical Case Analysis & Differential Diagnosis

A medical imaging AI framework using Google's MedGemma model for analyzing clinical case reports and multi-modal datasets (Chest X-Ray, Brain MRI) to perform structured data extraction and advanced differential diagnosis.

## 🚀 Overview

This project leverages **MedGemma** (google/medgemma-1.5-4b-it), a state-of-the-art medical vision-language model, to:
- **Structure Medical Data**: Extract clinical entities (symptoms, duration, findings) into a standardized schema from raw case reports.
- **Advanced Differential Diagnosis**: Use a **LangGraph-based** multi-step reasoning pipeline to generate, critique, and refine medical hypotheses.
- **Multi-Modal Analysis**: Process patient cases containing both clinical text and multiple medical images (X-Rays, MRIs).
- **Automated Evaluation**: Compare AI-generated findings against ground truth data from the MultiCaRe dataset.

## 📂 Project Structure

```text
medgemma/
├── database_schema_and_diagnosis/  # Core logic (Recently Organized)
│   ├── build_schema_dataset.py     # Parses raw datasets into structured JSON
│   ├── diagnosis_graph.py          # LangGraph implementation for Diff. Diagnosis
│   ├── transform_to_schema.py      # Schema transformation utilities
│   ├── schema.json                 # Target medical schema definition
│   └── diagnosis_notebook.ipynb    # Interactive playground for Graph Diagnosis
├── build_cxr_dataset.py            # Dataset builder for Chest X-Ray subset
├── diagnosis_graph.py              # Main graph-based reasoning engine
├── multicare_dataset.ipynb         # MultiCaRe dataset exploration
├── medgemma.ipynb                  # Base MedGemma experimentation
└── medical_datasets/               # Data storage
    ├── whole_multicare_dataset/    # Source MultiCaRe data
    └── cxr_schema_dataset/         # Processed structured dataset
```

## ✨ Key Features

### 🧠 Differential Diagnosis Pipeline (LangGraph)
The system uses a sophisticated agentic workflow:
1. **Initial Diagnosis**: MedGemma generates a ranked list of potential conditions based on clinical text + images.
2. **Bias Check**: A critical review step identifies cognitive shortcuts or "anchoring bias."
3. **Alternative Hypotheses**: The model is forced to consider lower-probability but critical "can't-miss" diagnoses.
4. **Final Synthesis**: Compiles a structured diagnostic report with evidence citations.

### 📋 Structured Schema Extraction
Converts messy, unstructured clinical text into a clean database format covering:
- Chief complaint & symptom duration
- Comorbidities & medications
- Detailed radiological findings (Pleural effusion, Cardiomegaly, etc.)
- Outcome details & ground truth confirmation

## 🛠️ Requirements & Setup

### Environment
- **Python**: 3.10+
- **GPU**: NVIDIA (16GB+ VRAM for 4b model, 40GB+ for larger variants)
- **Cuda**: 11.8+

### Installation
```bash
pip install torch transformers pillow pandas matplotlib langgraph
# Note: Ensure you have access to the MedGemma model on HuggingFace
export HF_TOKEN="your_token_here"
```

## 📖 Usage

### 1. Building the Structured Dataset
Transform the raw MultiCaRe case reports into the standardized schema:
```bash
python build_schema_dataset.py
```

### 2. Running Differential Diagnosis
You can run the full agentic diagnosis pipeline using the command line or the provided notebook:
```python
from diagnosis_graph import run_diagnosis

# Run diagnosis for a specific case ID
result = run_diagnosis("PMC1234567")
print(result['final_report'])
```

## 📊 Dataset & Model

- **Dataset**: [MultiCaRe (Multimodal Case Reports)](https://github.com/mauro-nievoff/MultiCaRe_Dataset), a collection of open-access PubMed Central reports.
- **Model**: **MedGemma 1.5 4B-IT**, Google's medical-tuned version of Gemma, optimized for clinical reasoning and image interpretation.

> ⚠️ **Disclaimer**: This tool is for **research purposes only**. It is NOT a medical device and should never be used for real-world clinical decision-making.

## 📜 License & Acknowledgments
- **MedGemma**: Subject to Google's Gemma Terms of Use.
- **Data**: PubMed Central / MultiCaRe contributors.
- **Developed by**: The MedGemma Research Team.
