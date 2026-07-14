# MMM Literature Mining Pipeline

An end-to-end scientific literature mining pipeline for automatically extracting, refining, verifying, and transforming experimental information from **Mixed Matrix Membrane (MMM)** publications into structured, machine-learning-ready datasets.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

---

## Overview

Mixed Matrix Membrane (MMM) research has expanded rapidly in recent years, resulting in thousands of scientific publications containing valuable experimental information. However, these data are reported in different formats and are scattered throughout articles, making manual extraction slow, inconsistent, and difficult to scale.

This project introduces an automated literature mining pipeline that combines **Large Language Models (LLMs)** with data refinement, semantic verification, and feature engineering to transform unstructured scientific literature into structured datasets suitable for downstream analysis and machine learning.

The pipeline focuses on extracting experimentally reported information such as polymer matrices, filler materials, operating conditions, membrane performance, and polymer–filler compatibility.

---

# Pipeline

```text
Scientific Publications (PDFs)
            │
            ▼
      Text Chunk Generation
            │
            ▼
    LLM-Based Information Extraction
            │
            ▼
        Data Refinement
            │
            ▼
     Semantic Verification
            │
            ▼
     Verified Experimental Data
            │
            ▼
      Feature Engineering
            │
            ▼
 Machine-Learning-Ready Dataset
```

---

# Main Features

- Automated extraction of structured scientific information
- GPT-based experimental data extraction
- Automatic DOI and title recovery
- Duplicate removal and data refinement
- Information quality scoring
- Semantic verification using an independent LLM
- Polymer and filler normalization
- Scientific feature engineering
- Machine-learning-ready dataset generation
- Modular multi-stage architecture
- Automatic logging and checkpoint support

---

# Extracted Information

The extraction stage identifies experimental information including:

- DOI
- Article title
- Polymer matrix
- Filler material
- Polymer modification
- Filler modification
- Gas type
- Application type
- Temperature
- Pressure
- Filler loading
- Gas permeability
- Selectivity
- Polymer–filler compatibility
- Experimental observations

---

# Repository Structure

```text
MMM-Literature-Mining-Pipeline/
│
├── pipeline/
│   ├── 01_llm_extraction.py
│   ├── 02_data_refinement.py
│   ├── 03_semantic_verification.py
│   └── 04_feature_engineering.py
│
├── outputs/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .env.example
```

---

# Pipeline Stages

## Stage 1 — LLM Information Extraction

The first stage processes article text chunks using an OpenAI model to extract structured experimental information.

Main tasks:

- Read article chunks
- Extract experimental conditions
- Standardize numerical values
- Recover missing DOI/title information
- Produce JSON and CSV outputs
- Handle retries and logging automatically

---

## Stage 2 — Data Refinement

The extracted dataset is cleaned and standardized.

Operations include:

- Removing duplicate records
- Filtering incomplete entries
- Calculating an information score
- Separating informative and low-quality records
- Standardizing missing values

---

## Stage 3 — Semantic Verification

A second language model independently validates extracted information against the original source text.

Outputs include:

- Field-level verification
- Confidence scores
- Weighted verification scores
- Verification reports

---

## Stage 4 — Feature Engineering

The verified dataset is transformed into a machine-learning-ready format.

This stage performs:

- Polymer normalization
- Filler normalization
- Polymer family classification
- Filler family classification
- Additive identification
- Modification strategy detection
- Numeric feature generation
- Log transformations
- Missing-value indicators
- Performance-related features

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/MMM-Literature-Mining-Pipeline.git
cd MMM-Literature-Mining-Pipeline
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```powershell
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a local `.env` file in the repository root.

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Never upload your API keys to GitHub.

---

# Usage

Run each stage sequentially.

```bash
python pipeline/01_llm_extraction.py
```

```bash
python pipeline/02_data_refinement.py
```

```bash
python pipeline/03_semantic_verification.py
```

```bash
python pipeline/04_feature_engineering.py
```

---

# Technologies

- Python
- OpenAI API
- Anthropic API
- Pandas
- NumPy
- Scientific NLP
- JSON
- CSV

---

# Current Limitations

- The pipeline requires external LLM APIs.
- Scientific PDFs are **not included** because of copyright restrictions.
- The current version focuses on structured data extraction and preparation.
- Model performance depends on the quality of the source text.

---

# Future Work

Planned improvements include:

- Automated PDF downloading
- PDF-to-text preprocessing
- Interactive dashboard
- Retrieval-Augmented Generation (RAG)
- Knowledge graph generation
- Domain-specific model fine-tuning
- Machine learning model training
- Web interface

---

# Citation

If you use this project in your research, please cite the repository.

```text
Sevda Elif Sayın

MMM Literature Mining Pipeline

2026
```

---

# Author

**Sevda Elif Sayın**

Biomedical Engineering

---

# License

This project is distributed under the MIT License.