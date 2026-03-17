# ADR Detect: Comparing LLM Generalization with Embedding-based Models for Adverse Drug Reaction Detection

**Authors**: Nicole Poliak & Naveh Nissan

---

## Overview

Adverse Drug Reactions (ADRs) are a major cause of patient harm and hospitalization, yet they are often buried in unstructured clinical notes or patient-written reviews. This project explores automated ADR detection from such texts using two different natural language processing (NLP) paradigms:

- **Zero-/Few-shot Large Language Models (LLMs)**  
- **Sentence Embeddings + Logistic Regression Classifier**

Our aim is to compare their effectiveness in identifying ADRs and evaluate their suitability for scalable, AI-driven medical decision support.

---
## Repository Contents

- `data_preparation.ipynb` – Dataset preprocessing and merging (ADE + PsyTAR)
- `combined_dataset.csv` – Final combined dataset (ADE + PsyTAR)
- `adr_classification_pipeline.ipynb` – Main pipeline: EDA, baseline models, embedding-based models, zero-/few-shot LLMs, evaluation
- `adr_project_presentation.pdf` – Presentation slides
- `overall_results.csv` – Table of final classification scores (accuracy, precision, recall, F1)
- `graphical_abstract.png` – Visual summary of the project
- `adr_graph_results.pdf` – Visual comparison of model performance
  
---

## Requirements for Getting Started

- A GPU runtime (e.g., Google Colab with GPU or local machine with CUDA)
- Access to:
  - An **Azure OpenAI endpoint** + **API key** (for GPT-4o, GPT-4o-mini and Phi-4-mini-instruct)
  - A **Hugging Face token** (for LLaMA-3.2-3B-Instruct)
 
---

## Task Definition

- **Input**: A single sentence from a biomedical abstract or patient review  
- **Output**: Binary label  
  - `1`: ADR present  
  - `0`: No ADR present  
- **Task**: Sentence-level binary classification (ADR vs. Non-ADR)

---

## Datasets

We used two publicly available datasets:

| Dataset        | Description                                      | Size     |
|----------------|--------------------------------------------------|----------|
| **ADE Corpus V2** | Contains expert-annotated PubMed sentences for classifying whether a sentence describes an Adverse Drug Event (ADE) and extracting the relation between the drug and the adverse event | 23,516   |
| **PsyTAR**     | Contains patient-written drug reviews from AskAPatient, capturing reported experiences and adverse reactions to psychiatric medications   | 6,009    |
| **Combined**   | Merged, cleaned, deduplicated dataset             | 26,867 → 12,862 after downsampling |

Each entry has:
- `text`: the sentence
- `label`: binary (1 = ADR, 0 = Non-ADR)
- `dataset`: source origin (ADE or PsyTAR)

> Due to class imbalance (~24% ADR), downsampling was applied to the majority class.

This is a dataset for Classification if a sentence is ADE-related (True) or not (False) and Relation Extraction between Adverse Drug Event and Drug. 

---

## Exploratory Data Analysis (EDA)

- **Final size after balancing**: 6,431 ADR + 6,431 Non-ADR = 12,862 total  
- **Average sentence length**: 17.2 words  
- **Duplicates removed**: 2,639  
- **Total words**: 462,108  
- **Total characters**: 2.6M+  

Text preprocessing included:
- Lowercasing  
- Punctuation removal  
- Rechecking duplicates  

---
## Evaluation Metrics

- **Accuracy** – Overall classification performance  
- **Precision** – How many predicted ADRs were correct  
- **Recall** – How many true ADRs were identified (most important for safety)  
- **F1-Score** – Harmonic mean of precision and recall
- **Confusion Matrix - Test Set** – Shows counts of correct and incorrect predictions across ADR and non-ADR classes.
- **ROC-AUC - Test Set** – Area under ROC curve (BoW + embedding models only)

---

## Modeling Approaches

### Baseline Models
- **Approach**: Bag-of-Words + Naïve Bayes & GPT-4o-mini Zero-Shot Prompt

### Baseline Results

| Method                          | Accuracy | Precision | Recall | F1-Score |
|---------------------------------|----------|-----------|--------|----------|
| BoW + Naïve Bayes               | 0.76     | 0.73      | 0.81   | 0.77     |
| GPT-4o-mini (Zero-Shot Prompt)  | 0.84     | 0.81      | 0.89   | 0.85     |

---


### 2. Embedding-Based Models
- **Embeddings**: `SBERT`, `BioBERT`, `InstructorXL`
- **Classifier**: Logistic Regression
- **Split**: Stratified 60/20/20 (train/dev/test)
- **Embedding Model Configurations:** max_iter=1000 in Logistic Regression and batch_size=16 during BioBERT embedding

### 3. LLMs (Zero-/Few-shot)
- **Models**: `GPT-4o`, `GPT-4o-mini`, `Phi-4-mini-instruct`, `LLaMA-3.2-3B-Instruct`
- The models were evaluated using only test data 
- **Prompting Strategy**:  
  - Zero-shot: no examples  
  - Few-shot: 4–8 examples 
- **Settings**: `max_tokens=5`, `temperature=0.0-0.1`, `top_p=1.0`
- **Inference Platform**: Azure OpenAI (GPTs, Phi), Hugging Face (LLaMA)
- **Platform**: Google Colab Pro+ with A100 GPU

---

## Results

| Model                    | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| BoW + Naive Bayes        | 0.76     | 0.73      | 0.81   | 0.77     |
| SBERT + LR               | 0.78     | 0.77      | 0.80   | 0.78     |
| BioBERT + LR             | 0.82     | 0.82      | 0.82   | 0.82     |
| InstructorXL + LR        | 0.81     | 0.81      | 0.81   | 0.81     |
| Phi-4-mini Zero-Shot     | 0.77     | 0.73      | 0.87   | 0.79     |
| Phi-4-mini Few-Shot      | 0.73     | 0.75      | 0.69   | 0.72     |
| LLaMA Zero-Shot          | 0.71     | 0.73      | 0.66   | 0.70     |
| LLaMA Few-Shot           | 0.60     | 0.74      | 0.36   | 0.49     |
| GPT-4o-mini Zero-Shot    | 0.84     | 0.81      | 0.89   | 0.85     |
| GPT-4o-mini Few-Shot     | 0.85     | 0.85      | 0.85   | 0.85     |
| GPT-4o Zero-Shot         | 0.81     | 0.75      | 0.95   | 0.84     |
| GPT-4o Few-Shot          | 0.84     | 0.81      | 0.89   | 0.85     |   

---

## Full Pipeline

1. **Data Preparation**  
   Merge ADE and PsyTAR datasets

2. **EDA**  
   Checking class distribution, duplicate removal, sentence and word length analysis, text cleaning, handling class imbalance

3. **Baseline Modeling**  
   Bag-of-Words + Naïve Bayes & GPT-4o-mini Zero-Shot Prompt

4. **Embedding Feature Extraction**  
   Generate dense vectors via InstructorXL, SBERT, BioBERT

5. **Classifier Training**  
   Logistic Regression using sentence embeddings

6. **LLM Prompting & Evaluation**  
   Zero-/few-shot testing on multiple LLMs

7. **Evaluation & Visualization**  
   Confusion matrices, ROC curves for BoW + embedding models, metric comparisons

---

## Insights 

- **Models prioritizing recall** are most suitable for ADR detection to avoid missing critical cases (e.g., GPT-4o Zero-Shot).  
- **ChatGPT-4 models (GPT-4o and GPT-4o-mini)** consistently delivered the strongest performance across both zero-shot and few-shot setups.  
- **Performance varied across models** - some LLMs outperformed embeddings, while others did not.  
- **BoW + Naive Bayes** served as a solid traditional baseline, but were clearly outperformed by modern LLMs and embedding-based models.  
- **Prompting strategy impacted performance** - prompts were adjusted multiple times to improve results; even small wording changes can significantly    affect the precision–recall tradeoff.
- **Tuning the number of few-shot examples per model** was necessary to achieve decent results, reinforcing that more examples don’t always help.  
- **Zero-shot often outperformed few-shot**, showing that giving examples doesn't always lead to better results.  
- **BioBERT + Logistic Regression** had the best evaluation scores among all the embedding models (F1 = 0.82).
- **LLaMA underperformed**, likely due to the lack of fine-tuning or use of enhanced LLaMA variants seen in prior research. It also produced a high      rate of invalid responses (14%), suggesting that the model is sensitive to certain input formats or phrasing.

---

## Graphical Abstract

![ADR Detect - Graphical Abstract](./graphical_abstract.png)

---

## Novelty & Scope

- Integration of **diverse data sources** (expert reports and patient reviews) into a unified dataset
- Comparison between general-purpose LLMs and biomedical embedding models

---

## References

- [Simmering.dev Blog (2025)](https://simmering.dev/blog/modernbert-vs-llm/)
- [ACL Anthology (2025)](https://aclanthology.org/2025.insights-1.11.pdf)
- [SCITEPRESS (2025)](https://www.scitepress.org/Papers/2025/131607/131607.pdf)
- [ADE Corpus V2 on Hugging Face](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2)
- [PsyTAR Dataset Info](https://www.askapatient.com/store/#!/Psytar-Data-Set/p/449080512/category=129206256)
