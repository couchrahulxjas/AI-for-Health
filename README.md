# AI-for-Health

## Overview

This project detects **breathing irregularities during sleep**. It was completed as a selection task for the **Summer Research Internship Program (SRIP) at IIT Gandhinagar**.

The pipeline processes overnight sleep recordings (~8 hours) from 5 participants, applies signal processing techniques, and trains a 1D Convolutional Neural Network to automatically classify breathing patterns as normal or abnormal.

---

## Dataset Description

The dataset comprises overnight sleep recordings from **5 participants**, each containing the following physiological signals:

| Signal | Sampling Rate | Description |
|--------|--------------|-------------|
| **Nasal Airflow** | 32 Hz | Airflow measurement through the nasal passages |
| **Thoracic Movement** | 32 Hz | Chest wall movement during respiratory cycles |
| **SpO₂ (Oxygen Saturation)** | 4 Hz | Peripheral blood oxygen saturation levels |

### Additional Data Files

- **Event Annotations** – Labeled breathing irregularities (apnea, hypopnea, etc.)
- **Sleep Stage Information** – REM/NREM classifications for context

---

## Project Pipeline

The project is organized into three main components:

```
Raw Signals → Preprocessing & Feature Engineering → Model Training → Evaluation
```

### 1. Data Visualization (vis.py)

Generates comprehensive visualizations of sleep signals to understand signal characteristics and identify patterns.

**Key Functions:**
- Load and synchronize nasal airflow, thoracic movement, and SpO₂ signals
- Align signals using timestamps to ensure temporal consistency
- Plot the complete 8-hour sleep recording
- Overlay annotated breathing events for visual correlation
- Export visualizations as **high-quality PDF files**

**Output:** PDF files containing multi-signal time-series plots with event annotations

---

### 2. Signal Preprocessing & Dataset Creation (create_dataset.py)

Transforms raw physiological signals into a machine-learning-ready dataset through signal processing and windowing.

#### **Step 1: Bandpass Filtering**

Normal human breathing occurs at **10–24 breaths per minute**, corresponding to a frequency range of **0.17–0.4 Hz**. A **Butterworth bandpass filter** is applied to:
- Extract the breathing frequency band
- Attenuate low-frequency drift and high-frequency noise
- Preserve signal integrity for downstream analysis

#### **Step 2: Windowing**

Signals are segmented into overlapping windows to create a structured dataset:

| Parameter | Value |
|-----------|-------|
| Window Length | 30 seconds |
| Overlap | 50% |
| Sampling Rate | 32 Hz |
| Window Samples | 960 samples |

**Rationale:** 30-second windows capture multiple breathing cycles while remaining short enough for localized pattern detection.

#### **Step 3: Window Labeling**

Each window is automatically labeled using event annotations:

- **Event-Window Overlap ≥ 50%** → Assign the event label (*Hypopnea*, *Obstructive Apnea*, etc.)
- **Event-Window Overlap < 50%** → Label as **Normal**

This threshold-based approach ensures reliable ground truth labels for supervised learning.

**Output:** Structured dataset with windows and corresponding labels, ready for model training

---

### 3. Model Training & Evaluation (train.py)

Trains a 1D Convolutional Neural Network to classify breathing patterns and evaluates performance using cross-validation.

#### **Model Architecture**

A lightweight **1D CNN** designed for temporal signal classification:

```
Input: (960, 2)  →  Conv1D Layers  →  Pooling  →  Dense Layers  →  Classification Output
```

**Input Specification:**
- **Temporal Length:** 960 samples (30 seconds at 32 Hz)
- **Channels:** 2 (Nasal Airflow + Thoracic Movement)
- **Output Classes:** Binary or multi-class (Normal vs. Abnormal; or granular breathing event types)

#### **Evaluation Strategy: Leave-One-Participant-Out (LOPO) Cross-Validation**

To ensure the model generalizes to unseen individuals:

1. **Iteration 1:** Train on Participants {2,3,4,5} → Test on Participant 1
2. **Iteration 2:** Train on Participants {1,3,4,5} → Test on Participant 2
3. **Iteration 3:** Train on Participants {1,2,4,5} → Test on Participant 3
4. **Iteration 4:** Train on Participants {1,2,3,5} → Test on Participant 4
5. **Iteration 5:** Train on Participants {1,2,3,4} → Test on Participant 5

This approach is crucial for sleep apnea detection, as it validates **inter-subject generalization** — a critical requirement for clinical deployment.

#### **Performance Metrics**

Model performance is assessed using standard classification metrics:

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall classification correctness |
| **Precision** | False positive rate (clinical relevance) |
| **Recall** | False negative rate (detection sensitivity) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Detailed error analysis across classes |

**Clinical Interpretation:**
- High **Recall** is critical to minimize missed apnea events
- High **Precision** reduces unnecessary interventions
- **ROC-AUC** can assess the model's discriminative ability

---

## Installation & Usage

### Prerequisites

```bash
python 3.8+
numpy
scipy
scikit-learn
tensorflow / pytorch
matplotlib
pandas
```


## File Structure

```
AI-for-Health/
├── vis.py                          # Signal visualization script
├── create_dataset.py               # Data preprocessing and windowing
├── train.py                        # Model training and evaluation
├── raw_data/                       # Input physiological signals
│   ├── participant_1/
│   ├── participant_2/
│   └── ...
├── processed_data/                 # Windowed dataset (output)
├── visualizations/                 # PDF plots (output)
├── models/                         # Trained models
└── requirements.txt                # Python dependencies
```

---

## Disclaimer

This project utilized **ChatGPT as the primary sole AI assistance tool** for the following reasons:

- **Signal Processing**: Learning and implementing Butterworth bandpass filters using NumPy and SciPy—concepts I was not previously very familiar with.

- **Validation Methodology**: ChatGPT provided insights into Leave-One-Participant-Out (LOPO) cross-validation and why it is essential for validating generalization in clinical applications.


---
