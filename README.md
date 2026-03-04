# AI-for-Health
 
### SRIP – IIT Gandhinagar (AI for Health)  
Instructor: Prof. Nipun Batra  

## Overview

This project was completed as part of the **selection task for the Summer Research Internship Program (SRIP) at IIT Gandhinagar**, under the supervision of **Prof. Nipun Batra**.

The objective of this task is to analyze physiological sleep signals and detect **breathing irregularities such as apnea and hypopnea** using signal processing and machine learning techniques.

The dataset contains **overnight sleep recordings (~8 hours) from 5 participants**. The signals are processed, visualized, and used to train a neural network model to detect abnormal breathing patterns during sleep.

---

## Dataset Description

Each participant folder contains the following physiological signals:

| Signal | Sampling Rate | Description |
|------|------|------|
| Nasal Airflow | 32 Hz | Measures airflow through the nose |
| Thoracic Movement | 32 Hz | Chest movement during breathing |
| SpO₂ (Oxygen Saturation) | 4 Hz | Blood oxygen saturation |

Additional files include:

- **Event file** – annotated breathing irregularities (e.g., apnea, hypopnea)
- **Sleep profile file** – sleep stage information

---

## Project Tasks

The project consists of three main components:

1. Data Visualization  
2. Signal Preprocessing and Dataset Creation  
3. Model Training and Evaluation  

---

# 1. Data Visualization

The script **vis.py** generates visualizations of sleep signals for a participant.

### Tasks Performed

- Load nasal airflow, thoracic movement, and SpO₂ signals  
- Align signals using timestamps  
- Plot the full **8-hour sleep recording**  
- Overlay annotated breathing events  
- Save the visualization as a **PDF file**

