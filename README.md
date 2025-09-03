# Establishing Large Language Model Baselines for Physiological Time Series: Insights from Intrapartum Electronic Fetal Monitoring
Relevant code and analysis for our paper: [URL]

## Overview:
We benchmark GPT-4o and LLaMA-4 on intrapartum cardiotocography tracings, showing that while large language models (LLM) can capture simple time-series patterns and sometimes rival expert consensus, they still lag far behind specialized deep learning models for acidemia prediction.

<img width="854" height="465" alt="Screenshot 2025-09-03 at 12 21 57 PM" src="https://github.com/user-attachments/assets/2a081890-f6a1-448e-8ec4-4433e1e6690a" />

## Repository Contents:

### /LLM-EFM-PROMPT
Prompt template used for zero-shot LLM analysis of raw electronic fetal monitoring (EFM) time-series tracings.

### /LLM_annotation_analysis.py
Includes the manual calculation of EFM time-series metrics (baseline FHR, variability, accelerations, decelerations, UC frequency), and comparative evaluation of model, expert, and ground-truth performance. Also includes computation of regression and classification metrics, including RMSE, MAE, bias, balanced accuracy, F1, and $R^2$.

### /interpretation_only_performance.py
Implementation of the “interpretation-only” configuration. In this setting, LLMs are provided precomputed, structured summaries of CTG metrics instead of raw signals. Models apply FIGO guideline criteria to these summaries to generate floating-point pH estimates. Performance is quantified against ground truth using RMSE, MAE, bias, and $R^2$.

## Data:
- Data is drawn from the CTU-UHB intrapartum cardiotocography database https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/.
- Expert annotations were derived from https://people.ciirc.cvut.cz/~spilkjir/data.html
- Replication requires access to the dataset under its original license.
