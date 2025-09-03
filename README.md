# llm-efm-baselines
**Establishing Large Language Model Baselines for Physiological Time Series: Insights from Intrapartum Electronic Fetal Monitoring**
Relevant code and analysis for [URL]

## Repository Contents:
### /LLM-EFM-PROMPT
Prompt template used for zero-shot large language model (LLM) analysis of raw electronic fetal monitoring (EFM) time-series tracings.

### /LLM_annotation_analysis.py
Includes the manual calculation of EFM time-series metrics (baseline FHR, variability, accelerations, decelerations, UC frequency), and comparative evaluation of model, expert, and ground-truth performance. Also includes computation of regression and classification metrics, including RMSE, MAE, bias, balanced accuracy, F1, and $R^2$.

### /interpretation_only_performance.py
Implementation of the “interpretation-only” configuration. In this setting, LLMs are provided precomputed, structured summaries of CTG metrics instead of raw signals. Models apply FIGO guideline criteria to these summaries to generate floating-point pH estimates. Performance is quantified against ground truth using RMSE, MAE, bias, and $R^2$.

## Data
- Data is drawn from the CTU-UHB intrapartum cardiotocography database (PhysioNet).
- Replication requires access to the dataset under its original license.
