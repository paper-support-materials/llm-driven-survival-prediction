# Beyond Classical Models: LLM-Driven Survival Analysis for Breast Cancer Prognosis  
Using European Cancer Registry Data

This repository accompanies the paper **“Beyond Classical Models: LLM-Driven Survival Analysis for Breast Cancer Prognosis Using European Cancer Registry Data”** by Sergio Consoli, Dimitris Katsimpokis, Antonello Meloni, Diego Reforgiato Recupero, and Matthijs Sloep.

It provides the code, experiments, and model configurations used to evaluate Large Language Models (LLMs) for survival prediction in oncology, with a focus on breast cancer prognosis. The work investigates whether modern LLM architectures can meaningfully outperform classical statistical and machine learning models, using synthetic data shaped after European cancer registries and validating results on real-world cohorts.

---

## Overview

Survival analysis remains a central task in oncology research, yet most predictive approaches still rely on traditional methods such as Cox Regression or Gradient Boosting. In this project we explore the use of **Generative AI and LLMs** for **time-to-event prediction**, leveraging their capacity to model complex feature interactions and handle censored data.

Our study includes:

- Creation and preprocessing of a **synthetic dataset of 60,000 breast cancer patients**, mirroring characteristics of the Netherlands Cancer Registry.
- Comprehensive **feature engineering**, **imputation**, and **quantile-based normalization**.
- Fine-tuning of diverse LLM families:
  - encoder-only models  
  - decoder-only models  
  - encoder-decoder models  
- Benchmarking against classical baselines:
  - Penalized Cox Regression  
  - Gradient Boosting Survival Trees  
- Validation on a **real-world cohort of 183,304 patients** from Dutch cancer registries.

---

## Key Findings

- Fine-tuned LLMs achieve **superior predictive performance** compared to classical survival models.  
- Models trained on synthetic data **generalize robustly** to real-world patient cohorts.  
- LLM architectures effectively handle **censoring**, **non-linear dependencies**, and **high-dimensional clinical variables**.  
- The workflow offers a **privacy-preserving** path aligned with European initiatives such as the **European Health Data Space**, enabling cross-border research without exposing sensitive health data.

---

## Repository Contents

- **/models**  
  Fine-tuning pipelines for encoder-only, decoder-only, and encoder-decoder LLMs.

- **/classical_baselines**  
  Implementations of Cox Regression and Gradient Boosting.

- **/evaluation**  
  Metrics for survival analysis: MAE, RMSE, Brier score, and concordance-based measures.

---

## Installation

```bash
git clone https://github.com/your-org/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

GPU acceleration (CUDA) is recommended for training LLMs.
