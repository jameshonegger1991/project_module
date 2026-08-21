# Beyond Accuracy: Evaluating the stability of SHAP explanations Across Machine Learning Models for PISA 2022 Mathematics in Switzerland

PROJECT MODULE / CSM500-2026-APR 

Lucas Pannatier

MSc Computer Science  
University of London

## Overview

This repository contains the source code for a Master's research project investigating the consistency and stability of feature importance across machine learning models for predicting mathematics performance among Swiss students.

Using PISA 2022 data, the study compares multiple regression and classification models. However, model explanations and the associated SHAP-based analyses (inter-model agreement and intra-model stability) are developed specifically for the selected task, which is regression.

## Project Structure

PROJECT CODE 

    /src
        config.py
        dataset_building.py
        dataset_cleaning_and_preprocessing.py
        feature_selection.py
        model_training.py
        SHAP_analysis.py
        tables.py
        utils.py
        visualisations.py
        main.py

    /tests
        test_dataset_building.py
        test_feature_selection.py
        test_model_training.py
        test_preprocessing.py
        test_SHAP_analysis.py

    /dataset
        swiss_reduced_dataset.csv
    
    /notebook
        project.ipynb
    
    README.md
    requirements.txt
    


## Requirements

The project was developed using Python 3.12.

Install the required dependencies with:

    pip install -r requirements.txt

The requirements file also includes pytest for running the test suite.

## Dataset

This project uses data from the OECD Programme for International Student Assessment (PISA). Due to their size, the raw data files are not included in this repository.

To recreate the working dataset (`swiss_reduced_dataset.csv`) from scratch, run the `reduced_swiss_dataset()` function. The required `data/` directory is created automatically if it does not already exist.

The following PISA 2022 student and school questionnaire datasets must first be downloaded manually and placed in the `data/` directory:

```text

CY08MSP_STU_QQQ.sav

CY08MSP_SCH_QQQ.sav

```

The source datasets are available from Zenodo or from the official OECD website:

https://zenodo.org/records/13382904

https://www.oecd.org/en/data/datasets/pisa-2022-database.html#data 

If the required source files are not found, the dataset construction process is stopped and the missing files are reported to the user.

## Running the Project

From the root directory of the project, run:

    python -m src.main

All generated tables and figures will be saved in the `outputs/` directory, which the script will create automatically when launched.

## Running the Tests

From the root directory, run:

    pytest

## Methodology

The main experimental pipeline consists of:

1. Dataset construction and preprocessing
2. Exploratory data analysis
3. Feature selection
4. Regression and classification model training and evaluation
5. Construction of a Rashomon set of similarly performing models
6. SHAP-based feature importance analysis
7. Inter-model feature ranking agreement analysis
8. Bootstrap-based intra-model stability analysis
9. Identification of robust features

Steps 6 to 9 are conducted for the regression task only, which was retained for the final interpretability and stability analysis.

The detailed methodology, experimental design, and justification of methodological choices are provided in the accompanying Master's project report.

## Reproducibility

Random seeds are used throughout the experimental pipeline where applicable to improve reproducibility. However, this introduces a trade-off between exact reproducibility and the assessment of variability across repeated random sampling procedures, particularly in the bootstrap analysis (see `intra_model_stability_assessment()`). This may affect the final set of features retained by `assess_features_robustness()`.

Package versions required to reproduce the computational environment are specified in `requirements.txt`.


