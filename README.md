# AI-Driven Sovereign Digital Twin (AI-SDT)

This repository contains the baseline experimental code for the paper:
"AI-Driven Sovereign Digital Twin: Self-Evolving Cyber Defense for Industrial IoT and SDN Environments".

## Overview
The code provided here reproduces the baseline experiments discussed in Section 5 of the paper. It evaluates the performance of traditional single-layer Intrusion Detection Systems (IDS) using Random Forest (RF) and XGBoost on two real-world datasets:
1. SDN Dataset: ITU-ML5G-PS-006
2. IIoT Dataset: X-IIoTID

Additionally, it reproduces the Semantic Poisoning experiment (Section 5.2) and generates the corresponding figures (Figures 4 and 5) used in the manuscript.

## Datasets
Due to size constraints, the datasets are not included in this repository. Please download them from their original sources:
* SDN Dataset (ITU-ML5G-PS-006): Available at https://zenodo.org/records/15303565
* IIoT Dataset (X-IIoTID): Available at https://www.kaggle.com/datasets/munaalhawawreh/xiiotid-iiot-intrusion-dataset

After downloading, place the CSV files in the data/ directory:
* data/test_data.csv (for SDN)
* data/X-IIoTID dataset.csv (for IIoT)

## Requirements
Ensure you have Python 3.8+ installed. You can install the required libraries using:

pip install -r requirements.txt

requirements.txt contents:
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0

## How to Run the Experiments

1. SDN Baseline
To train RF and XGBoost on the SDN dataset and generate Figure 4 (SDN parts):
python scripts/01_sdn_baseline.py

2. IIoT Baseline
To train RF and XGBoost on the IIoT dataset and generate Figure 4 (IIoT parts):
python scripts/02_iiot_baseline.py

3. Semantic Poisoning Experiment
To run the label-flipping poisoning experiment and generate Figure 5:
python scripts/03_poisoning_experiment.py

## Results Summary
The baseline results from the paper are summarized below:

| Layer | Model | Accuracy (%) | F1-Score | Train (s) | Infer (ms/sample) |
|-------|-------|:---:|:---:|:---:|:---:|
| SDN | Random Forest | 99.79 | 0.9979 | 31.8 | 0.0086 |
| SDN | XGBoost | 99.87 | 0.9987 | 21.4 | 0.0216 |
| IIoT | Random Forest | 87.80 | 0.8851 | 263.8 | 0.0628 |
| IIoT | XGBoost | 66.20 | 0.7358 | 335.9 | 0.1099 |

## Citation
If you use this code in your research, please cite our paper:

@article{aisdt2026,
  title={AI-Driven Sovereign Digital Twin: Self-Evolving Cyber Defense for Industrial IoT and SDN Environments},
  author={Your Name(s)},
  journal={Journal/Conference Name},
  year={2026}
}

## License
This project is licensed under the MIT License.