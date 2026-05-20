# Hybrid-MoE-vMF-HMM
This is the pytorch implementation of paper "A Hybrid Mixture-of-Experts von Mises–Fisher Hidden Markov Model for Interpretable Next-POI Prediction". This Paper has been subimitted to ACM SIGSPATIAL2026
![model-structure](Figures/vMF-HMM_framework-v5.png)

## Installation
```
pip install -r requirements.txt
```
## Requirements

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
torch>=2.0
tqdm>=4.65
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
contextily>=1.3
pyproj>=3.5
```

## Training

First, run `multimodal_line_embedding.py` to learn POI, user, category, and time embeddings. Then, run `train.py` to train the MoE vMF-HMM model.All hyperparameters are assigned default values and can be adjusted through command-line arguments.

## Dataset

To run the embedding module and the hybrid HMM model, please first download `dataset_TSMC2014_NYC.txt` from the [Foursquare Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) page.

The `dataset/` folder contains the following files:

- `NYC_data.pkl`: cleaned and preprocessed data used by the embedding module.
- `NYC_getnext_ready.pkl`: processed trajectory data used by the MoE vMF-HMM model.
- `category_cluster_map.csv`: macro-category mapping used for constructing the category-macro graph in graph-based embedding.
- `data_prepare.py`: code for data cleaning and preprocessing.
- Other files in this folder are generated outputs from the embedding module.

Due to data licensing and file size limitations, the original raw dataset is not included in this repository.

## Figures

This folder contains the visualization results generated from the proposed model. These figures follow the same interpretation rules described in the paper.

## Results

This folder contains the experimental results of the proposed model with 4 experts, 4 hidden states per expert, and an angular constraint of 0.05 radians.
