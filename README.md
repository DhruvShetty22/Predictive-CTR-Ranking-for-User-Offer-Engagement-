# Decision Science Track - The American Express Campus Challenge 2025

This project implements a machine learning pipeline to predict customer responses to offers using hybrid embeddings, event aggregates, and feature engineering. The model leverages LightGBM for classification and applies fallback strategies for missing embedding scenarios.

## Project Structure
```
.
├── Dataset/
│   ├── train_data.parquet
│   ├── test_data.parquet
├── TransEncoding/
│   ├── customer_id_to_embedding.pkl
│   ├── customer_fallbacks.pkl
├── OfferEncoding/
│   ├── offer_id_to_embedding.pkl
│   ├── offer_cluster_centers.pkl
│   ├── global_mean_offer_vector.npy
├── EventsEncoding/
│   ├── event_pair_agg.parquet
│   ├── event_customer_agg.parquet
│   ├── event_offer_agg.parquet
├── Code/
│   ├── train_model.py
│   ├── predict_and_save.py
│   ├── feature_engineering.py
│   ├── lightgbm_model.pkl
├── output/
│   ├── test_preds.npy
│   ├── test_ids.npy
│   ├── submission.csv
├── README.md
```
## Features
* Customer Embeddings: Uses precomputed embeddings with fallback to nearest cluster or global mean.

* Offer Embeddings: Same strategy as customer embeddings.

* Event Aggregates: Uses pre-aggregated metrics from customer-offer pairs, with multi-level fallback.

* Base Features: 366 raw features (f1–f366), numeric + encoded categorical.

* Model: Trained using LightGBMClassifier with early stopping and AUC evaluation.

## Setup Instructions 
### Install dependencies
```
pip install numpy pandas scikit-learn lightgbm joblib tqdm
```
### Directory layout
* Dataset/train_data.parquet

* Dataset/test_data.parquet

* TransEncoding/customer_id_to_embedding.pkl

* TransEncoding/customer_fallbacks.pkl

* OfferEncoding/offer_id_to_embedding.pkl

* OfferEncoding/offer_cluster_centers.pkl

* OfferEncoding/global_mean_offer_vector.npy

* EventsEncoding/event_pair_agg.parquet

* EventsEncoding/event_customer_agg.parquet

* EventsEncoding/event_offer_agg.parquet

## Running the Pipeline
### Train the Model
```
from train_model import train_model
model = train_model()
```
* Builds features from training data.

* Trains a LightGBM model with 80/20 split.

* Outputs validation AUC, log loss.

* Saves model and predictions (lightgbm_model.pkl, val_preds.npy).
### Predict on Test Set
```
from predict_and_save import predict_and_save
sub_df = predict_and_save()
```
* Builds features from test data.

* Loads the trained model.

* Saves test predictions and submission CSV to output directory.
