# Property Value Prediction

## Overview
A machine learning model to predict property values using LightGBM with log-transformation on the target variable. Built for a competitive prediction challenge using French real estate transaction data.

## How It Works
1. Loads training and test data from JSON files
2. Computes target encodings for commune codes and property types
3. Engineers features including area ratios, room counts and price estimates
4. Trains LightGBM on log-transformed target using full dataset
5. Generates predictions and exports to JSON

## Key Features Engineered
- Total area (built + land)
- Area per room ratio
- Land to built ratio
- Commune transaction count
- Combined commune and property type median encoding
- Price per built and land square metre

## Model
| Parameter | Value |
|-----------|-------|
| Algorithm | LightGBM Regressor |
| Objective | MAE (regression_l1) |
| Estimators | 2489 |
| Learning rate | 0.01 |
| Max depth | 8 |

## Results
| Metric | Score |
|--------|-------|
| Baseline MAE | ~80,000+ |
| Reference solution MAE | ~43,553 |
| Submission  MAE | ~44,144 |

## Technologies
- Python
- LightGBM
- Pandas, NumPy
- Scikit-learn
