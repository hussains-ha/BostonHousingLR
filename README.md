# BostonHousingLR
This notebook presents a linear regression analysis using the Boston Housing dataset to predict median home 
values based on socioeconomic and geographic features. Built using concepts and customized abstractions 
from the Dive into Deep Learning framework, it focuses on fundamental machine learning techniques like minibatch stochastic 
gradient descent and gradient-based optimization.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Key Features & Analysis](#key-features--analysis)
- [Model Implementation](#model-implementation)
- [Installation & Requirements](#installation--requirements)

## Project Overview

The goal of this project is to apply a single-layer linear regression model to a classic benchmark dataset. 
By keeping the architecture simple, the notebook highlights the core mechanics of:
- Data Loading: Efficiently handling feature matrices and target vectors.
- Training Loops: Implementing customized versions of standard deep learning abstractions.
- Optimization: Using gradient descent to minimize loss.

## Dataset Description
The Boston Housing dataset contains 13 features describing various aspects of Boston neighborhoods. 
The target variable is MEDV (Median value of owner-occupied homes in $1000s).

| Feature | Description                                                           |
|---------|-----------------------------------------------------------------------|
| CRIM    | Per capita crime rate by town                                         |
| ZN      | Proportion of residential land zoned for lots > 25,000 sq.ft.         |
| INDUS   | Proportion of non-retail business acres per town                      |
| CHAS    | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) |
| NOX     | Nitric oxides concentration (parts per 10 million)                    |
| RM      | Average number of rooms per dwelling                                  |
| AGE     | Proportion of owner-occupied units built prior to 1940                |
| DIS     | Weighted distances to five Boston employment centers                  |
| RAD     | Index of accessibility to radial highways                             |
| TAX     | Full-value property-tax rate per $10,000                              |
| PTRATIO | Pupil-teacher ratio by town                                           |
| B       | 1000(Bk − 0.63)^2 where Bk is proportion of blacks by town            |
| LSTAT   | % lower status of the population                                      |
| MEDV    | Median value of owner-occupied homes in thousands of USD (target)     |


## Key Features & Analysis

The notebook includes Exploratory Data Analysis (EDA) to visualize relationships within the data:
- Distribution: Housing prices are right-skewed, indicating high-value outliers.
- Correlations: * RM vs MEDV: A strong positive linear relationship exists, confirming that more rooms 
generally lead to higher home values. 
  - LSTAT vs MEDV: A negative relationship is observed, where higher percentages of lower-status populations correlate with lower median home values.


## Model Implementation

The model is built using a single linear layer. The notebook tracks the training progress by monitoring the loss across 
epochs and evaluates the final model performance.

### Insights from Learned Weights:
- Features like RM (Rooms) and LSTAT (Status) dominate price predictions due to their high weight magnitudes.
- Features with near-zero weights (e.g., AGE) indicate little influence on the target, aligning with trends observed in the scatterplots.


## Installation & Requirements

To run this notebook, you will need the following libraries:
- `pandas`
- `matplotlib`
- `os`
- `torch`

### Project Structure:
- model.ipynb: The main analysis and training pipeline.
- data/housing.csv: The raw dataset.
- abstractions/torch.py: Source for customized training and data loading abstractions.
Note: This project builds on concepts from Zhang et al., "Dive into Deep Learning" (d2l.ai).
