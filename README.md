# Taxi Fare Estimation

This repository provides a complete workflow for training taxi fare prediction models, tracking experiments, and generating fare predictions on new data.

## Getting Started

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/usmanhasan123/taxi_fare_estimation.git
```

This project was primarily developed using Jupyter Notebooks, but you may use any Python development environment of your choice.

---

## Repository Structure

After cloning the repository, you will find the following files and folders:

### Data Files

#### `training.csv`

Dataset used for model training and evaluation.

#### `prediction_set.csv`

Dataset on which trained models generate fare predictions.

#### `feature_store.csv`

Contains all engineered features used during model training.

Users can remove features from this file if they do not want them to be included in the training process.

---

### Configuration Files

#### `requirements.txt`

Lists all Python packages required to run the project.

#### `config.env`

Contains user-defined configurations and model parameters, including:

* Model selection
* Hyperparameter settings
* Cross-validation settings
* Feature exclusion settings
* Experiment selection for prediction generation

---

### Code Files

#### `predictive_tool.py`

Core utility module that:

* Reads configurations from `config.env`
* Performs feature engineering
* Trains machine learning models
* Generates predictions
* Tracks experiment results

This module is designed to make the workflow reusable, scalable, and easier to maintain.

#### `train_and_test_model.ipynb`

Main notebook used to:

* Train models
* Evaluate performance
* Track experiments
* Generate predictions

#### `adhoc_training.ipynb`

A sandbox notebook for experimentation, debugging, and custom analyses.

Users can freely modify this notebook without affecting the main training workflow.

---

### Experiment Tracking

#### `model_exp/`

Stores all experiment runs.

For each experiment, the repository saves:

* Trained model artifacts
* Evaluation metrics
* Experiment configurations

Each training run automatically creates a new experiment folder inside `model_exp`.

The repository may already contain experiment folders from previous runs for reference.

---

## Installation

Open `train_and_test_model.ipynb`.

Run the cell containing:

```bash
pip install -r requirements.txt
```

This step only needs to be performed once.

---

## Training a Model

1. Open `train_and_test_model.ipynb`.
2. Review and modify configurations in `config.env` as needed.
3. Run all notebook cells **except the final cell**.

During training, the workflow will:

* Perform feature engineering
* Create `train_set_feature_engineered_data.csv`
* Train the selected model
* Evaluate model performance
* Save evaluation metrics (MAE, MSE, and RMSE)
* Store trained models and experiment artifacts in `model_exp`

Each training run generates a new experiment folder inside `model_exp`.

### Configurable Options

The `config.env` file allows you to customize:

* Model type (e.g., Linear Regression, XGBoost)
* Hyperparameters
* Cross-validation settings
* Features to exclude from training
* Experiment selection for prediction generation

You may also exclude columns such as `tip_amount` and `tolls_amount` since these may introduce data leakage because they are often unavailable when predicting fares before a trip is completed.

---

## Generating Predictions

After reviewing experiment results and selecting your preferred model:

1. Identify the corresponding experiment folder inside `model_exp`.
2. Copy the experiment folder name.
3. Update the `model_exp_for_pred` parameter in `config.env` with that experiment name.

Example:

```text
model_exp_for_pred=experiment_20260115_103000
```

4. Return to `train_and_test_model.ipynb`.
5. Run the final notebook cell.

The prediction workflow will:

* Generate `prediction_set_feature_engineered_data.csv`
* Load the selected trained model
* Generate fare predictions for each trip
* Create `ride_level_predictions.csv`

---

## Output Files

### `train_set_feature_engineered_data.csv`

Feature-engineered version of the training dataset.

### `prediction_set_feature_engineered_data.csv`

Feature-engineered version of the prediction dataset.

### `ride_level_predictions.csv`

Contains predicted fare amounts for every ride in `prediction_set.csv`.

### `model_exp/`

Contains all experiment artifacts, including:

* Trained models
* Metrics (MAE, MSE, RMSE)
* Experiment configurations

---

## Typical Workflow

1. Configure parameters in `config.env`.
2. Run training cells in `train_and_test_model.ipynb`.
3. Review experiment metrics in `model_exp`.
4. Select the best-performing experiment.
5. Update `model_exp_for_pred` in `config.env`.
6. Run the prediction cell.
7. Review results in `ride_level_predictions.csv`.
