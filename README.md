# Taxi Fare Estimation

Train taxi fare prediction models, track experiments with MLflow, and test a selected model through FastAPI.

## 1. Set up the project

Clone the repository and enter the project directory:

```bash
git clone https://github.com/usmanhasan123/taxi_fare_estimation.git
cd taxi_fare_estimation
```

Choose an environment:

- **Docker:** run `docker compose up` from the project root. Service settings are defined in the Compose file.
- **Local Python:** install the dependencies:

  ```bash
  pip install -r requirements.txt
  ```

The main workflow uses Jupyter notebooks, but you can use another Python development environment.

## 2. Configure training

Review these files before training:

| File | Purpose |
| --- | --- |
| `data/raw/training.csv` | Raw training and evaluation data. |
| `data/feature_store/feature_store.csv` | List of engineered features used for training. Remove entries to exclude features. |
| `config.env` | Model selection, hyperparameters, cross-validation, feature exclusions, and experiment/prediction settings. |

Supported model choices include Linear Regression and XGBoost. Review features such as `tip_amount` and `tolls_amount`: exclude them if they would be unavailable at prediction time to avoid data leakage.

## 3. Start MLflow

From the project root, start the tracking server in a separate terminal and leave it running:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Open [MLflow](http://localhost:5000/) to compare runs. Each training execution creates a new run within the configured experiment.

## 4. Train and evaluate

Open `notebooks/train_and_test_model.ipynb` and run all cells **except the final cell**. The workflow:

1. Engineers features and creates `data/processed/train_set_feature_engineered_data.csv`.
2. Trains the model selected in `config.env`.
3. Evaluates performance and logs the model, parameters, and metrics to MLflow.

**Restart the Jupyter kernel before rerunning the training notebook for another experiment.**

## 5. Select and export a model

1. Compare the runs in MLflow and choose the model you want to use.
2. Copy its run ID from MLflow, or retrieve it with `mlflow.search_runs()`.
3. Enter the run ID in the final cell of `train_and_test_model.ipynb`.
4. Run that cell to export the selected model to the `app/` directory for inference.

## 6. Test the model with FastAPI

From the project root, run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open [the interactive API docs](http://localhost:8000/docs), enter the required request values, and execute a prediction.

The inference script looks up the relevant data, applies transformations, and uses the exported model to predict the fare.

## Project files

### Development

| File | Purpose |
| --- | --- |
| `requirements.txt` | Development dependencies. |
| `util/predictive_tool.py` | Configuration loading, feature engineering, training, MLflow tracking, and model export. |
| `notebooks/train_and_test_model.ipynb` | Main training, evaluation, prediction, and model-export workflow. |
| `notebooks/adhoc_training.ipynb` | Sandbox for experiments, debugging, and custom analysis. |
| `notebooks/inference_testing.ipynb` | Tests the inference code used at runtime. |

### Application

| File | Purpose |
| --- | --- |
| `app/config_prod.env` | Inference settings, including excluded columns, the target column, and the lookup ID. |
| `app/feature_store.csv` | Features used for prediction. |
| `app/prediction_set.csv` | Lookup data used to retrieve features for prediction. |
| `app/requirements.txt` | Application dependencies. |
| `app/model.pkl` | Exported model used for inference. |
| `app/inference.py` | Request processing, data lookup, transformations, and prediction. |
| `app/main.py` | FastAPI application entry point. |

## Streamlit version

A separate Streamlit application also demonstrates this workflow. It accepts an integer ride ID, retrieves the corresponding ride data, and predicts the fare.
**https://taxifareestimation-sgwcuhjlypdv4n8vdcvzi6.streamlit.app/**
