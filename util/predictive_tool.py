import pandas as pd
from scipy.stats import f_oneway
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv
import os
import pickle
import json
import ast
from pathlib import Path
import mlflow
import mlflow.sklearn

class fare_prediction_class:
    def __init__(self):
        self.root_dir=Path(__file__).resolve().parent.parent
        load_dotenv(os.path.join(self.root_dir, 'config.env'))
        print(self.root_dir)
        # self.root_dir=os.getenv('root_dir')

    def eda(self, col):
        # col=os.getenv('eda_col')
        train_data_path=os.path.join(self.root_dir, 'data/raw/training.csv')
        train_data=pd.read_csv(train_data_path)
        print(train_data[col].describe())
        print(train_data[col].hist())

    def read_and_transform_data(self, data='train_set'):
        if data=='train_set':
            file='training.csv'
        else:
            file='prediction_set.csv'
        train_data_path=os.path.join(self.root_dir, 'data/raw/', file)
        train_data=pd.read_csv(train_data_path)

        train_data['tpep_pickup_datetime']=pd.to_datetime(train_data['tpep_pickup_datetime'])
        train_data['tpep_dropoff_datetime']=pd.to_datetime(train_data['tpep_dropoff_datetime'])
        train_data['PULocationID']=train_data['PULocationID'].astype(str)
        train_data['DOLocationID']=train_data['DOLocationID'].astype(str)
        train_data['passenger_count']=train_data['passenger_count'].astype(int)
        train_data['RatecodeID']=train_data['RatecodeID'].astype(int).astype(str)
        train_data['payment_type']=train_data['payment_type'].astype(int)
        return train_data

    def feature_engineering(self, data='train_set'):
        train_data=self.read_and_transform_data(data)

        train_data['trip_duration_min']=(train_data['tpep_dropoff_datetime']-train_data['tpep_pickup_datetime']).dt.total_seconds()/60
        train_data['pickup_hour']=train_data['tpep_pickup_datetime'].dt.hour
        train_data['dow']=train_data['tpep_pickup_datetime'].dt.dayofweek
        # train_data['month']=train_data['tpep_pickup_datetime'].dt.month
        train_data['is_weekend']=train_data.apply(lambda x: '1' if x.dow>=5 else '0', axis=1)
        train_data['avg_speed_miles_per_min']=train_data['trip_distance']/train_data['trip_duration_min']
        
        train_data['tpep_pickup_datetime']=train_data['tpep_pickup_datetime'].dt.date
        train_data['tpep_dropoff_datetime']=train_data['tpep_dropoff_datetime'].dt.date
        cols_to_exclude=ast.literal_eval(os.getenv('cols_to_exclude'))
        to_keep=[]
        for i in train_data.columns.to_list():
            if i not in cols_to_exclude:
                to_keep.append(i)
        train_data=train_data.loc[:,to_keep]
        if data=='train_set':
            feature_eng_path=os.path.join(self.root_dir,'data/processed/', f'{data}_feature_engineered_data.csv')
            train_data.to_csv(feature_eng_path)
        return train_data

    def corr(self):
        feature_eng_path=os.path.join(self.root_dir,'data/processed/', 'train_set_feature_engineered_data.csv')
        feature_store_path=os.path.join(self.root_dir,'data/feature_store/', 'feature_store.csv')
        train_data=pd.read_csv(feature_eng_path)
        features_df=pd.read_csv(feature_store_path)
        # features=features_df['features'].to_list()
        # train_data=train_data[features]
        num_cols=features_df[features_df['type']=='num']['features'].to_list()
        cat_cols=features_df[features_df['type']=='cat']['features'].to_list()
        print(train_data[num_cols].corr())

        for i in train_data[cat_cols].columns.to_list():
            groups = [
                train_data[train_data[i] == c]["fare_amount"]
                for c in train_data[i].unique()
            ]
            
            f_stat, p_val = f_oneway(*groups)
            
            print(i,':',  p_val)
        
    def train_test_split(self):
        target_col=os.getenv('target_col')
        test_split=float(os.getenv('test_split'))
        feature_eng_path=os.path.join(self.root_dir,'data/processed/', 'train_set_feature_engineered_data.csv')
        feature_store_path=os.path.join(self.root_dir,'data/feature_store/', 'feature_store.csv')
        train_data=pd.read_csv(feature_eng_path)
        features_df=pd.read_csv(feature_store_path)
        features=features_df['features'].to_list()
        train_data=train_data[features]
        
        train_data=train_data.reset_index(drop=True)
        x=train_data.drop(target_col, axis=1)
        y=train_data[target_col]
        y=y.to_frame().reset_index(drop=True)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=test_split,
            random_state=42
        )
        return x_train, x_test, y_train, y_test

    def model_training_without_cv(self, model_pipe, model_algo, x_train, x_test, y_train, y_test, params_used):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"experiment_{timestamp}"): 
            model_pipe.fit(x_train, y_train)
    
            y_pred_train=model_pipe.predict(x_train)
            y_pred_test=model_pipe.predict(x_test)
    
            train=pd.DataFrame()
            train['y_test']=y_train
            train['y_pred']=y_pred_train
            mae_train=mean_absolute_error(y_train, y_pred_train)
            mse_train=mean_squared_error(y_train, y_pred_train)
            rmse_train=root_mean_squared_error(y_train, y_pred_train)
    
            test=pd.DataFrame()
            test['y_test']=y_test
            test['y_pred']=y_pred_test
            mae_test=mean_absolute_error(y_test, y_pred_test)
            mse_test=mean_squared_error(y_test, y_pred_test)
            rmse_test=root_mean_squared_error(y_test, y_pred_test)
    
            metrics={'training_MAE': mae_train, 'training_MSE': mse_train, 'training_RMSE': rmse_train, 'test_MAE': mae_test, 'test_MSE': mse_test, 'test_RMSE': rmse_test}

            mlflow.log_params(params_used)
            mlflow.log_metrics(metrics)
            model_info=mlflow.sklearn.log_model(sk_model=model_pipe, name='model')

            print("Run ID:", mlflow.active_run().info.run_id)
            print("Model URI:", model_info.model_uri)
    
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # experiment_folder = os.path.join(self.root_dir,'model_exp', f"experiment_{timestamp}")
            # os.makedirs(experiment_folder, exist_ok=True)
            # model_path = os.path.join(experiment_folder, "model.pkl")
            # with open (model_path, 'wb') as f:
            #     pickle.dump(model_pipe, f)
    
            # metrics_path = os.path.join(experiment_folder, "metrics.json")
            # with open (metrics_path, 'w') as f:
            #     json.dump(metrics, f, indent=4)

    def model_training_with_cv(self, model_pipe, model_algo, x_train, x_test, y_train, y_test, params_used):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"experiment_{timestamp}"):    
            # if os.getenv('params')!=None:
            #     params=ast.literal_eval(os.getenv('params'))
            # else:
            #     params={}
            folds=int(os.getenv('folds'))
            grid = GridSearchCV(
                estimator=model_pipe,
                param_grid=params_used,
                cv=folds,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
    
            grid.fit(x_train, y_train)
    
            y_pred_train=grid.predict(x_train)
            y_pred_test=grid.predict(x_test)
    
            train=pd.DataFrame()
            train['y_test']=y_train
            train['y_pred']=y_pred_train
            mae_train=mean_absolute_error(y_train, y_pred_train)
            mse_train=mean_squared_error(y_train, y_pred_train)
            rmse_train=root_mean_squared_error(y_train, y_pred_train)
    
            test=pd.DataFrame()
            test['y_test']=y_test
            test['y_pred']=y_pred_test
            mae_test=mean_absolute_error(y_test, y_pred_test)
            mse_test=mean_squared_error(y_test, y_pred_test)
            rmse_test=root_mean_squared_error(y_test, y_pred_test)
    
            metrics={'training_MAE':mae_train, 'training_MSE':mse_train, 'training_RMSE':rmse_train, 'test_MAE':mae_test, 'test_MSE':mse_test, 'test_RMSE':rmse_test}

            best_hyperparameters={'best_hyperparameters': grid.best_params_}

            mlflow.log_params(params_used)
            mlflow.log_params(best_hyperparameters)
            mlflow.log_metrics(metrics)
            model_info=mlflow.sklearn.log_model(sk_model=grid, name='model')

            print("Run ID:", mlflow.active_run().info.run_id)
            print("Model URI:", model_info.model_uri)
            
    
            
            # experiment_folder = os.path.join(self.root_dir,'model_exp', f"experiment_{timestamp}")
            # os.makedirs(experiment_folder, exist_ok=True)
            # model_path = os.path.join(experiment_folder, "model.pkl")
            # with open (model_path, 'wb') as f:
            #     pickle.dump(grid, f)
    
            # metrics_path = os.path.join(experiment_folder, "metrics.json")
            # with open (metrics_path, 'w') as f:
            #     json.dump(metrics, f, indent=4)


    def model_training(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("taxi-fare-prediction")
        model_algo=os.getenv('model')
        
        x_train, x_test, y_train, y_test=self.train_test_split()
        scaler_dict={'standard':StandardScaler(), 'minmax':MinMaxScaler()}
        encoder_dict={'onehot':OneHotEncoder(handle_unknown="ignore"), 'label':LabelEncoder()}

        if model_algo=='xgb':
            params_used={'n_estimators':int(os.getenv('n_estimators')), 'learning_rate': float(os.getenv('learning_rate')), 'reg_lambda':int(os.getenv('reg_lambda')), 'objective':"reg:squarederror"}
        else:
            params_used={}
            
        model_dict={'linear_regression': LinearRegression(), 
                    'xgb': XGBRegressor(
                                        n_estimators=int(os.getenv('n_estimators')),
                                        learning_rate=float(os.getenv('learning_rate')),
                                        reg_lambda=int(os.getenv('reg_lambda')),
                                        objective="reg:squarederror"
                                    )}
        feature_store_path=os.path.join(self.root_dir,'data/feature_store/', 'feature_store.csv')
        features_df=pd.read_csv(feature_store_path)

        num_cols=features_df[features_df['type']=='num']['features'].to_list()
        cat_cols=features_df[features_df['type']=='cat']['features'].to_list()

        if os.getenv('target_col') in num_cols:
            num_cols.remove(os.getenv('target_col'))

        scaler=os.getenv('scaler')
        encoder=os.getenv('encoder')
        # model_algo=os.getenv('model')

        preprocessor = ColumnTransformer([
            ("num", scaler_dict[scaler], num_cols),
            ("cat", encoder_dict[encoder], cat_cols)
        ])
        model = model_dict[model_algo]

        model_pipe=Pipeline([('preprocess', preprocessor),
                            ('model', model)])

        if os.getenv('cross_validation')=='False':
            self.model_training_without_cv(model_pipe, model_algo, x_train, x_test, y_train, y_test, params_used)
        else:
            if model_algo=='xgb':
                params_used_cv=ast.literal_eval(os.getenv('params'))
            else:
                params_used_cv={}
            self.model_training_with_cv(model_pipe, model_algo, x_train, x_test, y_train, y_test, params_used_cv)

    def predict(self, n):
        model=joblib.load(os.path.join(root_dir, 'model_exp/experiment_20260926_214935/model.pkl'))
        pred_data=class_obj.feature_engineering('prediction_set')
        feature_store_path=os.path.join(root_dir,'data/feature_store/', 'feature_store.csv')
        features_df=pd.read_csv(feature_store_path)
        features=features_df['features'].to_list()
        if 'fare_amount' in features:
            features.remove('fare_amount')
        if 'id' not in features:
            features.append('id')
        pred_data=pred_data[features]
        x=pred_data[pred_data['id']==n]
        preds=model.predict(x)
        return preds

    def deploy(self, run_id):
        experiment = mlflow.get_experiment_by_name(
            "taxi-fare-prediction"
        )
        
        logged_models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'"
        )
        
        model_uri = f"models:/{logged_models.model_id[0]}"
        
        model = mlflow.sklearn.load_model(model_uri)

        model_path = os.path.join(self.root_dir,"app", "model.pkl")
        with open (model_path, 'wb') as f:
            pickle.dump(model, f)

    # def predictions(self):
    #     pred_data=self.feature_engineering('prediction_set')
    #     model_exp=os.getenv('model_exp_for_pred')
    #     model_path=os.path.join(self.root_dir, 'model_exp', model_exp, 'model.pkl')
    #     with open(model_path, 'rb') as f:
    #         model=pickle.load(f)

    #     preds=model.predict(pred_data)
    #     pred_data['predictions']=preds
    #     pred_data[['id', 'predictions']].to_csv(os.path.join(self.root_dir,'data/processed/', 'ride_level_predictions.csv'))
        
        
        



        