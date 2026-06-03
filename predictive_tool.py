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

class fare_prediction_class:
    def __init__(self, path='./config.env'):
        load_dotenv(path)
        self.root_dir=os.getenv('root_dir')

    def eda(self, col):
        # col=os.getenv('eda_col')
        train_data_path=os.path.join(self.root_dir, 'training.csv')
        train_data=pd.read_csv(train_data_path)
        print(train_data[col].describe())
        print(train_data[col].hist())

    def read_and_transform_data(self, data='train_set'):
        if data=='train_set':
            file='training.csv'
        else:
            file='prediction_set.csv'
        train_data_path=os.path.join(self.root_dir, file)
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
        feature_eng_path=os.path.join(self.root_dir, f'{data}_feature_engineered_data.csv')
        train_data.to_csv(feature_eng_path)
        return train_data

    def corr(self):
        feature_eng_path=os.path.join(self.root_dir, 'train_set_feature_engineered_data.csv')
        feature_store_path=os.path.join(self.root_dir, 'feature_store.csv')
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
        feature_eng_path=os.path.join(self.root_dir, 'train_set_feature_engineered_data.csv')
        feature_store_path=os.path.join(self.root_dir, 'feature_store.csv')
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

    def model_training_without_cv(self, model_pipe, model_algo, x_train, x_test, y_train, y_test):
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

        metrics={'model': model_algo, 'training_metrics': {'MAE': mae_train, 'MSE': mse_train, 'RMSE': rmse_train}, 'test_metrics': {'MAE': mae_test, 'MSE': mse_test, 'RMSE': rmse_test}}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = os.path.join(self.root_dir,'model_exp', f"experiment_{timestamp}")
        os.makedirs(experiment_folder, exist_ok=True)
        model_path = os.path.join(experiment_folder, "model.pkl")
        with open (model_path, 'wb') as f:
            pickle.dump(model_pipe, f)

        metrics_path = os.path.join(experiment_folder, "metrics.json")
        with open (metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    def model_training_with_cv(self, model_pipe, model_algo, x_train, x_test, y_train, y_test):
        if os.getenv('params')!=None:
            params=ast.literal_eval(os.getenv('params'))
        else:
            params={}
        folds=int(os.getenv('folds'))
        grid = GridSearchCV(
            estimator=model_pipe,
            param_grid=params,
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

        metrics={'cross_validation': True, 'model': model_algo, 'best_hyperparameters': grid.best_params_, 'training_metrics': {'MAE': mae_train, 'MSE': mse_train, 'RMSE': rmse_train}, 'test_metrics': {'MAE': mae_test, 'MSE': mse_test, 'RMSE': rmse_test}}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = os.path.join(self.root_dir,'model_exp', f"experiment_{timestamp}")
        os.makedirs(experiment_folder, exist_ok=True)
        model_path = os.path.join(experiment_folder, "model.pkl")
        with open (model_path, 'wb') as f:
            pickle.dump(model_pipe, f)

        metrics_path = os.path.join(experiment_folder, "metrics.json")
        with open (metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)


    def model_training(self):
        x_train, x_test, y_train, y_test=self.train_test_split()
        scaler_dict={'standard':StandardScaler(), 'minmax':MinMaxScaler()}
        encoder_dict={'onehot':OneHotEncoder(handle_unknown="ignore"), 'label':LabelEncoder()}
        model_dict={'linear_regression': LinearRegression(), 
                    'xgb': XGBRegressor(
                                        n_estimators=int(os.getenv('n_estimators')),
                                        learning_rate=float(os.getenv('learning_rate')),
                                        reg_lambda=int(os.getenv('reg_lambda')),
                                        objective="reg:squarederror"
                                    )}
        feature_store_path=os.path.join(self.root_dir, 'feature_store.csv')
        features_df=pd.read_csv(feature_store_path)

        num_cols=features_df[features_df['type']=='num']['features'].to_list()
        cat_cols=features_df[features_df['type']=='cat']['features'].to_list()

        if os.getenv('target_col') in num_cols:
            num_cols.remove(os.getenv('target_col'))

        scaler=os.getenv('scaler')
        encoder=os.getenv('encoder')
        model_algo=os.getenv('model')

        preprocessor = ColumnTransformer([
            ("num", scaler_dict[scaler], num_cols),
            ("cat", encoder_dict[encoder], cat_cols)
        ])
        model = model_dict[model_algo]

        model_pipe=Pipeline([('preprocess', preprocessor),
                            ('model', model)])

        if os.getenv('cross_validation')=='False':
            self.model_training_without_cv(model_pipe, model_algo, x_train, x_test, y_train, y_test)
        else:
            self.model_training_with_cv(model_pipe, model_algo, x_train, x_test, y_train, y_test)

    def predictions(self):
        pred_data=self.feature_engineering('prediction_set')
        model_exp=os.getenv('model_exp_for_pred')
        model_path=os.path.join(self.root_dir, 'model_exp', model_exp, 'model.pkl')
        with open(model_path, 'rb') as f:
            model=pickle.load(f)

        preds=model.predict(pred_data)
        pred_data['predictions']=preds
        pred_data.to_csv(os.path.join(self.root_dir, 'ride_level_predictions.csv'))
        
        
        



        