from pathlib import Path
import joblib
import os
import pandas as pd
import ast
from dotenv import load_dotenv

root_dir=Path(__file__).resolve().parent
model=joblib.load(os.path.join(root_dir, 'model.pkl'))

load_dotenv(os.path.join(root_dir, 'config_prod.env'))

def read_and_transform_data():
    pred_data_path=os.path.join(root_dir, 'prediction_set.csv')
    pred_data=pd.read_csv(pred_data_path)

    pred_data['tpep_pickup_datetime']=pd.to_datetime(pred_data['tpep_pickup_datetime'])
    pred_data['tpep_dropoff_datetime']=pd.to_datetime(pred_data['tpep_dropoff_datetime'])
    pred_data['PULocationID']=pred_data['PULocationID'].astype(str)
    pred_data['DOLocationID']=pred_data['DOLocationID'].astype(str)
    pred_data['passenger_count']=pred_data['passenger_count'].astype(int)
    pred_data['RatecodeID']=pred_data['RatecodeID'].astype(int).astype(str)
    pred_data['payment_type']=pred_data['payment_type'].astype(int)
    return pred_data

def feature_engineering():
    pred_data=read_and_transform_data()

    pred_data['trip_duration_min']=(pred_data['tpep_dropoff_datetime']-pred_data['tpep_pickup_datetime']).dt.total_seconds()/60
    pred_data['pickup_hour']=pred_data['tpep_pickup_datetime'].dt.hour
    pred_data['dow']=pred_data['tpep_pickup_datetime'].dt.dayofweek
    # train_data['month']=train_data['tpep_pickup_datetime'].dt.month
    pred_data['is_weekend']=pred_data.apply(lambda x: '1' if x.dow>=5 else '0', axis=1)
    pred_data['avg_speed_miles_per_min']=pred_data['trip_distance']/pred_data['trip_duration_min']
    
    pred_data['tpep_pickup_datetime']=pred_data['tpep_pickup_datetime'].dt.date
    pred_data['tpep_dropoff_datetime']=pred_data['tpep_dropoff_datetime'].dt.date
    cols_to_exclude=ast.literal_eval(os.getenv('cols_to_exclude')) # add in app directory for prod
    to_keep=[]
    for i in pred_data.columns.to_list():
        if i not in cols_to_exclude:
            to_keep.append(i)
    pred_data=pred_data.loc[:,to_keep]
    return pred_data

def predict(n):
    pred_data=feature_engineering()
    feature_store_path=os.path.join(root_dir, 'feature_store.csv')
    features_df=pd.read_csv(feature_store_path)
    features=features_df['features'].to_list()
    if os.getenv('target_col') in features:
        features.remove(os.getenv('target_col'))
    if os.getenv('lookup_col') not in features:
        features.append(os.getenv('lookup_col'))
    pred_data=pred_data[features]
    x=pred_data[pred_data['id']==n]
    preds=model.predict(x)
    return preds
    # return float(preds[0])

