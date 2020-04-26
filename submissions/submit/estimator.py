
import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import math
from scipy.stats import mode
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# MERGE DATA

def _merge_external_data(X):
    __file__ = os.path.join('submissions', 'submit', 'estimator.py')
    filepath = os.path.join(os.path.dirname(__file__), 
                            'external_data.csv')
    
    
    X["DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    X_weather = pd.read_csv(filepath, parse_dates=["Date"], index_col=0)
    
    X_weather["Precipitationmm"] = X_weather["Precipitationmm"].replace('T',None)
    X_weather["Precipitationmm"] = X_weather["Precipitationmm"].astype(float)
    X_weather["Events"].fillna(mode(X_weather["Events"]).mode[0], inplace=True)
    X_weather.drop(columns=["Max Gust SpeedKm/h","Events"], inplace=True)
    
    X_weather = X_weather.rename(columns={'Date': 'DateOfDeparture', 
                                          'AirPort': 'Arrival'})
    X_merged = pd.merge(X, X_weather, 
                        how='left', 
                        on=['DateOfDeparture', 'Arrival'], 
                        sort=False)
    
    return X_merged



# DATE

def parse_date(x):
    date = pd.to_datetime(x, format="%Y-%m-%d")
    return pd.Series([date.year, date.month, date.day, date.weekday(), date.week])

def _encoder_dates(X):
    X[["year","month","day","weekday","week"]] = X[
    "DateOfDeparture"].apply(parse_date)
    return X.drop(columns=["DateOfDeparture",'day'])


# DISTANCES

def compute_dist(X):
    dict_lat = {'DFW': 32.89595,'DEN': 39.85841,'LAX': 33.94254,'ORD': 41.9796,
                'SFO': 37.619,'MCO': 28.428890000000003,'LAS': 36.08036,
                'CLT': 35.214009999999995,'MSP': 44.88055,'EWR': 40.6925,
                'PHX': 33.43417,'DTW': 42.21206,'MIA': 25.79325,'BOS': 42.36435,
                'PHL': 39.87195,'JFK': 40.63975,'ATL': 33.640440000000005,
                'LGA': 40.77724,'SEA': 47.44898,'IAH': 29.98047}
    
    dict_long = {'DFW': -97.0372, 'DEN': -104.667, 'LAX': -118.40807, 
                 'ORD': -87.90446, 'SFO': -122.37483999999999, 'MCO': -81.31603, 
                 'LAS': -115.15233, 'CLT': -80.94313000000001, 'MSP': -93.21692, 
                 'EWR': -74.16866, 'PHX': -112.00806000000001, 'DTW': -83.34884, 
                 'MIA': -80.29056, 'BOS': -71.00518000000001, 'PHL': -75.24114, 
                 'JFK': -73.77893, 'ATL': -84.42694, 'LGA': -73.87261, 
                 'SEA': -122.30931000000001, 'IAH': -95.33972}
    
    
    X["Distance"] = X.apply(lambda x : 
                            distance((dict_lat[x["Departure"]],
                                      dict_long[x["Departure"]]),
                                     (dict_lat[x["Arrival"]], 
                                      dict_long[x["Arrival"]])), axis=1)
    
    return X


# DUMMIES

def get_dummy(X):
    X = X.join(pd.get_dummies(X["year"],prefix='year'))
    X = X.join(pd.get_dummies(X["month"],prefix='month'))
    X = X.join(pd.get_dummies(X["Holiday"],prefix='holi'))
    X = X.join(pd.get_dummies(X["week"],prefix='week'))
    X = X.join(pd.get_dummies(X["weekday"],prefix='week_day'))
    
    X["Trajet"] = X["Departure"]+ '-' +  X["Arrival"] 
    X = X.join(pd.get_dummies(X["Trajet"],prefix='traj_'))
    
    
    return X.drop(columns=['year','month','week',
                           'weekday',"week","Departure",'Arrival',
                           "Trajet"])



# FEATURE SELECTION

def _selectfeatures(X):
    selected_features = np.array(['WeeksToDeparture', 'std_wtd', 'Max TemperatureC',
       'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
       'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity',
       'Min Humidity', 'Max Sea Level PressurehPa',
       'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa',
       'Max VisibilityKm', 'Mean VisibilityKm', 'Min VisibilitykM',
       'Max Wind SpeedKm/h', 'Mean Wind SpeedKm/h', 'Precipitationmm',
       'CloudCover', 'WindDirDegrees', 'Holiday', 'Distance', 'year_2011',
       'year_2013', 'month_1', 'month_4', 'month_5', 'month_6', 'month_7',
       'month_8', 'month_9', 'month_12', 'holi_0', 'week_1', 'week_2',
       'week_5', 'week_6', 'week_8', 'week_9', 'week_10', 'week_12',
       'week_14', 'week_16', 'week_17', 'week_18', 'week_20', 'week_21',
       'week_27', 'week_29', 'week_34', 'week_35', 'week_36', 'week_37',
       'week_38', 'week_41', 'week_42', 'week_44', 'week_46', 'week_47',
       'week_49', 'week_50', 'week_51', 'week_52', 'week_day_0',
       'week_day_1', 'week_day_2', 'week_day_3', 'week_day_5',
       'week_day_6', 'traj__ATL-BOS', 'traj__ATL-CLT', 'traj__ATL-DFW',
       'traj__ATL-DTW', 'traj__ATL-LAX', 'traj__ATL-LGA', 'traj__ATL-MCO',
       'traj__ATL-MIA', 'traj__ATL-MSP', 'traj__ATL-ORD', 'traj__ATL-PHL',
       'traj__BOS-ATL', 'traj__BOS-EWR', 'traj__BOS-LGA', 'traj__BOS-ORD',
       'traj__BOS-PHL', 'traj__BOS-SFO', 'traj__CLT-ATL', 'traj__DEN-ATL',
       'traj__DEN-DFW', 'traj__DEN-LAS', 'traj__DEN-MSP', 'traj__DEN-ORD',
       'traj__DEN-SEA', 'traj__DFW-ATL', 'traj__DFW-BOS', 'traj__DFW-DEN',
       'traj__DFW-LAS', 'traj__DFW-LAX', 'traj__DFW-LGA', 'traj__DFW-ORD',
       'traj__DFW-PHL', 'traj__DTW-ATL', 'traj__DTW-LGA', 'traj__DTW-ORD',
       'traj__EWR-BOS', 'traj__EWR-LAX', 'traj__EWR-MCO', 'traj__EWR-ORD',
       'traj__IAH-LAX', 'traj__JFK-LAX', 'traj__JFK-MCO', 'traj__JFK-MIA',
       'traj__JFK-SFO', 'traj__LAS-DEN', 'traj__LAS-DFW', 'traj__LAS-LAX',
       'traj__LAS-ORD', 'traj__LAS-SFO', 'traj__LAX-ATL', 'traj__LAX-DEN',
       'traj__LAX-DFW', 'traj__LAX-IAH', 'traj__LAX-JFK', 'traj__LAX-LAS',
       'traj__LAX-ORD', 'traj__LAX-SFO', 'traj__LGA-ATL', 'traj__LGA-BOS',
       'traj__LGA-DFW', 'traj__LGA-DTW', 'traj__LGA-ORD', 'traj__MCO-ATL',
       'traj__MCO-EWR', 'traj__MCO-PHL', 'traj__MIA-ATL', 'traj__MIA-JFK',
       'traj__MSP-ATL', 'traj__MSP-DEN', 'traj__MSP-ORD', 'traj__ORD-ATL',
       'traj__ORD-BOS', 'traj__ORD-DFW', 'traj__ORD-EWR', 'traj__ORD-IAH',
       'traj__ORD-LAS', 'traj__ORD-LAX', 'traj__ORD-LGA', 'traj__ORD-MSP',
       'traj__ORD-PHL', 'traj__ORD-PHX', 'traj__ORD-SEA', 'traj__ORD-SFO',
       'traj__PHL-ATL', 'traj__PHL-DFW', 'traj__PHL-MCO', 'traj__PHL-ORD',
       'traj__PHX-ORD', 'traj__SEA-DEN', 'traj__SEA-ORD', 'traj__SEA-SFO',
       'traj__SFO-ATL', 'traj__SFO-BOS', 'traj__SFO-JFK', 'traj__SFO-LAS',
       'traj__SFO-LAX', 'traj__SFO-ORD', 'traj__SFO-SEA'])

    return X[selected_features]


def get_estimator():
    warnings.filterwarnings("ignore")
    data_merger = FunctionTransformer(_merge_external_data)
    date_encoder = FunctionTransformer(_encoder_dates)
    dist_ = FunctionTransformer(compute_dist)
    dummy_ = FunctionTransformer(get_dummy)
    lasso_select = FunctionTransformer(_selectfeatures)
    
    regressor = XGBRegressor(n_estimators=200)

    return make_pipeline(data_merger, dist_, date_encoder, dummy_, lasso_select, regressor)
