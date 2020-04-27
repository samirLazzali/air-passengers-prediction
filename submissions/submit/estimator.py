import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
import math
from scipy.stats import mode
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor

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


def parse_date(x):
    date = pd.to_datetime(x, format="%Y-%m-%d")
    return pd.Series([date.year, date.month, date.weekday(), date.week])

def _encoder_dates(X):
    X[["year","month","weekday","week"]] = X[
    "DateOfDeparture"].apply(parse_date)
    X["isweekend"] = X["weekday"] >= 5
    X["isweekend"] = X["isweekend"].astype(int)
    
    return X

def compute_dist(X):  
    external_dico = {'PHL': {'lat': 39.87195, 'long': -75.24114, 
                         'state': 'Pennsylvania', 'altitude': 36, 
                         'tz': 'New_York', 'pop_2011': 5543}, 'DFW': 
                 {'lat': 32.89595, 'long': -97.0372, 'state': 'Texas', 'altitude': 607, 'tz': 'Chicago', 'pop_2011': 15567}, 'IAH': {'lat': 29.98047, 'long': -95.33972, 'state': 'Texas', 'altitude': 97, 'tz': 'Chicago', 'pop_2011': 15567}, 'ATL': {'lat': 33.640440000000005, 'long': -84.42694, 'state': 'Georgia', 'altitude': 1026, 'tz': 'New_York', 'pop_2011': 8082}, 'BOS': {'lat': 42.36435, 'long': -71.00518000000001, 'state': 'Massachusetts', 'altitude': 20, 'tz': 'New_York', 'pop_2011': 63593}, 'LAX': {'lat': 33.94254, 'long': -118.40807, 'state': 'California', 'altitude': 125, 'tz': 'Los_Angeles', 'pop_2011': 64980}, 'SFO': {'lat': 37.619, 'long': -122.37483999999999, 'state': 'California', 'altitude': 13, 'tz': 'Los_Angeles', 'pop_2011': 64980}, 'EWR': {'lat': 40.6925, 'long': -74.16866, 'state': 'New Jersey', 'altitude': 18, 'tz': 'New_York', 'pop_2011': 12957}, 'JFK': {'lat': 40.63975, 'long': -73.77893, 'state': 'New York', 'altitude': 13, 'tz': 'New_York', 'pop_2011': 20688}, 'LGA': {'lat': 40.77724, 'long': -73.87261, 'state': 'New York', 'altitude': 21, 'tz': 'New_York', 'pop_2011': 20688}, 'DTW': {'lat': 42.21206, 'long': -83.34884, 'state': 'Michigan', 'altitude': 645, 'tz': 'New_York', 'pop_2011': 9535}, 'DEN': {'lat': 39.85841, 'long': -104.667, 'state': 'Colorado', 'altitude': 5431, 'tz': 'Denver', 'pop_2011': 13884}, 'CLT': {'lat': 35.214009999999995, 'long': -80.94313000000001, 'state': 'North Carolina', 'altitude': 748, 'tz': 'New_York', 'pop_2011': 9728}, 'MSP': {'lat': 44.88055, 'long': -93.21692, 'state': 'Minnesota', 'altitude': 841, 'tz': 'Chicago', 'pop_2011': 5150}, 'SEA': {'lat': 47.44898, 'long': -122.30931000000001, 'state': 'Washington', 'altitude': 433, 'tz': 'Los_Angeles', 'pop_2011': 15777}, 'ORD': {'lat': 41.9796, 'long': -87.90446, 'state': 'Illinois', 'altitude': 672, 'tz': 'Chicago', 'pop_2011': 8610}, 'MCO': {'lat': 28.428890000000003, 'long': -81.31603, 'state': 'Florida', 'altitude': 96, 'tz': 'New_York', 'pop_2011': 23446}, 'MIA': {'lat': 25.79325, 'long': -80.29056, 'state': 'Florida', 'altitude': 8, 'tz': 'New_York', 'pop_2011': 23446}, 'LAS': {'lat': 36.08036, 'long': -115.15233, 'state': 'Nevada', 'altitude': 2181, 'tz': 'Los_Angeles', 'pop_2011': 81044}, 'PHX': {'lat': 33.43417, 'long': -112.00806000000001, 'state': 'Arizona', 'altitude': 1135, 'tz': 'Phoenix', 'pop_2011': 56061}}

    X["Distance"] = X.apply(lambda x : 
                            distance((external_dico[x["Departure"]]['lat'],
                                      external_dico[x["Departure"]]['long']),
                                     (external_dico[x["Arrival"]]['lat'], 
                                      external_dico[x["Arrival"]]['long'])), axis=1)
    return X



def merge_population (X):   
    external_dico = {'PHL': {'lat': 39.87195, 'long': -75.24114, 
                         'state': 'Pennsylvania', 'altitude': 36, 
                         'tz': 'New_York', 'pop_2011': 5543}, 'DFW': 
                 {'lat': 32.89595, 'long': -97.0372, 'state': 'Texas', 'altitude': 607, 'tz': 'Chicago', 'pop_2011': 15567}, 'IAH': {'lat': 29.98047, 'long': -95.33972, 'state': 'Texas', 'altitude': 97, 'tz': 'Chicago', 'pop_2011': 15567}, 'ATL': {'lat': 33.640440000000005, 'long': -84.42694, 'state': 'Georgia', 'altitude': 1026, 'tz': 'New_York', 'pop_2011': 8082}, 'BOS': {'lat': 42.36435, 'long': -71.00518000000001, 'state': 'Massachusetts', 'altitude': 20, 'tz': 'New_York', 'pop_2011': 63593}, 'LAX': {'lat': 33.94254, 'long': -118.40807, 'state': 'California', 'altitude': 125, 'tz': 'Los_Angeles', 'pop_2011': 64980}, 'SFO': {'lat': 37.619, 'long': -122.37483999999999, 'state': 'California', 'altitude': 13, 'tz': 'Los_Angeles', 'pop_2011': 64980}, 'EWR': {'lat': 40.6925, 'long': -74.16866, 'state': 'New Jersey', 'altitude': 18, 'tz': 'New_York', 'pop_2011': 12957}, 'JFK': {'lat': 40.63975, 'long': -73.77893, 'state': 'New York', 'altitude': 13, 'tz': 'New_York', 'pop_2011': 20688}, 'LGA': {'lat': 40.77724, 'long': -73.87261, 'state': 'New York', 'altitude': 21, 'tz': 'New_York', 'pop_2011': 20688}, 'DTW': {'lat': 42.21206, 'long': -83.34884, 'state': 'Michigan', 'altitude': 645, 'tz': 'New_York', 'pop_2011': 9535}, 'DEN': {'lat': 39.85841, 'long': -104.667, 'state': 'Colorado', 'altitude': 5431, 'tz': 'Denver', 'pop_2011': 13884}, 'CLT': {'lat': 35.214009999999995, 'long': -80.94313000000001, 'state': 'North Carolina', 'altitude': 748, 'tz': 'New_York', 'pop_2011': 9728}, 'MSP': {'lat': 44.88055, 'long': -93.21692, 'state': 'Minnesota', 'altitude': 841, 'tz': 'Chicago', 'pop_2011': 5150}, 'SEA': {'lat': 47.44898, 'long': -122.30931000000001, 'state': 'Washington', 'altitude': 433, 'tz': 'Los_Angeles', 'pop_2011': 15777}, 'ORD': {'lat': 41.9796, 'long': -87.90446, 'state': 'Illinois', 'altitude': 672, 'tz': 'Chicago', 'pop_2011': 8610}, 'MCO': {'lat': 28.428890000000003, 'long': -81.31603, 'state': 'Florida', 'altitude': 96, 'tz': 'New_York', 'pop_2011': 23446}, 'MIA': {'lat': 25.79325, 'long': -80.29056, 'state': 'Florida', 'altitude': 8, 'tz': 'New_York', 'pop_2011': 23446}, 'LAS': {'lat': 36.08036, 'long': -115.15233, 'state': 'Nevada', 'altitude': 2181, 'tz': 'Los_Angeles', 'pop_2011': 81044}, 'PHX': {'lat': 33.43417, 'long': -112.00806000000001, 'state': 'Arizona', 'altitude': 1135, 'tz': 'Phoenix', 'pop_2011': 56061}}


    X["population_Arrival"] = X.apply(lambda x : 
                       external_dico[x["Arrival"]]['pop_2011'], axis=1)
    
    X["population_departure"] = X.apply(lambda x : 
                       external_dico[x["Departure"]]['pop_2011'], axis=1)                
    return X


def _merge_external_data(X):
    __file__ = os.path.join('submissions', 'submit', 'estimator.py')
    filepath = os.path.join(os.path.dirname(__file__), 
                            'external_data.csv')
    
    # Importation des données 
    
    X["DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    X_external = pd.read_csv(filepath, parse_dates=["Date"], index_col=0)
    
    
    # Merge sur les départs de vols
    
    X_Departure = X_external.iloc[:,:-1]
    X_merged_Departure = pd.merge(X, X_Departure, 
                        how='left', 
                        left_on=['DateOfDeparture', 'Departure'],
                        right_on = ["Date","AirPort"])
    X_merged_Departure.drop(columns=["Date","AirPort"], inplace=True)
    
    
    # Merge sur les arrivées de vols 
    
    features = list(X_external.columns)
    del features[-2]
    
    X_Arrival = X_external[features]
    X_merged_Arrival = pd.merge(X, X_Arrival, 
                        how='left', 
                        left_on=['DateOfDeparture', 'Arrival'],
                        right_on = ["Date","AirPort"])

    X_merged_Arrival.drop(columns=["Date","AirPort"], inplace=True)
    
    
    # Merge des deux 
    join = list(X_merged_Arrival.columns)[:13]
    X_out = pd.merge(X_merged_Departure, X_merged_Arrival, 
                     how='left', on=join, suffixes=("_depart",'_arrive'))
    
    X_out.drop(columns=["Holiday_arrive"], inplace=True)
    X_out.rename(columns={"Holiday_depart":'Holiday'}, inplace=True)    
    return X_out


def get_dummy(X):
    X = X.join(pd.get_dummies(X["year"],prefix='year'))
    X = X.join(pd.get_dummies(X["month"],prefix='month'))
    X = X.join(pd.get_dummies(X["Holiday"],prefix='holi'))
    X = X.join(pd.get_dummies(X["week"],prefix='week'))
    X = X.join(pd.get_dummies(X["weekday"],prefix='week_day'))
    X = X.join(pd.get_dummies(X["Events_arrive"],prefix="_arri"))
    X = X.join(pd.get_dummies(X["Events_depart"],prefix="_dep"))
    X = X.join(pd.get_dummies(X["Departure"], prefix='dep'))
    X = X.join(pd.get_dummies(X["Arrival"],prefix='_arri'))
    
    
    
    # Encodage des trajets
    X["Trajet"] = X["Departure"]+ '-' +  X["Arrival"] 
    X = X.join(pd.get_dummies(X["Trajet"],prefix='traj_'))
    
    return X.drop(columns=['year','month','week',
                           'weekday',"week","Departure",'Arrival',
                           "Trajet",'DateOfDeparture',
                           'Events_depart',"Events_arrive"])


def soustraction(X):
    var_depart = []
    var_arriv = []

    for i in X.columns:
        if i[-7:] == "_depart":
            var_depart.append(i)
        if i[-7:] == '_arrive':
            var_arriv.append(i)
        else :
            pass
    
    for i,j in zip(var_depart,var_arriv):
        X[str(i[:-7])] = X[i] - X[j]
    return X


def _selectfeatures(X):

    selected_features = np.array(['WeeksToDeparture', 'std_wtd', 'isweekend', 'Distance',
       'population_Arrival', 'population_departure',
       'Max TemperatureC_depart', 'Mean TemperatureC_depart',
       'Min TemperatureC_depart', 'Dew PointC_depart',
       'MeanDew PointC_depart', 'Min DewpointC_depart',
       'Mean Humidity_depart', 'Max Sea Level PressurehPa_depart',
       'Mean Sea Level PressurehPa_depart',
       'Min Sea Level PressurehPa_depart', 'Max Wind SpeedKm/h_depart',
       'Mean Wind SpeedKm/h_depart', 'Max Gust SpeedKm/h_depart',
       'WindDirDegrees_depart', 'Holiday', 'TOTAL_DEPARTURE',
       'Max TemperatureC_arrive', 'Mean TemperatureC_arrive',
       'Min TemperatureC_arrive', 'Dew PointC_arrive',
       'MeanDew PointC_arrive', 'Min DewpointC_arrive',
       'Max Humidity_arrive', 'Mean Humidity_arrive',
       'Min Humidity_arrive', 'Max Sea Level PressurehPa_arrive',
       'Mean Sea Level PressurehPa_arrive',
       'Min Sea Level PressurehPa_arrive', 'Min VisibilitykM_arrive',
       'Max Wind SpeedKm/h_arrive', 'Mean Wind SpeedKm/h_arrive',
       'Max Gust SpeedKm/h_arrive', 'WindDirDegrees_arrive',
       'TOTAL_DESTINATION', 'year_2011', 'year_2012', 'year_2013',
       'month_12', 'week_1', 'week_10', 'week_14', 'week_27', 'week_35',
       'week_37', 'week_44', 'week_47', 'week_51', 'week_52',
       'week_day_0', 'week_day_1', 'week_day_2', 'week_day_3',
       'week_day_4', 'week_day_5', 'week_day_6', 'dep_BOS', 'dep_LGA',
       'dep_MIA', 'dep_ORD', '_arri_LAX', '_arri_LGA', '_arri_ORD',
       'traj__ATL-PHL', 'traj__DFW-LAX', 'traj__JFK-LAX', 'traj__JFK-MCO',
       'traj__JFK-SFO', 'traj__LAX-JFK', 'traj__LGA-ORD', 'traj__ORD-LGA',
       'traj__SFO-ATL', 'traj__SFO-JFK', 'Max TemperatureC',
       'Mean TemperatureC', 'Min TemperatureC', 'MeanDew PointC',
       'Min DewpointC', 'Min Humidity', 'Max Sea Level PressurehPa',
       'Mean Sea Level PressurehPa', 'Mean Wind SpeedKm/h',
       'Max Gust SpeedKm/h'])

    return X[selected_features]

def get_estimator():

    warnings.filterwarnings("ignore")


    data_merger = FunctionTransformer(_merge_external_data)
    date_encoder = FunctionTransformer(_encoder_dates)
    dist_ = FunctionTransformer(compute_dist)
    dummy_ = FunctionTransformer(get_dummy)
    soustraction__ = FunctionTransformer(soustraction)
    popu = FunctionTransformer(merge_population)
    lasso_select = FunctionTransformer(_selectfeatures)
    
    regressor = LGBMRegressor()

    return make_pipeline(date_encoder, dist_, popu, data_merger,dummy_, soustraction__,regressor)

