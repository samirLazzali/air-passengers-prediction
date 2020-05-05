import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
import math
from scipy.stats import mode
import numpy as np
import warnings
from sklearn.ensemble import GradientBoostingRegressor
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
    return X

date_encoder = FunctionTransformer(_encoder_dates)

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
    """
    Pour ajouter une donnée indexé par les aeroport et le temps : Première partie
    Pour ajouter une donnée indéxé par le temps : Deuxième partie
    """
    
    # Importation des données et parsage de la date
    
    filepath = os.path.join(os.path.dirname(__file__), 
                            'external_data.csv')
    X["DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    X_external = pd.read_csv(filepath, parse_dates=["Date"], index_col=0)
    
    
    # 1 – Partie : Merge des données indexées par les départs 
    
    X_Departure = X_external[['Date', 'AirPort', "Net_domestic_migration_rate_per_1000_hab",
                               'Min TemperatureC',"city_population","Median_income",
                              'CloudCover','TOTAL_DEPARTURE', "unemployment"]]
    
    X_merged_Departure = pd.merge(X, X_Departure, 
                        how='left', 
                        left_on=['DateOfDeparture', 'Departure'],
                        right_on = ["Date","AirPort"])
    X_merged_Departure.drop(columns=["Date","AirPort"], inplace=True)
    
    # 1 – Partie : Merge sur les données indéxées par les arrivées 
    
    X_Arrival = X_external[['Date', 'AirPort', "Net_domestic_migration_rate_per_1000_hab",
                               'Min TemperatureC',"city_population", "Median_income",
                              'CloudCover','TOTAL_DESTINATION','unemployment']]

    X_merged_Arrival = pd.merge(X, X_Arrival, 
                        how='left', 
                        left_on=['DateOfDeparture', 'Arrival'],
                        right_on = ["Date","AirPort"])

    X_merged_Arrival.drop(columns=["Date","AirPort"], inplace=True)
    
    
    # Merge des deux tables (variable départ-arrivé) 
    # sur (join) notre table en entrée (il faut faire attention à l'ordre de la pipeline)
    
    join = ['DateOfDeparture', 'Departure', 'Arrival','WeeksToDeparture',
            "std_wtd","year","month","weekday","week","Distance",
            "population_Arrival","population_departure"]
    
    X_out = pd.merge(X_merged_Departure, X_merged_Arrival, 
                     how='left', on=join, suffixes=("_depart",'_arrive'))
    
    
    # 2 Partie – On ajoute les données seulement indexées par le temps 
    
    X_out = X_out.merge(X_external[["Date", "AirPort","Events","Holiday",
                            'INDEX_PRICE','Index_passengers','Price']], how='left',
                        left_on=["DateOfDeparture","Arrival"], right_on=["Date","AirPort"])
      
    
    return X_out.drop(columns=['Date','AirPort'])



def get_dummy(X):
    X = X.join(pd.get_dummies(X["Events"]))
    X = X.join(pd.get_dummies(X["Departure"], prefix='_dep'))
    X = X.join(pd.get_dummies(X["Arrival"],prefix='_arri'))
    
    return X.drop(columns=["Events","Departure",'Arrival',
                           'DateOfDeparture'])


def _selectfeatures(X):
  
	array(['WeeksToDeparture', 'std_wtd', 'year', 'month', 'weekday', 'week',
       'Distance', 'population_Arrival', 'population_departure',
       'Net_domestic_migration_rate_per_1000_hab_depart',
       'Dew PointC_depart', 'city_population_depart',
       'Median_income_depart', 'CloudCover_depart', 'TOTAL_DEPARTURE',
       'Net_domestic_migration_rate_per_1000_hab_arrive',
       'Dew PointC_arrive', 'city_population_arrive',
       'Median_income_arrive', 'CloudCover_arrive', 'TOTAL_DESTINATION',
       'Holiday', 'Index_passengers', 'Cloud', '_dep_DFW', '_dep_DTW',
       '_dep_EWR', '_dep_IAH', '_dep_JFK', '_dep_LAX', '_dep_LGA',
       '_dep_MIA', '_dep_ORD', '_dep_SFO', '_arri_DFW', '_arri_DTW',
       '_arri_EWR', '_arri_IAH', '_arri_JFK', '_arri_LAX', '_arri_LGA',
       '_arri_PHX'])
	return X[selected_features]


def get_estimator():

    warnings.filterwarnings("ignore")

    data_merger = FunctionTransformer(_merge_external_data)
    date_encoder = FunctionTransformer(_encoder_dates)
    dist_ = FunctionTransformer(compute_dist)
    dummy_ = FunctionTransformer(get_dummy)
    popu = FunctionTransformer(merge_population)
    lasso_select = FunctionTransformer(_selectfeatures)
    
    regressor = LGBMRegressor(num_leaves = 45, n_estimators = 2000)

    return make_pipeline(date_encoder, dist_, popu, data_merger, dummy_, regressor)