import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import pygmt
import random

import requests
import datetime as dt
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry

import time
import pickle

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

if_weather = 0

# getting fire data

df = pd.read_csv('data/1ad52b0980f34045b1cc16faf61103eb.csv')
df = df.drop(['X','Y','id','iso2','iso3','country','noneu','area_code','eu_area','updated','map_source','admlvl2'], axis=1)
df = df.rename(columns={'admlvl3': 'department', 'admlvl5': 'community', 'admlvl1': 'region'})
df.head()

l = len(df)
df = df.dropna(axis=0, how='any')
print(f"dropped {(l-len(df))/l*100:.2f}%")

# editing time formats

def parse_datetime(row):
    try:
        try:
            return pd.to_datetime(row, format='%Y/%m/%d %H:%M:%S.%f%z')
        except ValueError:
            return pd.to_datetime(row, format='%Y/%m/%d %H:%M:%S.%f')
    except ValueError:
        try:
            return pd.to_datetime(row, format='%Y/%m/%d %H:%M:%S%z')
        except ValueError:
            return pd.to_datetime(row, format='%Y/%m/%d %H:%M:%S')

df['start'] = df['initialdate'].apply(parse_datetime)
df['end'] = df['finaldate'].apply(parse_datetime)

df.loc[:,'year']=df['start'].dt.year
df.loc[:,'month']=df['start'].dt.month

# deleting instances prior to Jan 2015

l = len(df)
df = df[df['start']>pd.to_datetime('2016-01-15')]
print(f"dropped {(l-len(df))/l*100:.2f}%")

# get communities

df_geo = pd.read_csv('data/communes-departement-region.csv')
df_geo['code_commune_INSEE'] = df_geo['code_commune_INSEE'].apply('{:0>5}'.format)

df_geo = df_geo.drop(['nom_commune_postal','libelle_acheminement','ligne_5','article','nom_commune_complet', 'code_postal','code_commune', 'code_departement','code_region'], axis=1)
df_geo = df_geo.rename(columns={'nom_commune': 'community', 'nom_departement': 'department', 'nom_region': 'region'})
df_geo.head()

l=len(df_geo)
df_geo = df_geo.dropna(axis=0, how='any')
print(f"dropped NaNs {(l-len(df_geo))/l*100:.2f}%")
l=len(df_geo)
df_geo.drop_duplicates(inplace=True)
print(f"dropped duplicates {(l-len(df_geo))/l*100:.2f}%")

population = pd.read_csv('data/population_municipale.csv',delimiter=';', skiprows=2)
l=len(population)
population = population[~population.isin(["N/A - résultat non disponible"]).any(axis=1)]
print(f"dropped {(l-len(population))/l*100:.2f}%")
population.head()

# population['Code'] = pd.to_numeric(population['Code'], errors='coerce').astype('Int64')
population['Population municipale 2021'] = pd.to_numeric(population['Population municipale 2021'], errors='coerce').astype('Int64')
population.drop(columns='Libellé', axis=0, inplace=True)
population.rename(columns={'Code':'code_commune_INSEE','Population municipale 2021':'population'}, inplace=True)
df_geo = pd.merge(df_geo, population, on=['code_commune_INSEE'])
df_geo.head()

df_geo.drop(columns='code_commune_INSEE', inplace=True)

# combining data sets

for col in ['community','department','region']:
    df[col] = df[col].str.replace('’', "'", regex=False)
    df[col] = df[col].str.replace('Ile', "Île", regex=False)
    df[col] = df[col].str.replace('Centre — Val', "Centre-Val", regex=False)

df.loc[(df['community'] == 'Arbéost') & (df['region'] == 'Nouvelle-Aquitaine'), 'department'] = 'Hautes-Pyrénées'
df.loc[(df['community'] == 'Arbéost') & (df['region'] == 'Nouvelle-Aquitaine'), 'region'] = 'Occitanie'
df.loc[(df['community'] == 'Cans et Cévennes'), 'community'] = 'Florac Trois Rivières'
df.loc[(df['community'] == 'Bors (Canton de Tude-et-Lavalette)'), 'community'] = 'Bors Canton de Tude-et-Lavalette'
df.loc[(df['community'] == 'Marseille'), 'community'] = 'Marseille 01'

def checking_for_lost_data(df, df_geo):
    not_found = []

    for i, row in df.iterrows():
        community = row['community']
        department = row['department']
        region = row['region']

        # Find matching row in df2
        matching_rows = df_geo[(df_geo['community'] == community) & (df_geo['department'] == department) & (df_geo['region'] == region)]

        if not matching_rows.empty:
            for _, match_row in matching_rows.iterrows():
                if match_row['department'] == department and match_row['region'] == region:
                    # Assign latitude and longitude
                    df.at[i, 'lat'] = match_row['latitude']
                    df.at[i, 'lon'] = match_row['longitude']
                else:
                    print(f"Discrepancy found for '{i}' community '{community}':")
                    print(f"  df -> Department: {department}, Region: {region}")
                    print(f"  df_geo -> Department: {match_row['department']}, Region: {match_row['region']}")
        else:
            # print(f"No match found for community '{community}' in df.")
            not_found.append([community, department, region])
            print(len(not_found))

    #print("\nUpdated df with latitude and longitude:")
    #print(df)
    return not_found

# not_found = checking_for_lost_data(df, df_geo)
# pd.DataFrame(not_found).drop_duplicates()

df_enriched = pd.merge(df, df_geo, on=['community','department','region'], how='left')

df_enriched.drop(['end'], axis=1, inplace=True)

# getting weather data

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

def get_weather(lat,lon,start,end):
	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": lat,
		"longitude": lon,
		"start_date": start,
		"end_date": end,
		"daily": ["temperature_2m_max", "temperature_2m_min", "daylight_duration", "precipitation_sum", "wind_speed_10m_max", "wind_direction_10m_dominant"]
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
	daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
	daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
	daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()
	daily_wind_speed_10m_max = daily.Variables(4).ValuesAsNumpy()
	daily_wind_direction_10m_dominant = daily.Variables(5).ValuesAsNumpy()

	daily_data = {"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)}
	daily_data["temperature_2m_max"] = daily_temperature_2m_max
	daily_data["temperature_2m_min"] = daily_temperature_2m_min
	daily_data["daylight_duration"] = daily_daylight_duration
	daily_data["precipitation_sum"] = daily_precipitation_sum
	daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
	daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant

	return pd.DataFrame(data = daily_data)

df_enriched['initialdate']=(df_enriched.start - pd.Timedelta(days=14)).dt.strftime('%Y-%m-%d')
df_enriched['finaldate']=df_enriched.start.dt.strftime('%Y-%m-%d')

cols = ["temperature_2m_max", "temperature_2m_min", "daylight_duration", "precipitation_sum", "wind_speed_10m_max", "wind_direction_10m_dominant"]

if if_weather:

    for i, row in df_enriched.iterrows():
        print(f"getting {i} of {length}...")
        lat,lon,s,e = row[['latitude','longitude','initialdate','finaldate']]
        weather = get_weather(lat, lon, s, e)
        for col in cols:
            df_enriched.at[i, col] = weather[col].mean()
        time.sleep(3) # to avoid time out of API

if if_weather:

    with open('data/weather.pickle', 'wb') as handle:
        pickle.dump(df_enriched, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    df = df_enriched

# creating zero data

with open('data/weather.pickle', 'rb') as handle:
    df = pickle.load(handle)

df = df.sort_values(by='start')
df['dtime'] = df['start'].diff()
print(df['dtime'].mean(), df['dtime'].min(), df['dtime'].max())

unique_cols = ['community', 'department', 'region']
df_unique = df.drop_duplicates(subset = unique_cols)
print(f"Percentage of communities with recorded forest fires is {len(df_unique)/ len(df_geo)*100:.2f}%.")

# The reference date for when to start calculating non-fire periods
start_date = pd.to_datetime('2015-01-01')
end_date = pd.to_datetime('2024-06-30')
community_diff = 23
random_places = 200

unique_geo = df_geo[['community', 'department', 'region', 'latitude', 'longitude']].drop_duplicates(subset=['community','department','region'])

def addvalues(results, community, department, region, latitude, longitude, current_date):
    results.append({
                'community': community,
                'department': department,
                'region': region,
                'latitude': latitude,
                'longitude': longitude,
                'fire_date': current_date,
                'fire_status': 'fire'
            })
    return results

results = []

random_communities = unique_geo.sample(n=random_places, random_state=42)

# Iterate over the unique combinations
# for i, row in unique_geo.iloc[::community_diff].iterrows():
for i, row in random_communities.iterrows():
    community, department, region, latitude, longitude = row['community'], row['department'], row['region'], row['latitude'], row['longitude']
    # Get the fire dates for the specific community
    community_fires = df[(df['community'] == community) & (df['department'] == department) & (df['region'] == region)].sort_values(by='start')

    # If there are no fires recorded since 2015, take the whole period since 2015
    if community_fires.empty:
        # print('made-up fires in', community)
        current_date = start_date + pd.DateOffset(days=random.uniform(21,90))
        while (current_date <= end_date):
            results = addvalues(results, community, department, region, latitude, longitude, current_date)
            # print('made-uo fire in', community, 'at', current_date)
            current_date += pd.DateOffset(days=random.uniform(21,180))

    else:
        current_date = start_date + pd.DateOffset(days=random.uniform(21,90))
    #     print('fires in', community)
        # Add the period from 2015 to the first fire date (if there's a gap)
        if community_fires.iloc[0]['start'] > current_date:
            results = addvalues(results, community, department, region, latitude, longitude, current_date)

        # Add periods between fires
        for i in range(1, len(community_fires)):
            if community_fires.iloc[i]['start'] - community_fires.iloc[i-1]['start'] > pd.to_timedelta("90 days"):
                current_date += pd.DateOffset(days=random.uniform(21,90))
                results = addvalues(results, community, department, region, latitude, longitude, current_date)

        # Add the period from the last fire date to now (if there's a gap)
        if community_fires.iloc[-1]['start'] < end_date - pd.to_timedelta("90 days"):
            current_date += pd.DateOffset(days=random.uniform(21,90))
            results = addvalues(results, community, department, region, latitude, longitude, current_date)

# Convert the results list to a DataFrame
df_no_fires = pd.DataFrame(results)

df_no_fires = df_no_fires[df_no_fires['latitude'] > 40]

print(f"ratio between fire data and non-fire data: {len(df)/len(df_no_fires):.2f}")

# fill no fire with zero data for burnt areas

new_columns = ['broadleaved_forest_percent', 'coniferous_forest_percent',
       'mixed_forest_percent', 'sclerophillous_vegetation_percent',
       'transitional_vegetation_percent', 'other_natural_percent',
       'agriculture_percent', 'artificial_percent', 'other_percent',
       'natura2k_percent']

for col in new_columns:
    df_no_fires[col] = df['longitude'].fillna(0)

df_no_fires['initialdate']=(df_no_fires.fire_date - pd.Timedelta(days=14)).dt.strftime('%Y-%m-%d')
df_no_fires['finaldate']=df_no_fires.fire_date.dt.strftime('%Y-%m-%d')

cols = ["temperature_2m_max", "temperature_2m_min", "daylight_duration", "precipitation_sum", "wind_speed_10m_max", "wind_direction_10m_dominant"]

# getting weather for zero points

length = df_no_fires.shape[0]

if if_weather:

    for i, row in df_no_fires.iterrows():
        print(f"getting {i} of {length}...")
        lat,lon,s,e = row[['latitude','longitude','initialdate','finaldate']]
        weather = get_weather(lat, lon, s, e)
        for col in cols:
            df_no_fires.at[i, col] = weather[col].mean()
        time.sleep(3) # to avoid time out of API
    
if if_weather:

    forest_data_cols = ['broadleaved_forest_percent',
       'coniferous_forest_percent', 'mixed_forest_percent',
       'sclerophillous_vegetation_percent', 'transitional_vegetation_percent',
       'other_natural_percent', 'agriculture_percent', 'artificial_percent',
       'other_percent', 'natura2k_percent']
    for col in forest_data_cols:
        df_no_fires[col]=0

    with open('data/weather_nofires.pickle', 'wb') as handle:
        pickle.dump(df_no_fires, handle, protocol=pickle.HIGHEST_PROTOCOL)


# loading data set

with open('data/weather.pickle', 'rb') as handle:
    df = pickle.load(handle)
with open('data/weather_nofires.pickle', 'rb') as handle:
    df_no_fires = pickle.load(handle)

df['fire'] = 1

df_no_fires['area_ha']=0
df_no_fires['fire']=0
df_no_fires['year'] = pd.to_datetime(df_no_fires['initialdate']).dt.year
df_no_fires['month'] = pd.to_datetime(df_no_fires['initialdate']).dt.month
df_no_fires.drop(columns={'fire_status'}, inplace=True)
df_no_fires.rename(columns={'fire_date':'start'}, inplace=True)
# df.drop(columns={'start'}, inplace=True)

missing_in_df_no_fires = [col for col in df if col not in df_no_fires]
missing_in_df = [col for col in df_no_fires if col not in df]
print(missing_in_df, missing_in_df_no_fires)

# resorting the columns to be the same
cols_df = df.columns
df_no_fires = df_no_fires[cols_df]

df_combined = pd.concat([df, df_no_fires], ignore_index=True)

# introducing neighboring fires

from geopy.distance import geodesic

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

from scipy.spatial import cKDTree

# Step 1: Filter the DataFrame to include only rows where fire == 1
df_fire = df_combined[df_combined['fire'] == 1].copy()

# Step 2: Initialize the 'neighbor' column to False in the filtered DataFrame
df_fire['neighbor'] = False
l = len(df_fire)

# Step 3: Pre-process the data by creating a KD-Tree for spatial queries
coordinates = np.array(list(zip(df_fire['latitude'], df_fire['longitude'])))
tree = cKDTree(coordinates)

# Step 4: Iterate over the fire events in the filtered DataFrame
for i, row in df_fire.iterrows():
    
    # Step 4.1: Get all points within a 10 km radius using the KD-Tree
    idx = tree.query_ball_point([row['latitude'], row['longitude']], r=10/6371)  # Normalize by Earth's radius
    
    # Step 4.2: Iterate through the indices of nearby points
    for j in idx:
        if i != j:  # Avoid self-comparison
            other_row = df_fire.iloc[j]
            
            # Step 4.3: Calculate the time difference in days
            time_diff = (row['start'] - other_row['start']).days
            
            # Step 4.4: Check if the time difference is <= 3 days
            if time_diff <= 3 and time_diff > 0:
                df_fire.at[i, 'neighbor'] = True
                break  # No need to check further once a neighbor is found

# Step 5: Merge the 'neighbor' results back into the original DataFrame
df_combined['neighbor']=False
df_combined.update(df_fire[['neighbor']])

print(df_combined['neighbor'].value_counts())

# add population if not existent before

df_combined['lat_bin'] = df_combined['latitude'].multiply(5).round().multiply(.2)  # Create bins for latitude
df_combined['lon_bin'] = df_combined['longitude'].multiply(5).round().multiply(.2)  # Create bins for longitude

df_grid = df_combined[['latitude', 'longitude', 'population']].copy()
grid_aggregated = df_grid.groupby(['latitude', 'longitude']).agg({
        "population": 'sum'
    }).reset_index()
df_combined = df_combined.merge(grid_aggregated, on=['lat_bin', 'lon_bin'])

with open('data/complete_dataset.pickle', 'wb') as handle:
        pickle.dump(df_combined, handle, protocol=pickle.HIGHEST_PROTOCOL)

