import numpy as np
import math

# Function to calculate the Haversine distance between two points
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    # Differences
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def lat_long_to_euclidean(lat, lon):
    R = 6371  # Radius of the Earth in kilometers
    x = R * np.radians(lon) * np.cos(np.radians(lat))
    y = R * np.radians(lat)
    return x, y


def euclidean_to_lat_long(x, y):
    R = 6371  # Radius of the Earth in kilometers
    lat = np.degrees(y / R)
    R_lat = R * np.cos(np.radians(lat))
    lon = np.degrees(x / R_lat)
    return lat, lon


def populate_location_list(location_list, location_ids, latitudes, longitudes, demands):
  for location_id, lat, long, demand in zip(location_ids, latitudes, longitudes, demands):
      location_list.append({'id': location_id, 'lat': lat, 'long': long, 'demand': demand})
