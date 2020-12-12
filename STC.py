#cell 0
import pandas as pd
import numpy as np
from sodapy import Socrata
import datetime
import time
from matplotlib import pyplot as plt

#cell 1
import pytz
eastern_tz = pytz.timezone('America/New_York')
def stcDateToUTC(year, week, day, hour):
    # print("%s-%s-%s-%s" % (year, week, day, hour))

    date = datetime.date.fromisocalendar(int(year), int(week), int(day)+1)
    # print(date)
    # dt = edatetime.datetime(date.year, date.month, date.day, hour=int(hour), tzinfo=pytz.utc)
    dt = eastern_tz.localize(datetime.datetime(date.year, date.month, date.day, hour=int(hour)))
    # dt = datetime.datetime(date.year, date.month, date.day, hour=int(hour))
    return dt.timestamp()


def get_filename2stcfeat(file_path="./dataset/annotations.csv"):
    """
        Build filename to stc feature. 
    """
    data = pd.read_csv(file_path, dtype=object)
    audio_filenames = data["audio_filename"]
    split = data["split"]
    sensor_ids = data["sensor_id"]
    boroughs = data["borough"]
    blocks = data["block"]
    latitudes = data["latitude"]
    longitudes = data["longitude"]
    years = data["year"]
    weeks = data["week"]
    days = data["day"]
    hours = data["hour"]

    columns = ['split', 'sensor_id', 'borough', 'block', 'latitude', 'longitude', 'year', 'week', 'day', 'hour', 'unix_timestamp']

    filename2stcdfeat = {}
    for i in range(len(audio_filenames)):


        filename2stcdfeat[audio_filenames[i]] = [
            split[i],                          
            sensor_ids[i],
            boroughs[i],
            blocks[i],
            float(latitudes[i]),
            float(longitudes[i]),
            int(years[i]),
            int(weeks[i]),
            int(days[i]),
            int(hours[i]),
            stcDateToUTC(years[i], weeks[i], days[i], hours[i])
        ]

    db = pd.DataFrame.from_dict(filename2stcdfeat, orient='index', columns=columns)

    return columns, db



#cell 2
# NUM_CTX_FEATURES = 8
stc_columns, stc_metadata = get_filename2stcfeat()
stc_feature_columns = ['DOB', 'collision_hour', 'collision_3hours', 'nearby_subway', 'tax_lot_residential', 'tax_lot_commercial', 'tax_lot_manufacturing', 'tax_lot_misc']
stc_features = pd.DataFrame(data=np.zeros((len(stc_metadata), len(stc_feature_columns))), columns=stc_feature_columns, index=stc_metadata.index)
# stc_features = np.zeros((len(stc_metadata), NUM_CTX_FEATURES))

#cell 3
#CONSTANTS for STC features
LATITUDE_TOLERANCE = 0.00075 #decimal degrees. Use +- this for areas. Corresponds to +-~250 ft
LONGITUDE_TOLERANCE = 0.0005 #decimal degrees. Use +- this for areas. Corresponds to +-~250 ft

DATASET_START_TIME = 1456837200 #anything later than this UNIX timestamp is valid to look out; filter based on this for starters

#### Ignore below; indices for features in an array, but moving towards dataframe to allow names for features vs. knowing indices.
# #binary features 
# DOB_FEATURE_IND = 0
# CAR_COLLISION_LAST_HOUR_FEATURE_IND = 1
# CAR_COLLISION_LAST_3_HOURs_FEATURE_IND = 2
# ACTIVE_CONSTRUCTION_FEATURE_IND = 3
# NEARBY_SUBWAY_FEATURE_IND = 4
# #percentages
# TAX_LOT_RESIDENTIAL_FEATURE_IND = 5
# TAX_LOT_COMMERCIAL_FEATURE_IND = 6
# TAX_LOT_MANUFACTURING_FEATURE_IND = 7
# TAX_LOT_MISC_FEATURE_IND = 8



#cell 4
from sodapy import Socrata

socrata_domain_opennyc = 'data.cityofnewyork.us'
socrata_app_token = '2IfDglt97FXtZ4oDNBaTTu7WP'

dob_permit_issuance_rsrc  = 'ipu4-2q9a'
car_collisions_rsrc = 'h9gi-nx95'
active_construction_rsrc = '8586-3zfm'
subway_entrance_rsrc = 'drex-xx56'


# client = Socrata(socrata_domain_opennyc, socrata_app_token)
client = Socrata(socrata_domain_opennyc, socrata_app_token)

#cell 5
def createFrameFromOpendata(client, resource, column_items=[]):
    """
    client: a socrata client that points to the top level of a set of databases
    resource: the string that identifies the resource within the client
    column_items: the set of items to keep in the entire pandas frame/database
    """
    data = client.get(resource, limit=2)
    print(data)
    big_db = pd.DataFrame.from_dict(data)
    if len(column_items) > 0:
        print('filter)')
        big_db = big_db.filter(items=column_items)
    print(big_db)

    offset = big_db.shape[0]

    while True:
        try:
            print(offset)
            data = client.get(resource, limit=50000, offset = offset)
            db = pd.DataFrame.from_dict(data)

            if len(column_items) > 0:
                db = db.filter(items=column_items)
                
            if db.empty:
                print('Done!')
                break

            big_db = big_db.append(db)
            # print(big_db.shape)
            offset += db.shape[0]
        except RuntimeError:
            print('error')
            break


    return big_db

#cell 6
## Department of Buildings Permit Issuance Feature
Use to detect classes like saws, machinery impacts, etc. that would be present while a building is being altered, constructed, or demolished.
Basd on what the job is, the weighting of this feature will change; the mean of thosse weightings is used if there are multiple ongoing jobs in a geographic region near the sensor (about 400x400ft area) when the audio sampled was taken


#cell 7
dob_items = ['issuance_date', 'expiration_date', 'job_start_date', 'gis_latitude', 'gis_longitude', 'job_type']
dob_db = createFrameFromOpendata(client, dob_permit_issuance_rsrc, dob_items)

#cell 8
import datetime
import time
import pytz
eastern_tz = pytz.timezone('America/New_York')
def convertDates(date, axis=0, result_type=''):
    # a date string, whose format may vary. This is only for the date; time within that day is not considered
    # print(date)
    if type(date) == int or type(date) == float:
        return date #leave unchanged if it has already been parsed as a date
    elif not type(date) == str:
        return 0

    if '/' in date:
        try:
            return time.mktime(eastern_tz.localize(datetime.datetime.strptime(date, '%Y/%m/%d')).timetuple())        
        except ValueError:
            try:
                return time.mktime(eastern_tz.localize(datetime.datetime.strptime(date, '%m/%d/%Y')).timetuple())
            except ValueError:
                return 0
        except: 
            return 0
    elif '-' in date:   
        try:
            return time.mktime(eastern_tz.localize(datetime.datetime.strptime(date, '%Y-%m-%d')).timetuple())        
        except ValueError:
            try:
                return time.mktime(deastern_tz.localize(datetime.datetime.strptime(date, '%m-%d-%Y')).timetuple())
            except ValueError:
                return 0
        except: 
            return 0
    else:
        print("Warning; Unsure how to handle this case: %s" % str(date))
        return date

def convertTimes(time_str, axis=0, result_type=''):
    # print(time_str)
    if type(time_str) == float or type(time_str) == int:
        return time_str
    elif not type(time_str) == str:
        return 0

    try: 
        split_str = time_str.split(':')
        hour = int(split_str[0])
        minute = int(split_str[1])
        td = datetime.timedelta(hours=hour, minutes=minute)
        # print(td)
        return  td.total_seconds()      
    except TypeError:
        print('Error on time: %s' % time_str)
        return 0


    

def jobTypeMultiplier(job_type, axis=0, result_type=''):

    if job_type == 'A1':
        return 0.25
    elif job_type == 'A2':
        return 0.5
    elif job_type == 'A3':
        return 0.75
    elif job_type == 'SG':
        return 0.5
    elif job_type == 'NB':
        return 1
    elif job_type == 'DM':
        return 1
    else: 
        return 0

#cell 9
#convert dates to a time-value so I can do a filter by dates
DATASET_START_TIME = 1456837200 #anything later than this UNIX timestamp is valid to look out; filter based on this for starters

#convert the column with dates to timestamps to allow numeric comparison
if type(dob_db.iloc[0]['job_start_date']) == str: 
    dob_db['job_start_date'] = dob_db['job_start_date'].apply(convertDates, axis=1, result_type='broadcast') #convert dates to timestamps for numerical comparison
if type(dob_db.iloc[0]['expiration_date']) == str:
    dob_db['expiration_date'] = dob_db['expiration_date'].apply(convertDates, axis=1, result_type='broadcast') #convert dates to timestamps for numerical comparison
if type(dob_db.iloc[0]['gis_latitude']) == str:
    dob_db['gis_latitude'] = dob_db['gis_latitude'].apply(lambda x: float(x))  #force latitude to be a floating point for easier comparison
    print(dob_db['gis_latitude'])
if type(dob_db.iloc[0]['gis_longitude']) == str:
    dob_db['gis_longitude'] = dob_db['gis_longitude'].apply(lambda x: float(x)) #force longtitude to be a floating point for easier comparison
    print(dob_db['gis_longitude'])


dob_db['job_multiplier'] = dob_db['job_type'].apply(jobTypeMultiplier)

#cell 10
#filter the list to only include those relevant by date
dob_db_filt = dob_db[dob_db['job_start_date'] > DATASET_START_TIME]
print(dob_db_filt)

#cell 11

#iterate over the stc_metadata (a pandas frame) to fill in the feature for DOB. Filter the (filtered) data frame by timestamp and location
num_nonzero = 0
for filename, stc in stc_metadata.iterrows():
    #filter by location
    lat_db = dob_db_filt['gis_latitude']
    lon_db = dob_db_filt['gis_longitude']
    lat_ind = np.logical_and(lat_db >= (stc['latitude'] - LATITUDE_TOLERANCE), lat_db <= (stc['latitude'] + LATITUDE_TOLERANCE))
    lon_ind = np.logical_and(lon_db >= (stc['longitude'] - LONGITUDE_TOLERANCE), lon_db <= (stc['longitude'] + LONGITUDE_TOLERANCE))
    # location_ind = np.logical_and(np.asarray(lat_ind), np.asarray(lon_ind))
    location_ind = np.logical_and(lat_ind, lon_ind)
    valid_loc_db = dob_db_filt[location_ind]
    
    #filter by time
    timestamp = stc['unix_timestamp']
    start = valid_loc_db['job_start_date']
    end = valid_loc_db['expiration_date']

    start_ind = np.asarray(start <= timestamp)
    end_ind = np.asarray(end >= timestamp)
    time_ind = np.logical_and(start_ind, end_ind)

    valid_db = valid_loc_db[time_ind]

    if valid_db.empty:
        stc_features.loc[filename,'DOB'] = 0
        # print('no construction here: %s' % filename)
        num_nonzero += 1

    else: 
        # print(valid_db.shape[0])
        stc_features.loc[filename,'DOB'] = valid_db.loc[:,'job_multiplier'].mean() 


print('Number of samples without an ongoing dept. of buildings permit: %d ' % num_nonzero)


#cell 12
#analyze DOB feature
dob_feature = np.asarray(stc_features.loc[:,'DOB'])
plt.hist(dob_feature, bins=100)
print('DOB Feature mean and standard deviation: %f, %f' % (np.mean(dob_feature), np.std(dob_feature)))

#cell 13
## Car Collisions Feature
Hopefully use this to find non-macheriny impacts, people talking/yelling/shouting (This is NYC, after all)

#cell 14
collisions_columns = ['latitude', 'longitude', 'crash_date', 'crash_time', ]
collisions_db = createFrameFromOpendata(client, car_collisions_rsrc, collisions_columns)
collisions_db_OG = collisions_db.copy()
# collisions_data = client.get(car_collisions_rsrc, limit=10)
# print(collisions_data)
# collisions_db = pd.DataFrame.from_dict(collisions_data)
# print(collisions_db)


# dob_items = ['issuance_date', 'expiration_date', 'job_start_date', 'gis_latitude', 'gis_longitude', 'job_type']
# dob_db = createFrameFromOpendata(client, dob_permit_issuance_rsrc, dob_items)

#cell 15
# convert information to more conveneient form for numerical comparison

record = collisions_db.iloc[0]
print(type(record['crash_date']))
print(type(record['crash_time']))
print(type(record['latitude']))
print(type(record['longitude']))


if type(collisions_db.iloc[0]['crash_date']) == str: 
    collisions_db['crash_date'] = collisions_db['crash_date'].apply(lambda t: t.split('T')[0]) #convert dates to timestamps for numerical comparison
    collisions_db['crash_date'] = collisions_db['crash_date'].apply(convertDates, axis=1, result_type='broadcast') #convert dates to timestamps for numerical comparison
    print(collisions_db['crash_date'])
if type(collisions_db.iloc[0]['crash_time']) == str:
    collisions_db['crash_time'] = collisions_db['crash_time'].apply(convertTimes, axis=1, result_type='broadcast') #convert dates to timestamps for numerical comparison
    print(collisions_db['crash_time'])

if type(collisions_db.iloc[0]['latitude']) == str:
    collisions_db['latitude'] = collisions_db['latitude'].apply(lambda x: float(x))  #force latitude to be a floating point for easier comparison
    print(collisions_db['latitude'])
if type(collisions_db.iloc[0]['longitude']) == str:
    collisions_db['longitude'] = collisions_db['longitude'].apply(lambda x: float(x)) #force longtitude to be a floating point for easier comparison
    print(collisions_db['longitude'])

collisions_db['crash_timestamp'] = collisions_db['crash_date'].add(collisions_db['crash_time'])

#cell 16
# Filter based on time
collisions_db['crash_timestamp'] = collisions_db['crash_date'].add(collisions_db['crash_time'])
print(collisions_db.shape)
collisions_db_filtered = collisions_db[collisions_db['crash_timestamp'] > DATASET_START_TIME]
print(collisions_db_filtered.shape)


#cell 17
#iterate over the stc_metadata (a pandas frame) to fill in the feature for DOB. Filter the (filtered) data frame by timestamp and location
for filename, stc in stc_metadata.iterrows():
    print(stc)
    #filter by location
    lat_db = collisions_db_filtered['latitude']
    lon_db = collisions_db_filtered['longitude']
    lat_ind = np.logical_and(lat_db >= (stc['latitude'] - LATITUDE_TOLERANCE*2), lat_db <= (stc['latitude'] + LATITUDE_TOLERANCE*2))
    lon_ind = np.logical_and(lon_db >= (stc['longitude'] - LONGITUDE_TOLERANCE*2), lon_db <= (stc['longitude'] + LONGITUDE_TOLERANCE*2))
    # location_ind = np.logical_and(np.asarray(lat_ind), np.asarray(lon_ind))
    location_ind = np.logical_and(lat_ind, lon_ind)
    valid_loc_db = collisions_db_filtered[location_ind]
    
    #filter by time
    timestamp = stc['unix_timestamp']
    crash_time = valid_loc_db['crash_timestamp']

    start_ind = np.asarray(timestamp >= crash_time)
    end_ind_1hr = np.asarray(timestamp <= crash_time.add(1*60*60))
    end_ind_3hr = np.asarray(timestamp <= crash_time.add(3*60*60))

    time_ind_1hr = np.logical_and(start_ind, end_ind_1hr)
    time_ind_3hr = np.logical_and(start_ind, end_ind_3hr)

    valid_db_1hr = valid_loc_db[time_ind_1hr]
    valid_db_3hr = valid_loc_db[time_ind_3hr]

    if valid_db_1hr.empty:
        stc_features.loc[filename,'collision_hour'] = 0
        # print('no construction here: %s' % filename)

    else: 
        # print(valid_db.shape[0])
        stc_features.loc[filename,'collision_hour'] = 1


    if valid_db_3hr.empty:
        stc_features.loc[filename,'collision_3hours'] = 0
        # print('no construction here: %s' % filename)

    else: 
        # print(valid_db.shape[0])
        stc_features.loc[filename,'collision_3hours'] = 1



#cell 18
print(stc_features.loc[:, 'collision_hour'].mean())
print(stc_features.loc[:, 'collision_3hours'].mean())

#cell 19
## Active Construction Feature
Find active construction projects near sensor locations; use this to help find instances of impacts, drills, saws, etc. that may be present in the audio due to nearby construction

Nada; this is only active, and contains no historical data. Lost cause.

#cell 20
# constr_data = client.get_all(active_construction_rsrc)
# constr_db = createFrameFromOpendata(client, active_construction_rsrc, [])
constr_db = pd.DataFrame.from_dict(client.get(active_construction_rsrc))
print(constr_db)

# dob_items = ['issuance_date', 'expiration_date', 'job_start_date', 'gis_latitude', 'gis_longitude', 'job_type']
# dob_db = createFrameFromOpendata(client, dob_permit_issuance_rsrc, dob_items)

#cell 21
## Subway Entrance Feature
Find nearby subway entrances to help classify instances that might be based on a high density of people, such as talking, shouting, etc.

#cell 22
# subway_db = createFrameFromOpendata(client, subway_entrance_rsrc, subway_items)
# subway_data = client.get(active_construction_rsrc)
# print(subway_data)
# subway_db = pd.DataFrame.from_dict(subway_data)

# Downloaded file containing info; map will not be served the same as others
subway_csv = pd.read_csv('./STC/NYC_Subway_Entrances.csv')
# print(subway_db)
# subway_db = subway_csv['the_geom']
subway_db = subway_csv
print(subway_db)
print(subway_db.items)
print(subway_db.shape)


#cell 23
import math
def getLocations(point):
    # @point: a string containing a GPS location in format "POINT (lat lon)"
    # print(point)

    if not type(point) == str:
        if math.isnan(point):
            return (0,0)

    location = point[point.find('(')+1 : point.find(')')]

    location_split = location.split(' ')
    lon = float(location_split[0])
    lat = float(location_split[1])
    return (lat, lon)

#cell 24
# subway_db['locations'] = subway_db.apply(getLocations)
locations = subway_db['the_geom'].apply(getLocations)

subway_lat = locations.apply(lambda x: x[0])
subway_lon = locations.apply(lambda x: x[1])

subway_db['latitude'] = subway_lat
subway_db['longitude'] = subway_lon


#cell 25
#iterate over the stc_metadata (a pandas frame) to fill in the feature for DOB. Filter the (filtered) data frame by timestamp and location
for filename, stc in stc_metadata.iterrows():
    # print(stc)
    #filter by location
    lat_db = subway_db['latitude']
    lon_db = subway_db['longitude']
    lat_ind = np.logical_and(lat_db >= (stc['latitude'] - LATITUDE_TOLERANCE*2), lat_db <= (stc['latitude'] + LATITUDE_TOLERANCE*2))
    lon_ind = np.logical_and(lon_db >= (stc['longitude'] - LONGITUDE_TOLERANCE*2), lon_db <= (stc['longitude'] + LONGITUDE_TOLERANCE*2))
    # location_ind = np.logical_and(np.asarray(lat_ind), np.asarray(lon_ind))
    location_ind = np.logical_and(lat_ind, lon_ind)
    valid_loc_db = subway_db[location_ind]
    
    
    if valid_loc_db.empty:
        stc_features.loc[filename,'nearby_subway'] = 0

    else: 
        # print(valid_db.shape[0])
        stc_features.loc[filename,'nearby_subway'] = 1


#cell 26
print(stc_features['nearby_subway'].mean())

#cell 27
### PLUTO Tax Lot Features
These features will percentages of the makeup of tax lots within the block that the sensor lies on. The elements are residential, commercial, manufacturing, and miscellaneous

#cell 28
pluto_csv = pd.read_csv('./STC/pluto.csv')
pluto_columns = ['bbl', 'latitude', 'longitude', 'borough', 'block', 'lot', 'zonedist1', 'zonedist2', 'zonedist3', 'zonedist4']

pluto_db = pluto_csv.filter(items=pluto_columns)
pluto_db['borough_int'] = pluto_db['bbl'].apply(lambda x: x // 1000000000) # isolate the borough number

#cell 29
def decodeLotType(lot_type):
    l = []
    # print('decode LoT Type')
    # print(lot_type)

    if type(lot_type) != str:
        return []

    if '/' in lot_type:
        parts = lot_type.split('/')
        l.extend(decodeLotType(parts[0]))
        l.extend(decodeLotType(parts[1]))

    else:
        l.append(lot_type[0])

    # print(l)
    return l

def computeLotValues(type_list):
    res = 0
    comm = 0
    manu = 0
    misc = 0

    # print('Compute lot values')

    for l in type_list:
        # print(l)
        # print(type(l))
        if type(l) != str:
            continue

        if l == 'R':
            res += 1
        elif l == 'C':
            comm += 1
        elif l ==  'M':
            manu += 1
        else: 
            misc += 1

    # print(res)
    # print(comm)
    # print(manu)
    # print(misc)

    total = len(type_list)
    # print(total)
    if total == 0:
        return np.asarray([0,0,0,0])
    res = res / total
    comm = comm / total
    manu = manu / total 
    misc = misc / total

    result = np.asarray([res, comm, manu, misc])
    print(result)
    return result

def createTaxLotFeatures(df, borough, block):
    #df: a pandas dataframe containing all necessaryinformation about tax lots in the city
    borough_ind = df['borough_int'] == borough
    block_ind = df['block'] == block
    ind = np.logical_and(borough_ind, block_ind)
    num_lots = np.sum(ind)

    lots = []

    df_bb = df.loc[ind,:]
    print(df_bb)

    for index, row in df_bb.iterrows():
        print(row)
        lots.extend(decodeLotType(row.loc['zonedist1']))
        lots.extend(decodeLotType(row.loc['zonedist2']))
        lots.extend(decodeLotType(row.loc['zonedist3']))
        lots.extend(decodeLotType(row.loc['zonedist4']))

    print(lots)
    return computeLotValues(lots)




#cell 30
for

#cell 31
#
# iterate over the stc_metadata (a pandas frame) to fill in the feature for DOB. Filter the (filtered) data frame by timestamp and location
stc_feature_columns = ['DOB', 'collision_hour', 'collision_3hours', 'nearby_subway', 'tax_lot_residential', 'tax_lot_commercial', 'tax_lot_manufacturing', 'tax_lot_misc']

for filename, stc in stc_metadata.iterrows():
    # print(stc)
    #filter by location
    lat_db = subway_db['latitude']
    lon_db = subway_db['longitude']
    lat_ind = np.logical_and(lat_db >= (stc['latitude'] - LATITUDE_TOLERANCE*2), lat_db <= (stc['latitude'] + LATITUDE_TOLERANCE*2))
    lon_ind = np.logical_and(lon_db >= (stc['longitude'] - LONGITUDE_TOLERANCE*2), lon_db <= (stc['longitude'] + LONGITUDE_TOLERANCE*2))
    # location_ind = np.logical_and(np.asarray(lat_ind), np.asarray(lon_ind))
    location_ind = np.logical_and(lat_ind, lon_ind)
    valid_loc_db = subway_db[location_ind]
    
    block = stc['block']
    borough = stc['borough']
    print(block)
    print(borough)

    tax_feat = createTaxLotFeatures(pluto_db, borough, block)
    
    stc_features.loc[filename,'tax_lot_residential'] = tax_feat[0]
    stc_features.loc[filename,'tax_lot_commercial'] = tax_feat[1]
    stc_features.loc[filename,'tax_lot_manufacturing'] = tax_feat[2]
    stc_features.loc[filename,'tax_lot_misc'] = tax_feat[3]


#cell 32


