import numpy as np
from datetime import timedelta 
import random 
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from rmab.database import run_query, open_connection, close_connection
from rmab.utils import haversine, binary_search_count
import rmab.secret as secret 
from rmab.utils import partition_volunteers


def get_food_rescue(all_population_size):
    """Get the transitions for Food Rescue
    
    Arguments:
        all_population_size: Integer, Number of total arms 
            This is larger than N; we select the N arms out of this population size
    
    Returns: Two Things
        Numpy array of size Nx2x2x2
        probs_by_partition: Probabilities for matching for each volunteer
            List of lists of size N"""

    probs_by_user = json.load(open("../../results/food_rescue/match_probs.json","r"))
    donation_id_to_latlon, recipient_location_to_latlon, rescues_by_user, all_rescue_data, user_id_to_latlon = get_db_data()
    probs_by_num = {}
    for i in rescues_by_user:
        if str(i) in probs_by_user and probs_by_user[str(i)] > 0 and len(rescues_by_user[i]) >= 3:
            if len(rescues_by_user[i]) not in probs_by_num:
                probs_by_num[len(rescues_by_user[i])] = []
            probs_by_num[len(rescues_by_user[i])].append(probs_by_user[str(i)])

    partitions = partition_volunteers(probs_by_num,all_population_size)
    probs_by_partition = []
    all_transitions = get_all_transitions_partition(all_population_size,partitions)

    for i in range(len(partitions)):
        temp_probs = []
        for j in partitions[i]:
            temp_probs += (probs_by_num[j])
        probs_by_partition.append(temp_probs)

    for i,partition in enumerate(partitions):
        current_transitions = np.array(all_transitions[i])
        partition_scale = np.array([len(probs_by_num[j]) for j in partition])
        partition_scale = partition_scale/np.sum(partition_scale)
        prod = current_transitions*partition_scale[:,np.newaxis,np.newaxis,np.newaxis]
        new_transition = np.sum(prod,axis=0)
        all_transitions[i] = new_transition
    all_transitions = np.array(all_transitions)

    return all_transitions, probs_by_partition

def get_food_rescue_top(all_population_size):
    """Get the transitions for Food Rescue
        For volunteers who completed more than 100 trips
    
    Arguments:
        all_population_size: Integer, Number of total arms 
            This is larger than N; we select the N arms out of this population size
    
    Returns: Two Things
        Numpy array of size Nx2x2x2
        probs_by_partition: Probabilities for matching for each volunteer
            List of lists of size N"""

    probs_by_user = json.load(open("../../results/food_rescue/match_probs.json","r"))
    donation_id_to_latlon, recipient_location_to_latlon, rescues_by_user, all_rescue_data, user_id_to_latlon = get_db_data()
    probs_by_num = {}
    for i in rescues_by_user:
        if str(i) in probs_by_user and probs_by_user[str(i)] > 0 and len(rescues_by_user[i]) >= 100:
            if len(rescues_by_user[i]) not in probs_by_num:
                probs_by_num[len(rescues_by_user[i])] = []
            probs_by_num[len(rescues_by_user[i])].append(probs_by_user[str(i)])

    partitions = partition_volunteers(probs_by_num,all_population_size)
    probs_by_partition = []

    for i in range(len(partitions)):
        temp_probs = []
        for j in partitions[i]:
            temp_probs += (probs_by_num[j])
        probs_by_partition.append(temp_probs)

    all_transitions = get_all_transitions_partition(all_population_size,partitions)

    for i,partition in enumerate(partitions):
        current_transitions = np.array(all_transitions[i])
        partition_scale = np.array([len(probs_by_num[j]) for j in partition])
        partition_scale = partition_scale/np.sum(partition_scale)
        prod = current_transitions*partition_scale[:,np.newaxis,np.newaxis,np.newaxis]
        new_transition = np.sum(prod,axis=0)
        all_transitions[i] = new_transition
    all_transitions = np.array(all_transitions)

    return all_transitions, probs_by_partition

def get_data_all_users(cursor):
    """Retrieve the list of rescue times by user, stored in a dictionary
    
    Arguments: 
        cursor: Cursor the Food Rescue PSQL database
        
    Returns: Dictionary, with keys as user ID, and contains a list of times"""

    query = (
        "SELECT USER_ID, PUBLISHED_AT "
        "FROM RESCUES "
        "WHERE PUBLISHED_AT <= CURRENT_DATE "
        "AND USER_ID IS NOT NULL "
    )

    all_user_published = run_query(cursor,query)

    data_by_user = {}
    for i in all_user_published:
        user_id = i['user_id']
        published_at = i['published_at']

        if user_id not in data_by_user:
            data_by_user[user_id] = []

        data_by_user[user_id].append(published_at)

    for i in data_by_user:
        data_by_user[i] = sorted(data_by_user[i])

    return data_by_user 

def get_all_transitions_partition(population_size,partition):
    """Get a numpy matrix with all the transition probabilities for each type of agent
    
    Arguments: 
        population_size: Number of agents (2...population_size) we're getting data for
    
    Returns: List of numpy matrices of size 2x2x2; look at get transitions for more info"""

    db_name = secret.database_name 
    username = secret.database_username 
    password = secret.database_password 
    ip_address = secret.ip_address
    port = secret.database_port

    connection_dict = open_connection(db_name,username,password,ip_address,port)
    connection = connection_dict['connection']
    cursor = connection_dict['cursor']

    data_by_user = get_data_all_users(cursor)

    close_connection(connection,cursor)

    transitions = []

    for p in partition:
        temp_transition = []
        for i in p:
            temp_transition.append(get_transitions(data_by_user,i))
        transitions.append(temp_transition)
    return transitions


def get_all_transitions(population_size):
    """Get a numpy matrix with all the transition probabilities for each type of agent
    
    Arguments: 
        population_size: Number of agents (2...population_size) we're getting data for
    
    Returns: List of numpy matrices of size 2x2x2; look at get transitions for more info"""

    db_name = secret.database_name 
    username = secret.database_username 
    password = secret.database_password 
    ip_address = secret.ip_address
    port = secret.database_port

    connection_dict = open_connection(db_name,username,password,ip_address,port)
    connection = connection_dict['connection']
    cursor = connection_dict['cursor']

    data_by_user = get_data_all_users(cursor)

    close_connection(connection,cursor)

    transitions = []

    for i in range(3,population_size+3):
        transitions.append(get_transitions(data_by_user,i))
    
    return np.array(transitions)

def get_transitions(data_by_user,num_rescues):
    """Get the transition probabilities for a given agent with a total of 
        num_rescues rescues
    
    Arguments:
        data_by_user: A dictionary mapping each user_id to a list of times they serviced
        num_rescues: How many resuces the agent should have 

    Returns: Matrix of size 2 (start state) x 2 (actions) x 2 (end state)
        For each (start state, action), the resulting end states sum to 1"""
    
    count_matrix = np.zeros((2,2,2))

    # Edge case; with 1-rescue volunteers, they always go to inactive
    if num_rescues == 1:
        for i in range(count_matrix.shape[0]):
            for j in range(count_matrix.shape[1]):
                count_matrix[i][j][0] = 1
        return count_matrix 

    for user_id in data_by_user:
        if len(data_by_user[user_id]) == num_rescues:
            start_rescue = data_by_user[user_id][0]
            end_rescue = data_by_user[user_id][-1]

            week_dates = [start_rescue]
            current_date = start_rescue 

            while current_date <= end_rescue:
                current_date += timedelta(weeks=1)
                week_dates.append(current_date) 
            
            has_event = [0 for i in range(len(week_dates))]

            current_week = 0
            for i, rescue in enumerate(data_by_user[user_id]):
                while rescue>week_dates[current_week]+timedelta(weeks=1):
                    current_week += 1 
                has_event[current_week] = 1
            
            for i in range(len(has_event)-2):
                start_state = has_event[i]
                action = has_event[i+1]
                end_state = has_event[i+2]
                count_matrix[start_state][action][end_state] += 1
    
    for i in range(len(count_matrix)):
        for j in range(len(count_matrix[i])):
            if np.sum(count_matrix[i][j]) != 0:
                count_matrix[i][j]/=(np.sum(count_matrix[i][j]))
            else:
                count_matrix[i][j] = np.array([0.5,0.5])
    
    return count_matrix 

def compute_days_till(data_by_user,num_rescues=-1):
    """Compute the number of days till the rescues, as the number of rescues increases
    
    Arguments:
        data_by_user: Dictionary with data on rescue times for each user 
        num_rescues: Optional, Integer; consider only volunteers with k rescues
        
    Returns: List of size num_rescues (or 100 if num_rescues=-1)"""

    differences_between = []

    max_rescues = num_rescues-1 
    if num_rescues == -1:
        max_rescues = 100

    for i in range(max_rescues):
        num_with = 0
        total_diff = 0

        for j in data_by_user:
            if len(data_by_user[j])>=i+2:
                if num_rescues == -1 or len(data_by_user[j]) == num_rescues:
                    num_with += 1

                    total_diff += (data_by_user[j][i+1]-data_by_user[j][i]).days  

        total_diff /= (num_with)
        differences_between.append(total_diff)

    return differences_between 



def get_match_probs(cohort_idx): 
    """Get real homogenous probabilities based on a cohort
    
    Arguments:
        cohort_idx: List of volunteers, based on the number of trips completed
        
    Returns: Match probabilities, list of floats between 0-1"""

    db_name = secret.database_name 
    username = secret.database_username 
    password = secret.database_password 
    ip_address = secret.ip_address
    port = secret.database_port
    connection_dict = open_connection(db_name,username,password,ip_address,port)
    connection = connection_dict['connection']
    cursor = connection_dict['cursor']

    query = "SELECT * FROM RESCUES"
    all_rescue_data = run_query(cursor,query)

    query = ("SELECT * FROM ADDRESSES")
    all_addresses = run_query(cursor,query)
    len(all_addresses)

    address_id_to_latlon = {}
    address_id_to_state = {}
    for i in all_addresses:
        address_id_to_state[i['id']] = i['state']
        address_id_to_latlon[i['id']] = (i['latitude'],i['longitude'])

    user_id_to_latlon = {}
    user_id_to_state = {}
    user_id_to_start = {}
    user_id_to_end = {}

    query = ("SELECT * FROM USERS")
    user_data = run_query(cursor,query)

    for user in user_data:
        if user['address_id'] != None: 
            user_id_to_latlon[user['id']] = address_id_to_latlon[user['address_id']]
            user_id_to_state[user['id']] = address_id_to_state[user['address_id']]
            user_id_to_start[user['id']] = user['created_at']
            user_id_to_end[user['id']] = user['updated_at']


    query = (
        "SELECT * "
        "FROM RESCUES "
        "WHERE PUBLISHED_AT <= CURRENT_DATE "
        "AND USER_ID IS NOT NULL "
    )

    all_user_published = run_query(cursor,query)

    query = (
        "SELECT * FROM donor_locations"
    )
    data = run_query(cursor,query)
    donor_location_to_latlon = {}
    for i in data:
        donor_location_to_latlon[i['id']] = address_id_to_latlon[i['address_id']]

    query = (
        "SELECT * FROM donations"
    )
    donation_data = run_query(cursor,query)

    donation_id_to_latlon = {}
    for i in donation_data:
        donation_id_to_latlon[i['id']] = donor_location_to_latlon[i['donor_location_id']]

    query = (
        "SELECT USER_ID, PUBLISHED_AT "
        "FROM RESCUES "
        "WHERE PUBLISHED_AT <= CURRENT_DATE "
        "AND USER_ID IS NOT NULL "
    )

    all_user_published = run_query(cursor,query)

    data_by_user = {}
    for i in all_user_published:
        user_id = i['user_id']
        published_at = i['published_at']

        if user_id not in data_by_user:
            data_by_user[user_id] = []

        data_by_user[user_id].append(published_at)

    num_rescues_to_user_id = {}
    for i in data_by_user:
        if len(data_by_user[i]) not in num_rescues_to_user_id:
            num_rescues_to_user_id[len(data_by_user[i])] = []
        num_rescues_to_user_id[len(data_by_user[i])].append(i)

    rescue_to_latlon = {}
    rescue_to_time = {}
    for i in all_rescue_data:
        if i['published_at'] != None and donation_id_to_latlon[i['donation_id']] != None and donation_id_to_latlon[i['donation_id']][0] != None:
            rescue_to_latlon[i['id']] = donation_id_to_latlon[i['donation_id']]
            rescue_to_latlon[i['id']] = (float(rescue_to_latlon[i['id']][0]),float(rescue_to_latlon[i['id']][1]))
            rescue_to_time[i['id']] = i['published_at']

    def num_notifications(user_id):
        user_location = user_id_to_latlon[user_id]
        if user_location[0] == None:
            return 0
        user_location = (float(user_location[0]),float(user_location[1]))
        user_start = user_id_to_start[user_id]
        user_end = user_id_to_end[user_id]

        relevant_rescues = [i for i in rescue_to_time if user_start <= rescue_to_time[i] and rescue_to_time[i] <= user_end]
        relevant_rescues = [i for i in relevant_rescues if haversine(user_location[0],user_location[1],rescue_to_latlon[i][0],rescue_to_latlon[i][1]) < 5]
        return len(relevant_rescues)

    temp_dict = json.load(open("../results/food_rescue/match_probs.json","r"))
    all_match_probs = {}
    for i in temp_dict:
        all_match_probs[int(i)] = temp_dict[i]

    for i in num_rescues_to_user_id:
        num_rescues_to_user_id[i] = [j for j in num_rescues_to_user_id[i] if j in all_match_probs]

    match_probs = []
    for i in cohort_idx:
        random_id = random.choice(num_rescues_to_user_id[i])
        while random_id not in all_match_probs:
            random_id = random.choice(num_rescues_to_user_id[i])
        match_probs.append(all_match_probs[random_id])
    return match_probs

def get_dict_match_probs(): 
    """Get real matching probabilities based on a cohort
    
    Arguments:
        cohort_idx: List of volunteers, based on the number of trips completed
        
    Returns: Match probabilities, list of floats between 0-1"""
    
    db_name = secret.database_name 
    username = secret.database_username 
    password = secret.database_password 
    ip_address = secret.ip_address
    port = secret.database_port
    connection_dict = open_connection(db_name,username,password,ip_address,port)
    cursor = connection_dict['cursor']

    query = "SELECT * FROM RESCUES"
    all_rescue_data = run_query(cursor,query)

    query = ("SELECT * FROM ADDRESSES")
    all_addresses = run_query(cursor,query)

    address_id_to_latlon = {}
    address_id_to_state = {}
    for i in all_addresses:
        address_id_to_state[i['id']] = i['state']
        address_id_to_latlon[i['id']] = (i['latitude'],i['longitude'])

    # Get user information
    user_id_to_latlon = {}
    user_id_to_state = {}
    user_id_to_start = {}
    user_id_to_end = {}

    query = ("SELECT * FROM USERS")
    user_data = run_query(cursor,query)
    for user in user_data:
        if user['address_id'] != None: 
            user_id_to_latlon[user['id']] = address_id_to_latlon[user['address_id']]
            user_id_to_state[user['id']] = address_id_to_state[user['address_id']]
            user_id_to_start[user['id']] = user['created_at']
            user_id_to_end[user['id']] = user['updated_at']

    query = (
        "SELECT * "
        "FROM RESCUES "
        "WHERE PUBLISHED_AT <= CURRENT_DATE "
        "AND USER_ID IS NOT NULL "
    )
    all_user_published = run_query(cursor,query)

    query = (
        "SELECT * FROM donor_locations"
    )
    data = run_query(cursor,query)
    donor_location_to_latlon = {}
    for i in data:
        donor_location_to_latlon[i['id']] = address_id_to_latlon[i['address_id']]

    query = (
        "SELECT * FROM donations"
    )
    donation_data = run_query(cursor,query)
    donation_id_to_latlon = {}
    for i in donation_data:
        donation_id_to_latlon[i['id']] = donor_location_to_latlon[i['donor_location_id']]

    query = (
        "SELECT USER_ID, PUBLISHED_AT "
        "FROM RESCUES "
        "WHERE PUBLISHED_AT <= CURRENT_DATE "
        "AND USER_ID IS NOT NULL "
    )
    all_user_published = run_query(cursor,query)
    data_by_user = {}
    for i in all_user_published:
        user_id = i['user_id']
        published_at = i['published_at']

        if user_id not in data_by_user:
            data_by_user[user_id] = []

        data_by_user[user_id].append(published_at)

    # Get rescue location info
    num_rescues_to_user_id = {}
    for i in data_by_user:
        if len(data_by_user[i]) not in num_rescues_to_user_id:
            num_rescues_to_user_id[len(data_by_user[i])] = []
        num_rescues_to_user_id[len(data_by_user[i])].append(i)
    rescue_to_latlon = {}
    rescue_to_time = {}
    for i in all_rescue_data:
        if i['published_at'] != None and donation_id_to_latlon[i['donation_id']] != None and donation_id_to_latlon[i['donation_id']][0] != None:
            rescue_to_latlon[i['id']] = donation_id_to_latlon[i['donation_id']]
            rescue_to_latlon[i['id']] = (float(rescue_to_latlon[i['id']][0]),float(rescue_to_latlon[i['id']][1]))
            rescue_to_time[i['id']] = i['published_at']

    def num_notifications(user_id):
        """Compute the number of times a user was notified
            Use the fact that all users within a 5 mile radius 
            are notified 
            
        Arguments: 
            user_id: Integer, the ID for the user
            
        Returns: Integer, number of notifications"""
        if user_id not in user_id_to_latlon:
            return 0
        user_location = user_id_to_latlon[user_id]
        if user_location[0] == None:
            return 0
        user_location = (float(user_location[0]),float(user_location[1]))
        user_start = user_id_to_start[user_id]
        user_end = user_id_to_end[user_id]

        relevant_rescues = [i for i in rescue_to_time if user_start <= rescue_to_time[i] and rescue_to_time[i] <= user_end]
        relevant_rescues = [i for i in relevant_rescues if haversine(user_location[0],user_location[1],rescue_to_latlon[i][0],rescue_to_latlon[i][1]) < 5]
        return len(relevant_rescues)

    id_to_match_prob = {}

    num_done = 0
    for i in num_rescues_to_user_id:
        num_done += 1
        if num_done % 10 == 0:
            print("Done with {} out of {}".format(num_done,len(num_rescues_to_user_id)))
        for _id in num_rescues_to_user_id[i]:
            notifications = num_notifications(_id)
            if notifications < 1000:
                id_to_match_prob[_id] = 0 
            else:
                id_to_match_prob[_id] = i/notifications 
    return id_to_match_prob

def get_db_data():
    """Get data from the Food Rescue database so we can predict match probs
    
    Arguments: None
    
    Returns: The following dictionaries
        donation_id_to_latlon - Maps to tuples
        recipient_location_to_latlon - Maps to tuples 
        rescues_by_user - Maps to sorted list of datetime 
        all_rescue_data - List"""

    db_name = secret.database_name 
    username = secret.database_username 
    password = secret.database_password 
    ip_address = secret.ip_address
    port = secret.database_port
    connection_dict = open_connection(db_name,username,password,ip_address,port)
    cursor = connection_dict['cursor']

    query = "SELECT * FROM RESCUES"
    all_rescue_data = run_query(cursor,query)

    rescues_by_user = {}
    for i in all_rescue_data:
        if i['user_id'] not in rescues_by_user:
            rescues_by_user[i['user_id']] = []
        rescues_by_user[i['user_id']].append(i['created_at'])
    for i in rescues_by_user:
        rescues_by_user[i] = sorted(rescues_by_user[i])

    query = ("SELECT * FROM ADDRESSES")
    all_addresses = run_query(cursor,query)
    address_id_to_latlon = {}
    address_id_to_state = {}
    for i in all_addresses:
        address_id_to_state[i['id']] = i['state']
        address_id_to_latlon[i['id']] = (i['latitude'],i['longitude'])

    query = ("SELECT * FROM USERS")
    user_data = run_query(cursor,query)
    user_id_to_latlon = {}
    user_id_to_state = {}
    user_id_to_start = {}
    user_id_to_end = {}
    for user in user_data:
        if user['address_id'] != None: 
            user_id_to_latlon[user['id']] = address_id_to_latlon[user['address_id']]
            user_id_to_state[user['id']] = address_id_to_state[user['address_id']]
            user_id_to_start[user['id']] = user['created_at']
            user_id_to_end[user['id']] = user['updated_at']
    query = (
        "SELECT * FROM donor_locations"
    )
    data = run_query(cursor,query)
    donor_location_to_latlon = {}
    for i in data:
        donor_location_to_latlon[i['id']] = address_id_to_latlon[i['address_id']]


    query = (
        "SELECT * FROM donations"
    )
    donation_data = run_query(cursor,query)
    donation_id_to_latlon = {}
    for i in donation_data:
        donation_id_to_latlon[i['id']] = donor_location_to_latlon[i['donor_location_id']]

    query = (
        "SELECT * FROM recipient_locations"
    )
    data = run_query(cursor,query)
    recipient_location_to_latlon = {}
    for i in data:
        recipient_location_to_latlon[i['id']] = address_id_to_latlon[i['address_id']]

    return donation_id_to_latlon, recipient_location_to_latlon, rescues_by_user, all_rescue_data, user_id_to_latlon

def get_data_predict_match(rescue,user_id, donation_id_to_latlon, recipient_location_to_latlon, user_id_to_latlon, rescues_by_user):
    """Get the features for a particular rescue-user combination
    
    Arguments:
        rescue: Dictionary of info for a particular rescue
        user_id: integer, which user we calculate prob for 
        donation_id_to_latlon: Dictionary mapping donation_id to lattitude, longitude
        recipient_id_to_latlon: Dictionary mapping recipient_id to lattitude, longitude
        user_id_to_latlon: Mapping user ids to lattitude and longitude
        rescues_by_user: List of rescues completed by each user 
    """

    if rescue['donation_id'] not in donation_id_to_latlon or rescue['recipient_location_id'] not in recipient_location_to_latlon:
        return None    

    donor_lat, donor_lon = donation_id_to_latlon[rescue['donation_id']]
    recipient_lat, recipient_lon = recipient_location_to_latlon[rescue['recipient_location_id']]

    if donor_lat == None or donor_lon == None or recipient_lat == None or recipient_lon == None:
        return None  
    
    donor_lat = float(donor_lat)
    donor_lon = float(donor_lat)
    recipient_lat = float(recipient_lat)
    recipient_lon = float(recipient_lon)

    state = rescue['state']

    if rescue['published_at'] == None:
        return None  

    year = rescue['published_at'].year
    month = rescue['published_at'].month 
    day = rescue['published_at'].day 
    hour = rescue['published_at'].hour
    distance = haversine(donor_lat,donor_lon,recipient_lat,recipient_lon)

    if user_id not in user_id_to_latlon or user_id_to_latlon[user_id][0] == None:
        return None 
    volunteer_lat, volunteer_lon = user_id_to_latlon[user_id]
    volunteer_lat = float(volunteer_lat)
    volunteer_lon = float(volunteer_lat)

    volunteer_dist_donor = haversine(volunteer_lat,volunteer_lon,donor_lat,donor_lon)
    volunteer_dist_recipient = haversine(volunteer_lat,volunteer_lon,recipient_lat,recipient_lon)
    if rescue['published_at'] == None:
        return None 

    num_rescues = binary_search_count(rescues_by_user[user_id],rescue['published_at'])

    data_x = [donor_lat,donor_lon,recipient_lat,recipient_lon,state,year,month,day,hour,distance,volunteer_lat,volunteer_lon,volunteer_dist_donor,volunteer_dist_recipient,num_rescues]
    return data_x 

def get_train_test_data(rescues_by_user,donation_id_to_latlon, recipient_location_to_latlon, user_id_to_latlon,all_rescue_data):
    """Get training, val, test data on matches between users, food rescue
    
    Arguments:
        rescues_by_user: List of rescues completed by each user 
        donation_id_to_latlon: Dictionary mapping donation_id to lattitude, longitude
        recipient_id_to_latlon: Dictionary mapping recipient_id to lattitude, longitude
        user_id_to_latlon: Mapping user ids to lattitude and longitude
        rescue_data: Metadata for each rescue, list
    """

    positive_X = []
    positive_Y = []
    negative_X = []
    negative_Y = []
    all_users = [i for i in rescues_by_user if len(rescues_by_user)>0]
    num_negatives = 1
    dataset_size = 10000
    odd_selection = dataset_size/300000

    for rescue in all_rescue_data:
        user_id = rescue['user_id']
        data_x = get_data_predict_match(rescue,user_id, donation_id_to_latlon, recipient_location_to_latlon, user_id_to_latlon, rescues_by_user)
        data_y = 1

        if data_x != None and random.random() < odd_selection:
            positive_X.append(data_x)
            positive_Y.append(data_y)

        for i in range(num_negatives):
            user_id = random.sample(all_users,1)[0]
            data_x = get_data_predict_match(rescue,user_id, donation_id_to_latlon, recipient_location_to_latlon, user_id_to_latlon, rescues_by_user)

            if data_x != None and random.random() < odd_selection:
                negative_X.append(data_x)
                negative_Y.append(0)
    all_X = positive_X + negative_X 
    all_Y = positive_Y + negative_Y 
    all_data = list(zip(all_X,all_Y))
    random.shuffle(all_data)
    train_data = all_data[:int(len(all_data)*0.8)]
    valid_data = all_data[int(len(all_data)*0.8):int(len(all_data)*0.9)]
    test_data = all_data[int(len(all_data)*0.9):]

    train_X = [i[0] for i in train_data]
    train_Y = [i[1] for i in train_data]

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    valid_X = [i[0] for i in valid_data]
    valid_Y = [i[1] for i in valid_data]

    valid_X = np.array(valid_X)
    valid_Y = np.array(valid_Y)

    test_X = [i[0] for i in test_data]
    test_Y = [i[1] for i in test_data]

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y

def train_rf():
    """Train a Random Forest Classifier to predict match probabilities
    
    Arguments: None
    
    Returns: A SkLearn Random Forest Classifier, and a dictionary
        with accuracy, precision, and recall scores"""

    donation_id_to_latlon, recipient_location_to_latlon, rescues_by_user, all_rescue_data, user_id_to_latlon = get_db_data() 
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = get_train_test_data(rescues_by_user,donation_id_to_latlon, recipient_location_to_latlon, user_id_to_latlon,all_rescue_data)
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train_X, train_Y)
    predictions = rf_classifier.predict(test_X)
    accuracy = accuracy_score(test_Y, predictions)
    precision = precision_score(test_Y, predictions)
    recall = recall_score(test_Y, predictions)
    return rf_classifier, {'accuracy': accuracy, 'precision': precision, 'recall': recall}

def get_match_probs(rescue,user_ids,rf_classifier,donation_id_to_latlon, recipient_location_to_latlon, user_id_to_latlon, rescues_by_user):
    """Get the match probability for a specific rescue-user_id combo
    
    Arguments:
        rescue: Dictionary with information on a particular rescue 
        user_id: Integer, a particular user_id
        rf_classifier: Random Forest, SkLearn model 
        donation_id_to_latlon: Dictionary mapping donation_id to lattitude, longitude
        recipient_id_to_latlon: Dictionary mapping recipient_id to lattitude, longitude
        user_id_to_latlon: Mapping user ids to lattitude and longitude
        rescues_by_user: List of rescues completed by each user 
    """

    
    data_points = []
    is_none = []
    ret = []

    for user_id in user_ids: 
        data_point = get_data_predict_match(rescue,user_id,donation_id_to_latlon, recipient_location_to_latlon, user_id_to_latlon, rescues_by_user)
        if data_point == None: 
            is_none.append(True)
            ret.append(0)
            data_points.append([0 for i in range(15)])
        else:
            is_none.append(False)
            ret.append(1)
            data_points.append(data_point)
    data_points = np.array(data_points)
    is_none= np.array(is_none)
    ret = np.array(ret)
    if len(data_points[is_none == False]) > 0:
        rf_probabilities = rf_classifier.predict_proba(data_points[is_none == False])[:,1]
        ret[is_none == False] = rf_probabilities
    return ret, data_points

def get_match_probabilities(T,volunteers_per_group,groups,rf_classifier,rescues_by_user,all_rescue_data,donation_id_to_latlon, recipient_location_to_latlon, user_id_to_latlon):
    """Get match probabilities for T different random rescues for a set of volunteers
    
    Arguments:
        T: Integer, number of rescues
        volunteers_per_group: Integer, how many volunteers should be of each type
        groups: List of integers, each signifying one group of volunteers to simulate
        user_id: Integer, a particular user_id
        rf_classifier: Random Forest, SkLearn model 
        donation_id_to_latlon: Dictionary mapping donation_id to lattitude, longitude
        recipient_id_to_latlon: Dictionary mapping recipient_id to lattitude, longitude
        user_id_to_latlon: Mapping user ids to lattitude and longitude
        rescues_by_user: List of rescues completed by each user 

    Returns: match_probabilities: T x N array and features: T x N x M, where M is # of features
    """

    volunteer_ids = []
    match_probabilities = []
    features = []

    for g in groups:
        all_users = [i for i in rescues_by_user if len(rescues_by_user[i]) == g]
        volunteer_ids += random.choices(all_users,k=volunteers_per_group)

    rescues = random.sample(all_rescue_data,T)
    for i in range(T):
        match_probs, current_feats = get_match_probs(rescues[i],
                        volunteer_ids,rf_classifier,donation_id_to_latlon, 
                        recipient_location_to_latlon, user_id_to_latlon, rescues_by_user)

        match_probabilities.append(match_probs)
        features.append(current_feats)

    return np.array(match_probabilities), np.array(features) 