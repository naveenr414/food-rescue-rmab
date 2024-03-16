import numpy as np
from datetime import timedelta 
from rmab.database import run_query, open_connection, close_connection 
from rmab.utils import haversine
import rmab.secret as secret 
import random 
import json

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

    for i in range(1,population_size+1):
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
            count_matrix[i][j]/=(np.sum(count_matrix[i][j]))
    
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