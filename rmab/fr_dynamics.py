import numpy as np
from datetime import timedelta 
from rmab.database import run_query, open_connection, close_connection 
import rmab.secret as secret 

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