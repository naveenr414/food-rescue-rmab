from rmab.utils import haversine
import datetime
from rmab.database import run_query, open_connection, close_connection
import rmab.secret as secret 
import random 

def get_volunteer_trip_info(volunteer_id,donation_location_id,recipient_location_id):
    """Create a dictionary with information about a volunteer from their ID
    
    Arguments:
        volunteer_id: Integer, ID
    
    Returns: Two dictionaries: volunteer_info and trip_info"""

    db_name = secret.database_name 
    username = secret.database_username 
    password = secret.database_password 
    ip_address = secret.ip_address
    port = secret.database_port
    connection_dict = open_connection(db_name,username,password,ip_address,port)
    cursor = connection_dict['cursor']

    query = "SELECT r.*, d.donor_location_id FROM RESCUES r JOIN donations d ON r.donation_id = d.ID WHERE r.PUBLISHED_AT <= CURRENT_DATE AND r.USER_ID = {}".format(volunteer_id)
    user_trips = run_query(cursor,query)

    num_trips = len(user_trips) 
    previous_trips = [i for i in user_trips if i['donor_location_id'] == donation_location_id and i['recipient_location_id'] == recipient_location_id]
    previous_trip = len(previous_trips)>0
    last_completed = None 
    if previous_trip:
        last_completed = max([i['published_at'] for i in previous_trips])
    trip_difficulty = random.randint(1,5)
    last_trip = max([i['published_at'] for i in user_trips])

    query = "SELECT u.*, a.latitude, a.longitude FROM USERS u JOIN ADDRESSES a ON u.address_id = a.ID WHERE u.ID = {};".format(volunteer_id)
    user_info = run_query(cursor,query)[0]
    name = user_info['first_name'] + ' '+user_info['last_name']
    volunteer_location = (float(user_info['latitude']),float(user_info['longitude']))

    return {
        'volunteer_name': name,
        'num_trips': num_trips, 
        'last_trip': last_trip, 
        'volunteer_location': volunteer_location
    }, {
        'trip_difficulty': trip_difficulty,
        'previous_trip': previous_trip,
        'last_completed': last_completed,
    }

def get_donor_info(donor_location_id):
    """Get a dictionary with information on a donor, given the location ID
    
    Arguments:
        donor_location_id: Integer, which location to focus on
    
    Returns: Dictionary, with keys such as donor name and neighborhood"""

    db_name = secret.database_name 
    username = secret.database_username 
    password = secret.database_password 
    ip_address = secret.ip_address
    port = secret.database_port
    connection_dict = open_connection(db_name,username,password,ip_address,port)
    cursor = connection_dict['cursor']
    query = "SELECT d.*, a.* FROM DONOR_LOCATIONS dl JOIN DONORS d ON dl.donor_id = d.ID JOIN ADDRESSES a ON dl.address_id = a.ID WHERE dl.ID = {};".format(donor_location_id)

    donor_info = run_query(cursor,query)[0]
    return {
        'donor_name': donor_info['name'],
        'donor_neighborhood': donor_info['neighborhood'],
        'donor_location': (float(donor_info['latitude']),float(donor_info['longitude']))
    }

def get_recipient_info(recipient_location_id):
    """Get a dictionary with information on a donor, given the location ID
    
    Arguments:
        donor_location_id: Integer, which location to focus on
    
    Returns: Dictionary, with keys such as donor name and neighborhood"""

    db_name = secret.database_name 
    username = secret.database_username 
    password = secret.database_password 
    ip_address = secret.ip_address
    port = secret.database_port
    connection_dict = open_connection(db_name,username,password,ip_address,port)
    cursor = connection_dict['cursor']
    query = "SELECT d.*, a.* FROM RECIPIENT_LOCATIONS dl JOIN RECIPIENTS d ON dl.recipient_id = d.ID JOIN ADDRESSES a ON dl.address_id = a.ID WHERE dl.ID = {};".format(recipient_location_id)

    recipient_info = run_query(cursor,query)[0]
    return {
        'recipient_name': recipient_info['name'],
        'recipient_neighborhood': recipient_info['neighborhood'],
        'recipient_location': (float(recipient_info['latitude']),float(recipient_info['longitude']))
    }


def get_notification_text(volunteer_info,donor_info,recipient_info,trip_info):
    """Craft a multiple notification system, based on the volunteer info
        donor info, and recipient info
        
    Arguments:
        volunteer_info: Dictionary with keys such as volunteer_trips, etc.
        donor_info: Dictionary with keys such as donor_name, etc.
        recipient_info: Dictionary with keys such as recipient_name, etc.
        trip_info: Dictionary, with keys such as trip dificulty 
    
    Returns: String, personalized message for the volunteer"""

    distance_donor = round(haversine(donor_info['donor_location'][0],
                    donor_info['donor_location'][1],
                    volunteer_info['volunteer_location'][0],
                    volunteer_info['volunteer_location'][1]),1)
    if volunteer_info['num_trips'] <= 4 and trip_info['trip_difficulty'] <= 2:
        return "A new trip is available that would be perfect for your {} trip. The trip has good reviews and is from {} in {} to {} in {}".format(num_to_string[volunteer_info['num_trips']+1],
            donor_info['donor_name'],
            donor_info['donor_neighborhood'],
            recipient_info['recipient_name'],
            recipient_info['recipient_neighborhood'])
    elif trip_info['previous_trip']:
        if datetime.datetime.today()-trip_info['last_completed'] < datetime.timedelta(weeks=1):
            return "A trip you completed in the past week is available again! From {} in {} to {} in {}".format(donor_info['donor_name'],
            donor_info['donor_neighborhood'],
            recipient_info['recipient_name'],
            recipient_info['recipient_neighborhood'])
        elif datetime.datetime.today()-trip_info['last_completed'] < datetime.timedelta(days=31):
            return "A trip you completed in the past month is available again! From {} in {} to {} in {}".format(donor_info['donor_name'],
            donor_info['donor_neighborhood'],
            recipient_info['recipient_name'],
            recipient_info['recipient_neighborhood'])
        else:
            return "A trip you completed in past is available again! From {} in {} to {} in {}".format(donor_info['donor_name'],
            donor_info['donor_neighborhood'],
            recipient_info['recipient_name'],
            recipient_info['recipient_neighborhood'])

    elif distance_donor < 2: 
        return "New trip that's only {} miles away! From {} in {} to {} in {}".format(distance_donor,donor_info['donor_name'],
            donor_info['donor_neighborhood'],
            recipient_info['recipient_name'],
            recipient_info['recipient_neighborhood'])
    else: 
        return "New trip available from {} in {} to {} in {}, {} miles away".format(
            donor_info['donor_name'],
            donor_info['donor_neighborhood'],
            recipient_info['recipient_name'],
            recipient_info['recipient_neighborhood'],
            distance_donor 
            )