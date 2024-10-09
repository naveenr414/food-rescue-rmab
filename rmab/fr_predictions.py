from rmab.fr_dynamics import *
import random 
from rmab.database import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


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

    if os.path.exists("../../results/food_rescue/rf_classifier.pkl"):
        rf = pickle.load(open("../../results/food_rescue/rf_classifier.pkl","rb"))
        return rf, {}

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
    ret = np.array(ret,dtype=float)
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
