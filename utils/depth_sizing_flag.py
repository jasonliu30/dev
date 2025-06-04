import os
import joblib


model_root = 'depth_sizing\\models'

def load_model(config):
    
    """This function loads model

    Args:
        config(dictionary): a dictionary containing constants required to run the code. can be found in sizing_config.yaml under 'config'
            
    Returns:
            ml_model: loaded model
    """
    
    # load model
    ml_model = joblib.load(os.path.join(model_root, config['FLAGGING_MODEL']))

    return ml_model

def load_encoder():
    
    """This function loads encoders

    Args:
            
    Returns:
            encoders(list): list of encoders
    """
    
    # load encoders
    le_ch = joblib.load(os.path.join(model_root, "debris_le_ch.pkl"))
    le_probe = joblib.load(os.path.join(model_root, "debris_le_probe.pkl"))
    le_outage = joblib.load(os.path.join(model_root, "debris_le_outage.pkl"))
    
    encoders = [le_ch, le_outage, le_probe]
    
    return encoders

def apply_encoders(X, encoders):
    
    """This function applies encoders to the respective features

    Args:
        X(dataframe): dataframe with the required features for the flagging model
        encoders: encoders to encode the features
            
    Returns:
        X(dataframe): dataframe with encoded features
    """
    
    # encoders
    [le_ch, le_outage, le_probe] = encoders
    
    # apply encoder
    X['Channel'] = le_ch.transform(X['Channel'])
    X['Outage Number'] = le_outage.transform(X['Outage Number'])
    X['PROBE'] = le_probe.transform(X['PROBE'])
    
    return X

def read_txt_list(config):
    
    """This function read text file and convert to list

    Args:
        config(dictionary): a dictionary containing constants required to run the code. can be found in sizing_config.yaml under 'config'
 
    Returns:
        cols(list): list of the features
    """
    
    with open(os.path.join(model_root, config['FLAGGING_COLUMNS'])) as f:
        cols = f.readlines()
    for i in range(len(cols)):
        cols[i]=cols[i].replace('\n','')
        
    return cols

def flag_cases_debris(df, config):
    
    """
    This function flags the potential high error Debris cases  

    Args:
        df(dataframe): dataframe with the required features for the flagging model
        config(dictionary): a dictionary containing constants required to run the code. can be found in sizing_config.yaml under 'config'
            
    Returns:
        predictions(array): array of bool, True for the flagged cases
    """
    
    # load models and encoders
    ml_model = load_model(config)
    encoders = load_encoder()
    
    # get the required features for flagging
    cols = read_txt_list(config)
    X = df[cols]
    
    # apply encoders
    X = apply_encoders(X, encoders)
    
    # Apply ml model
    predictions = ml_model.predict_proba(X)
    predictions = predictions[:,1] >= config['FLAGGING_THRESH']
    
    # Flag cases with null depth
    depth_not_found_cases = X['pred_depth_nb1_nb2'].isnull()
    
    if depth_not_found_cases.any():
        predictions[depth_not_found_cases] = True
    
    return predictions

def flag_cases(df, config):
    
    """This function flags the potential high error FBBPF cases  

    Args:
        df(dataframe): Dataframe containing the features required for the flagging model
        config(dictionary): a dictionary containing constants required to run the code. can be found in sizing_config.yaml under 'config'

    Returns:
        y_pred(array): array of bool, True for the flagged cases
    """
    
    # load model
    ml_model = load_model(config)
    
    # get the required features for flagging
    cols = read_txt_list(config)
    
    # Apply ml model
    y_all = ml_model.predict(df[cols].fillna(0))
    y_pred = y_all>config['FLAGGING_THRESH']
    
    return y_pred