import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")

def add_constants(
        df, REFLECTION_MIN_STD, WAVEGROUP_WIDTH, N_REFLECTIONS_FWG_FRAME_THRSH,
        REFLECTION_FWG_AMP_DIFF_THRSH, REFLECTION_AMP_THRSH, PROBE, ITERATION):
    
    """This function add provided constants to the df. These constant are required in the flagging model

    Args:
        df(dataframe): dataframe with every row representing single flaw 
        REFLECTION_MIN_STD(int): Minimum std require across reflection window 
        WAVEGROUP_WIDTH(int): Focus wave group(FWG) width
        N_REFLECTIONS_FWG_FRAME_THRSH(int): Minimum number of the consecutive reflections required to consider the reflections in a frame
        REFLECTION_FWG_AMP_DIFF_THRSH(int): Amp threshold for difference between surface and flaw feature amp 
        REFLECTION_AMP_THRSH(int): Minimum Amp require for a reflection to be considered
        PROBE(str): Probe used
        ITERATION(int): Number of iteration i.e. 0, 1 or 2
            
    Returns:
            df((dataframe)): after adding constants in the columns
    """
    
    df['PROBE'] = PROBE
    df['ITERATION'] = ITERATION
    df['REFLECTION_MIN_STD'] = REFLECTION_MIN_STD
    df['WAVEGROUP_WIDTH'] = WAVEGROUP_WIDTH
    df['N_REFLECTIONS_FWG_FRAME_THRSH'] = N_REFLECTIONS_FWG_FRAME_THRSH
    df['REFLECTION_FWG_AMP_DIFF_THRSH'] = REFLECTION_FWG_AMP_DIFF_THRSH
    df['REFLECTION_AMP_THRSH'] = REFLECTION_AMP_THRSH
    
    return df

def data_preprocess(df_nb1, df_nb2, df_nb1_na_1, df_nb2_na_1, df_nb1_na_2, df_nb2_na_2, config):
    
    """This function calculates depth for a single flaw instances in row (single row of df)

    Args:
        df_nb1(dataframe): dataframe with results from probe nb1 and iteration 0
        df_nb2(dataframe): dataframe with results from probe nb2 and iteration 0
        df_nb1_na_1(dataframe): dataframe with results from probe nb1 and iteration 1
        df_nb2_na_1(dataframe): dataframe with results from probe nb2 and iteration 1
        df_nb1_na_2(dataframe): dataframe with results from probe nb1 and iteration 2
        df_nb2_na_2(dataframe): dataframe with results from probe nb2 and iteration 2
        conf(dictionary): a dictionary containing constants required to run the code.
            
    Returns:
        df_nb1_nb2(dataframe): dataframe with all the probe and iteration results
    """
    
    df_nb1 = add_constants(df_nb1,
                           REFLECTION_MIN_STD = config['REFLECTION_MIN_STD_0'],
                           WAVEGROUP_WIDTH = config['WAVEGROUP_WIDTH_0'],
                           N_REFLECTIONS_FWG_FRAME_THRSH = config['MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH_0'],
                           REFLECTION_FWG_AMP_DIFF_THRSH = config['REFLECTION_FWG_AMP_DIFF_THRSH_0'],
                           REFLECTION_AMP_THRSH = config['REFLECTION_AMP_THRSH_0'],
                           PROBE = 'NB1',
                           ITERATION = 0)

    df_nb2 = add_constants(df_nb2,
                           REFLECTION_MIN_STD = config['REFLECTION_MIN_STD_0'],
                           WAVEGROUP_WIDTH = config['WAVEGROUP_WIDTH_0'],
                           N_REFLECTIONS_FWG_FRAME_THRSH = config['MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH_0'],
                           REFLECTION_FWG_AMP_DIFF_THRSH = config['REFLECTION_FWG_AMP_DIFF_THRSH_0'],
                           REFLECTION_AMP_THRSH = config['REFLECTION_AMP_THRSH_0'],
                           PROBE = 'NB2',
                           ITERATION = 0)

    
    df_nb1_nb2 = df_nb1.append([df_nb2], ignore_index=True)
    
    return df_nb1_nb2

def debris_post_processor(df_all_probe_iter, config):
    """Post process depth results for iteration 0 with NB1 and NB2 probes"""
    
    [df_nb1, df_nb2] = df_all_probe_iter
    df_nb1_nb2 = data_preprocess(df_nb1, df_nb2, None, None, None, None, config)
    
    # null conditions
    n1 = df_nb1_nb2['pred_depth'].isnull()
    n2 = df_nb1_nb2['pred_depth_invert'].isnull()
    n3 = df_nb1_nb2['pred_depth_fwg_rmv'].isnull()
    
    # Probes
    p1 = df_nb1_nb2['PROBE'] == 'NB1'
    p2 = df_nb1_nb2['PROBE'] == 'NB2'
    
    df_out = pd.DataFrame([])
    
    # NB1 rules
    df_p1_nn1 = df_nb1_nb2[(p1) & (~n1)]
    df_p1_nn1['pred_depth_nb1_nb2'] = df_p1_nn1['pred_depth']
    df_out = df_out.append([df_p1_nn1], ignore_index=True)
    
    # NB2 rules 
    df_p2_nn1 = df_nb1_nb2[(p2) & (~n1)]
    df_p2_nn1['pred_depth_nb1_nb2'] = df_p2_nn1['pred_depth']
    df_p2_nn1 = df_p2_nn1[~df_p2_nn1['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_p2_nn1], ignore_index=True)
    
    # Handle reflections and peak cases
    dominant_ref_cases = ((df_out['pred_depth_fwg_rmv'] - df_out['pred_depth']) >= 0.25) & (df_out['PROBE'] == 'NB1')
    df_out.loc[dominant_ref_cases, 'pred_depth_nb1_nb2'] = df_out.loc[dominant_ref_cases, 'pred_depth_fwg_rmv']
    
    df_out = pd.merge(left=df_out, right=df_nb1[['Indication', 'pred_depth_first_peak']], 
                     how='inner', on=['Indication'], suffixes=('', '_nb1'))
    
    multi_ref_cases = (~df_out['pred_depth_first_peak'].isnull())
    df_out.loc[multi_ref_cases, 'pred_depth_nb1_nb2'] = df_out.loc[multi_ref_cases, 'pred_depth_first_peak']

    # Handle null depths
    df_null_depth = df_nb1[~df_nb1['Indication'].isin(df_out['Indication'])]
    df_null_depth['pred_depth_nb1_nb2'] = np.nan
    df_out = df_out.append([df_null_depth], ignore_index=True)

    assert df_out.shape[0] == df_nb1.shape[0]
    
    return df_out

