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

    df_nb1_na_1 = add_constants(df_nb1_na_1,
                                REFLECTION_MIN_STD = config['REFLECTION_MIN_STD_1'],
                               WAVEGROUP_WIDTH = config['WAVEGROUP_WIDTH_1'],
                               N_REFLECTIONS_FWG_FRAME_THRSH = config['MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH_1'],
                               REFLECTION_FWG_AMP_DIFF_THRSH = config['REFLECTION_FWG_AMP_DIFF_THRSH_1'],
                               REFLECTION_AMP_THRSH = config['REFLECTION_AMP_THRSH_1'],
                               PROBE = 'NB1',
                               ITERATION = 1)
    

    df_nb2_na_1 = add_constants(df_nb2_na_1,
                                REFLECTION_MIN_STD = config['REFLECTION_MIN_STD_1'],
                                 WAVEGROUP_WIDTH = config['WAVEGROUP_WIDTH_1'],
                                 N_REFLECTIONS_FWG_FRAME_THRSH = config['MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH_1'],
                                 REFLECTION_FWG_AMP_DIFF_THRSH = config['REFLECTION_FWG_AMP_DIFF_THRSH_1'],
                                 REFLECTION_AMP_THRSH = config['REFLECTION_AMP_THRSH_1'],
                                 PROBE = 'NB2',
                                 ITERATION = 1)
    
    df_nb1_na_2 = add_constants(df_nb1_na_2,
                                REFLECTION_MIN_STD = config['REFLECTION_MIN_STD_2'],
                                 WAVEGROUP_WIDTH = config['WAVEGROUP_WIDTH_2'],
                                 N_REFLECTIONS_FWG_FRAME_THRSH = config['MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH_2'],
                                 REFLECTION_FWG_AMP_DIFF_THRSH = config['REFLECTION_FWG_AMP_DIFF_THRSH_2'],
                                 REFLECTION_AMP_THRSH = config['REFLECTION_AMP_THRSH_2'],
                                 PROBE = 'NB1',
                                 ITERATION = 2)
    
    
    df_nb2_na_2 = add_constants(df_nb2_na_2,
                                REFLECTION_MIN_STD = config['REFLECTION_MIN_STD_2'],
                                 WAVEGROUP_WIDTH = config['WAVEGROUP_WIDTH_2'],
                                 N_REFLECTIONS_FWG_FRAME_THRSH = config['MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH_2'],
                                 REFLECTION_FWG_AMP_DIFF_THRSH = config['REFLECTION_FWG_AMP_DIFF_THRSH_2'],
                                 REFLECTION_AMP_THRSH = config['REFLECTION_AMP_THRSH_2'],
                                 PROBE = 'NB2',
                                 ITERATION = 2)
    
    df_nb1_nb2 = df_nb1.append([df_nb2, df_nb1_na_1, df_nb2_na_1, df_nb1_na_2, df_nb2_na_2], ignore_index=True)
    
    return df_nb1_nb2

def debris_post_processor(df_all_probe_iter, config):
    
    """This function creates df suitable for flagging model and helps in choosing the final predicted depth based on the rules

    Args:
        df_all_probe_iter(list): list of dataframes with results from multiple iterations of NB1, NB2 probes
        conf(dictionary): a dictionary containing constants required to run the code.
            
    Returns:
            df_out(dataframe): dataframe with final predicted depth selected based on the rules
    """
    
    # list of dataframes with results from multiple iterations of NB1, NB2 probes
    [df_nb1, df_nb2, df_nb1_na_1, df_nb2_na_1, df_nb1_na_2, df_nb2_na_2] = df_all_probe_iter
    
    # construct a single dataframe with all the results
    df_nb1_nb2 = data_preprocess(df_nb1, df_nb2, df_nb1_na_1, df_nb2_na_1, df_nb1_na_2, df_nb2_na_2, config)
    
    # null conditions
    n1 = df_nb1_nb2['pred_depth'].isnull()
    n2 = df_nb1_nb2['pred_depth_invert'].isnull()
    n3 = df_nb1_nb2['pred_depth_fwg_rmv'].isnull()
    
    # Iterations
    i0 = df_nb1_nb2['ITERATION'] == 0
    i1 = df_nb1_nb2['ITERATION'] == 1
    i2 = df_nb1_nb2['ITERATION'] == 2
    
    # Probes
    p1 = df_nb1_nb2['PROBE'] == 'NB1'
    p2 = df_nb1_nb2['PROBE'] == 'NB2'
    
    # Final output df
    df_out = pd.DataFrame([])
    
    ######################## Rules to select final predicted depth ####################################
    # For iteration = 0, probe = nb1, select all non null pred depth from main fwg
    df_i0_p1_nn1 = df_nb1_nb2[(i0) & (p1) & (~n1)]
    df_i0_p1_nn1['pred_depth_nb1_nb2'] = df_i0_p1_nn1['pred_depth']
    df_out = df_out.append([df_i0_p1_nn1], ignore_index=True)
    
    # For iteration = 0, probe = nb2, select all non null pred depth from main fwg
    df_i0_p2_nn1 = df_nb1_nb2[(i0) & (p2) & (~n1)]
    df_i0_p2_nn1['pred_depth_nb1_nb2'] = df_i0_p2_nn1['pred_depth']
    df_i0_p2_nn1 = df_i0_p2_nn1[~df_i0_p2_nn1['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i0_p2_nn1], ignore_index=True)
    
    # For iteration = 0, probe = nb1, select all non null pred depth invert, if the pred depth from main fwg is null
    df_i0_p1_n1_nn2 = df_nb1_nb2[(i0) & (p1) & (n1) & (~n2)]
    df_i0_p1_n1_nn2['pred_depth_nb1_nb2'] = df_i0_p1_n1_nn2['pred_depth_invert']
    df_i0_p1_n1_nn2 = df_i0_p1_n1_nn2[~df_i0_p1_n1_nn2['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i0_p1_n1_nn2], ignore_index=True)
    
    # For iteration = 0, probe = nb2, select all non null pred depth invert, if the pred depth from main fwg is null
    df_i0_p2_n1_nn2 = df_nb1_nb2[(i0) & (p2) & (n1) & (~n2)]
    df_i0_p2_n1_nn2['pred_depth_nb1_nb2'] = df_i0_p2_n1_nn2['pred_depth_invert']
    df_i0_p2_n1_nn2 = df_i0_p2_n1_nn2[~df_i0_p2_n1_nn2['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i0_p2_n1_nn2], ignore_index=True)
    
    # For iteration = 0, probe = nb1, select all non null pred depth by removing fwg, if both pred depth and pred depth invert is null
    df_i0_p1_n1_n2_nn3 = df_nb1_nb2[(i0) & (p1) & (n1) & (n2) & (~n3)]
    df_i0_p1_n1_n2_nn3['pred_depth_nb1_nb2'] = df_i0_p1_n1_n2_nn3['pred_depth_fwg_rmv']
    df_i0_p1_n1_n2_nn3 = df_i0_p1_n1_n2_nn3[~df_i0_p1_n1_n2_nn3['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i0_p1_n1_n2_nn3], ignore_index=True)
    
    # For iteration = 0, probe = nb1, select all non null pred depth by removing fwg, if both pred depth and pred depth invert is null
    df_i0_p2_n1_n2_nn3 = df_nb1_nb2[(i0) & (p2) & (n1) & (n2) & (~n3)]
    df_i0_p2_n1_n2_nn3['pred_depth_nb1_nb2'] = df_i0_p2_n1_n2_nn3['pred_depth_fwg_rmv']
    df_i0_p2_n1_n2_nn3 = df_i0_p2_n1_n2_nn3[~df_i0_p2_n1_n2_nn3['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i0_p2_n1_n2_nn3], ignore_index=True)
    
    df_i1_p1_nn1 = df_nb1_nb2[(i1) & (p1) & (~n1)]
    df_i1_p1_nn1['pred_depth_nb1_nb2'] = df_i1_p1_nn1['pred_depth']
    df_i1_p1_nn1 = df_i1_p1_nn1[~df_i1_p1_nn1['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i1_p1_nn1], ignore_index=True)
    
    # df_i1_p1_n1_nn2 = df_nb1_nb2[(i1) & (p1) & (n1) & (~n2)]
    # df_i1_p1_n1_nn2['pred_depth_nb1_nb2'] = df_i1_p1_n1_nn2['pred_depth_invert']
    # df_i1_p1_n1_nn2 = df_i1_p1_n1_nn2[~df_i1_p1_n1_nn2['Indication'].isin(df_out['Indication'])]
    # df_out = df_out.append([df_i1_p1_n1_nn2], ignore_index=True)
    
    df_i1_p2_nn1 = df_nb1_nb2[(i1) & (p2) & (~n1)]
    df_i1_p2_nn1['pred_depth_nb1_nb2'] = df_i1_p2_nn1['pred_depth']
    df_i1_p2_nn1 = df_i1_p2_nn1[~df_i1_p2_nn1['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i1_p2_nn1], ignore_index=True)
    
    df_i1_p2_n1_nn2 = df_nb1_nb2[(i1) & (p2) & (n1) & (~n2)]
    df_i1_p2_n1_nn2['pred_depth_nb1_nb2'] = df_i1_p2_n1_nn2['pred_depth_invert']
    df_i1_p2_n1_nn2 = df_i1_p2_n1_nn2[~df_i1_p2_n1_nn2['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i1_p2_n1_nn2], ignore_index=True)
    
    df_i1_p1_n1_n2_nn3 = df_nb1_nb2[(i1) & (p1) & (n1) & (n2) & (~n3)]
    df_i1_p1_n1_n2_nn3['pred_depth_nb1_nb2'] = df_i1_p1_n1_n2_nn3['pred_depth_fwg_rmv']
    df_i1_p1_n1_n2_nn3 = df_i1_p1_n1_n2_nn3[~df_i1_p1_n1_n2_nn3['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i1_p1_n1_n2_nn3], ignore_index=True)
    
    df_i1_p2_n1_n2_nn3 = df_nb1_nb2[(i1) & (p2) & (n1) & (n2) & (~n3)]
    df_i1_p2_n1_n2_nn3['pred_depth_nb1_nb2'] = df_i1_p2_n1_n2_nn3['pred_depth_fwg_rmv']
    df_i1_p2_n1_n2_nn3 = df_i1_p2_n1_n2_nn3[~df_i1_p2_n1_n2_nn3['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i1_p2_n1_n2_nn3], ignore_index=True)
    
    
    df_i2_p1_nn1 = df_nb1_nb2[(i2) & (p1) & (~n1)]
    df_i2_p1_nn1['pred_depth_nb1_nb2'] = df_i2_p1_nn1['pred_depth']
    df_i2_p1_nn1 = df_i2_p1_nn1[~df_i2_p1_nn1['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i2_p1_nn1], ignore_index=True)
    
    df_i2_p1_n1_nn2 = df_nb1_nb2[(i2) & (p1) & (n1) & (~n2)]
    df_i2_p1_n1_nn2['pred_depth_nb1_nb2'] = df_i2_p1_n1_nn2['pred_depth_invert']
    df_i2_p1_n1_nn2 = df_i2_p1_n1_nn2[~df_i2_p1_n1_nn2['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i2_p1_n1_nn2], ignore_index=True)
    
    df_i2_p2_nn1 = df_nb1_nb2[(i2) & (p2) & (~n1)]
    df_i2_p2_nn1['pred_depth_nb1_nb2'] = df_i2_p2_nn1['pred_depth']
    df_i2_p2_nn1 = df_i2_p2_nn1[~df_i2_p2_nn1['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i2_p2_nn1], ignore_index=True)
    
    df_i2_p2_n1_nn2 = df_nb1_nb2[(i2) & (p2) & (n1) & (~n2)]
    df_i2_p2_n1_nn2['pred_depth_nb1_nb2'] = df_i2_p2_n1_nn2['pred_depth_invert']
    df_i2_p2_n1_nn2 = df_i2_p2_n1_nn2[~df_i2_p2_n1_nn2['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i2_p2_n1_nn2], ignore_index=True)
    
    df_i2_p1_n1_n2_nn3 = df_nb1_nb2[(i2) & (p1) & (n1) & (n2) & (~n3)]
    df_i2_p1_n1_n2_nn3['pred_depth_nb1_nb2'] = df_i2_p1_n1_n2_nn3['pred_depth_fwg_rmv']
    df_i2_p1_n1_n2_nn3 = df_i2_p1_n1_n2_nn3[~df_i2_p1_n1_n2_nn3['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i2_p1_n1_n2_nn3], ignore_index=True)
    
    df_i2_p2_n1_n2_nn3 = df_nb1_nb2[(i2) & (p2) & (n1) & (n2) & (~n3)]
    df_i2_p2_n1_n2_nn3['pred_depth_nb1_nb2'] = df_i2_p2_n1_n2_nn3['pred_depth_fwg_rmv']
    df_i2_p2_n1_n2_nn3 = df_i2_p2_n1_n2_nn3[~df_i2_p2_n1_n2_nn3['Indication'].isin(df_out['Indication'])]
    df_out = df_out.append([df_i2_p2_n1_n2_nn3], ignore_index=True)
    
    # for dominant reflections, used depth from reflections
    dominant_ref_cases = ((df_out['pred_depth_fwg_rmv'] - df_out['pred_depth']) >= 0.25) & (df_out['ITERATION'] == 0) & (df_out['PROBE'] == 'NB1')
    df_out['pred_depth_nb1_nb2'][dominant_ref_cases] = df_out['pred_depth_fwg_rmv'][dominant_ref_cases] 
    df_out = pd.merge(left = df_out, right = df_nb1[['Indication', 'pred_depth_first_peak']], how = 'inner', on = ['Indication'], suffixes = ('', '_nb1'))
    
    # for multi reflections cases, use first peak
    first_peak_col = 'pred_depth_first_peak'
    multi_ref_cases = (~df_out[first_peak_col].isnull())
    df_out['pred_depth_nb1_nb2'][multi_ref_cases] = df_out[first_peak_col][multi_ref_cases]

    # fill na for all indications for which no depth is found in any iterations
    df_null_depth = df_nb1[~df_nb1['Indication'].isin(df_out['Indication'])]
    df_null_depth['pred_depth_nb1_nb2'] = np.nan
    df_out = df_out.append([df_null_depth], ignore_index=True)

    assert df_out.shape[0] == df_nb1.shape[0]
    
    return df_out

