# -*- coding: utf-8 -*-
# In[1]:
    
##junction_map = {

##"1278": { "Device 1/nA":"A1A2", "Device 2/nA":"B7B8", "Device 3/nA":"C1C2", "Device 4/nA":"D7D8", "Device 5/nA":"E1E2", "Device 6/nA":"F7F8", "Device 7/nA":"G1G2", "Device 8/nA":"H7H8", "Channel1":"A1A2", "Channel2":"B7B8", "Channel3":"C1C2", "Channel4":"D7D8", "Channel5":"E1E2", "Channel6":"F7F8", "Channel7":"G1G2", "Channel8":"H7H8" },

##"3456": { "Device 1/nA":"A3A4", "Device 2/nA":"B5B6", "Device 3/nA":"C3C4", "Device 4/nA":"D5D6", "Device 5/nA":"E3E4", "Device 6/nA":"F5F6", "Device 7/nA":"G3G4", "Device 8/nA":"H5H6", "Channel1":"A3A4", "Channel2":"B5B6", "Channel3":"C3C4", "Channel4":"D5D6", "Channel5":"E3E4", "Channel6":"F5F6", "Channel7":"G3G4", "Channel8":"H5H6" },

##"5634": { "Device 1/nA":"A5A6", "Device 2/nA":"B3B4", "Device 3/nA":"C5C6", "Device 4/nA":"D3D4", "Device 5/nA":"E5E6", "Device 6/nA":"F3F4", "Device 7/nA":"G5G6", "Device 8/nA":"H3H4", "Channel1": "A5A6", "Channel2":"B3B4", "Channel3":"C5C6", "Channel4":"D3D4", "Channel5":"E5E6", "Channel6":"F3F4", "Channel7":"G5G6", "Channel8":"H3H4" },

##"7812": { "Device 1/nA":"A7A8", "Device 2/nA":"B1B2", "Device 3/nA":"C7C8", "Device 4/nA":"D1D2", "Device 5/nA":"E7E8", "Device 6/nA":"F1F2", "Device 7/nA":"G7G8", "Device 8/nA":"H1H2", "Channel1":"A7A8", "Channel2":"B1B2", "Channel3":"C7C8", "Channel4":"D1D2", "Channel5":"E7E8", "Channel6":"F1F2", "Channel7":"G7G8", "Channel8":"H1H2" } }

if __name__ == "__main__":
    
    ## IMPORT LIBRARIES
    # import numpy as np
    # General libraries
    # import numpy as np
    import pandas as pd
    # import random
    
    # Feature extraction
    from tsfresh import extract_features, extract_relevant_features, select_features, feature_extraction
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction.feature_calculators import set_property
    from tsfresh.feature_extraction import feature_calculators,EfficientFCParameters


    dev_num = 13
    conditions = ['Control_100mV','CTPR16_100mV']
    
    for condition in conditions:    
        ##### Select data for modeling
        
        df = pd.read_csv(f'./dev_{dev_num}_{condition}_norm_window_10000pts_overlap_0pts_ts.csv')
        print(df.head())
        
        ##### Create training and testing samples
        df = df.reset_index()
        df = df.rename(columns={"index": "num"})
        
        ##### Create a unique num id based on the index
        
        df_wo_labels = df.drop(['label','id'],axis=1)
        label_df = df[['num','label','id']]
      
        
        df_wo_labels = pd.melt(df_wo_labels,id_vars="num", var_name="time")
        df_wo_labels.sort_values(["num", "time"]).reset_index(drop=True)
        print(df_wo_labels.info)
        
        print(f'dev {dev_num} {condition} data loaded, formatted and ready for feature extraction...')
        
        label_df = label_df.set_index("num")

        print(type(label_df))
        
        
        df_wo_labels_all_feat = extract_features(df_wo_labels, column_id="num", column_sort="time",
                                                 default_fc_parameters=EfficientFCParameters(), n_jobs = 8)
        
        df_feat_labels = pd.concat([df_wo_labels_all_feat, label_df], axis=1)
        print(df_feat_labels.info)
        df_feat_labels.to_csv(f'dev_{dev_num}_{condition}_all_extracted_features.csv',header=True)
