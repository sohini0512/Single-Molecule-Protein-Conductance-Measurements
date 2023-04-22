#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
import pandas as pd
from scipy import signal
import os
from natsort import natsorted
import seaborn as sns
import math
from matplotlib import gridspec
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
# from sklearn.metrics import r2_score

print("Package Versions:")
import sklearn; print("  scikit-learn:", sklearn.__version__)
import scipy; print("  scipy:", scipy.__version__)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
sns.set(style='whitegrid', palette=colors, rc={'axes.labelsize': 16})

def func(x, a, b):
    return a*np.exp(b*x)

##checking if the max and min current values of UP sweep are close to the max and min values of the DOWN sweep

def sweepstakes(current):  
    curr_per_index = list(current)
    current_1 = curr_per_index[: int(len(curr_per_index)/2)]  ## up sweep , take last point 200 mV
    current_2 = curr_per_index[int(len(curr_per_index)/2):]   ## down sweep, take last point 200 mV
    if abs(current_1[-1] - current_2[0]) <= 1 and abs(current_1[0] - current_2[-1]) <= 1:   
        if current_1[-1] >= current_2[0] :
            return current_1[-1]
        else:
            return current_2[0]         ##### return mean([mean(current_1[-100:]),mean(current_2[:100])])
    else:
        return math.nan

# In[2]:


ctpr_num = 8 ## enter ctpr length
run_num = 2 ## enter run number

# In[3]:

raw_df = pd.read_csv(f'I_vs_T_raw_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',index_col=0) ##enter I-t file name
time_index=[]
duration = 0.00001 ## seconds , 0.8 s per file, 100 kHz sampling frequency
j = 0
for i in range(len(raw_df)):
    time_index.append(j)
    j+=duration
#print(time_index)
raw_df['TIME INDEX'] = time_index
print('NUMBER OF SWEEPS RECORDED :', len(set(raw_df['FILE INDEX'])))

# In[4]:

max_curr_df = raw_df.groupby(['FILE INDEX'])['CURRENT'].apply(sweepstakes).to_frame()
max_curr_df.reset_index(inplace=True)
#print(max_curr_df.head)
max_curr_df.dropna(axis=0,inplace=True)
selected_file_index = set(list(max_curr_df['FILE INDEX'].values))
selected_raw_df = raw_df[raw_df['FILE INDEX'].isin(selected_file_index)]

# In[5]:

selected_curr = []
fin_file_index = []
time_stamp = []
for i in selected_file_index :
    window_df = selected_raw_df.loc[selected_raw_df['FILE INDEX'] == i]
    y = window_df['CURRENT'].values
    x = window_df['FILE INDEX'].values
    t = window_df['TIME INDEX'].values
    if y[39999] > y[40000]:
        selected_curr.append(y[39999])
        fin_file_index.append(x[39999])
        time_stamp.append(t[39999])
    else:
        selected_curr.append(y[40000])
        fin_file_index.append(x[40000])  ##taking max between last point of up sweep vs first point of down sweep
        time_stamp.append(t[40000])
        
max_curr_selected_df = pd.DataFrame({'CURRENT': selected_curr,'TIME INDEX': time_stamp,'FILE INDEX': fin_file_index})
print(max_curr_selected_df.info)

# In[6]:

expt_max_curr = max_curr_selected_df.copy().dropna()
expt_max_curr.sort_values(by="TIME INDEX",ascending=True,inplace=True) #,ignore_index=True)
expt_max_curr['CURRENT INV'] = np.multiply(expt_max_curr['CURRENT'].values,-1)
print('NUMBER OF SELECTED SWEEPS AFTER CHECKING PROXIMITY BETWEEN UP AND DOWN SWEEPS:', len(expt_max_curr))

# In[7]:

plt.figure(figsize=(16,8))
sns.scatterplot(x='TIME INDEX',y='CURRENT',data=expt_max_curr,color='black',alpha=1,label = 'EXPERIMENT')
plt.ylabel('Current',fontsize=14)
plt.xlabel('Timestamp',fontsize=14)
plt.legend(fontsize=14)
plt.show()
plt.close()

max_curr_positive = expt_max_curr.loc[expt_max_curr['CURRENT']>0,['CURRENT INV','CURRENT','FILE INDEX','TIME INDEX']] #'TIME',
max_curr_positive['Log CURRENT'] = np.log(max_curr_positive['CURRENT'])
max_curr_positive.sort_values(by=['TIME INDEX'],inplace=True)

print(max_curr_positive.info)

num_less_0_1_nA = max_curr_positive[max_curr_positive['CURRENT'] <= 0.1].count()
num_less_0_2_nA = max_curr_positive[max_curr_positive['CURRENT'] <= 0.2].count()

print('number of points less than 0.1 nA :',num_less_0_1_nA)
print('number of points less than 0.2 nA :',num_less_0_2_nA)

plt.figure(figsize=(16,8))
sns.scatterplot(x='TIME INDEX',y='Log CURRENT',data=max_curr_positive,color='black',alpha=1,label = 'EXPERIMENT')
plt.ylabel('Log Current',fontsize=14)
plt.xlabel('TIME INDEX',fontsize=14)
plt.legend(fontsize=14)
plt.show()
plt.close()

# In[8]:
##WINDWOWING
expt_ln_curr_data = max_curr_positive['Log CURRENT'].values
expt_curr_data = max_curr_positive['CURRENT'].values
expt_inv_data = max_curr_positive['CURRENT INV'].values
expt_time_data = max_curr_positive['TIME INDEX'].values
expt_file_index = max_curr_positive['FILE INDEX'].values
peaks,_= signal.find_peaks(expt_ln_curr_data,prominence=0.6, height=-1, threshold=-3) ##play with the paramaters for correct windowing of the I-t trace

indices = list(peaks)
print(indices)
indices.sort()

indices_lst_fin = [0] + indices + [len(expt_time_data)-1]
indices_lst_fin = list(set(indices_lst_fin))
indices_lst_fin.sort()


plt.figure(figsize=(16,8))
plt.scatter(expt_time_data,expt_ln_curr_data,lw=2)
plt.plot(expt_time_data[indices_lst_fin], expt_ln_curr_data[indices_lst_fin], "rx",ms=14)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Log Current', fontsize=14)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(16,8))
plt.scatter(expt_time_data,expt_curr_data,lw=2)
plt.plot(expt_time_data[indices_lst_fin], expt_curr_data[indices_lst_fin], "rx",ms=14)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Current', fontsize=14)
plt.tight_layout()
plt.savefig(f'./STM_automated_filtering/Peaks_and_windows/Peaks_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
plt.show()
plt.close()

#print(indices_lst_fin)

##windows
cluster = np.zeros(len(expt_time_data))

# find minimium required rows given we want 4 columns
ncols = 8
nrows = (len(indices_lst_fin)-1) // ncols + ((len(indices_lst_fin)-1) % ncols > 0)

plt.figure(figsize=(ncols*2, nrows*2))
plt.subplots_adjust(hspace=0.5)

for j in range(len(indices_lst_fin)-1):
    
    curr_slice = expt_curr_data[indices_lst_fin[j]+1:indices_lst_fin[j+1]+1]
    num_slice = expt_time_data[indices_lst_fin[j]+1:indices_lst_fin[j+1]+1]
    
    cluster[indices_lst_fin[j]+1:indices_lst_fin[j+1]+1] = j
    
    ax = plt.subplot(nrows,ncols,j+1)
    # ax.set_ylabel('Current (nA)')
    # ax.set_xlabel('Time (s)')
    ax.scatter(num_slice,curr_slice,color='orange',label = f'Window {j}')
    ax.legend(loc='upper center',fontsize=6)
    plt.savefig(f'./STM_automated_filtering/Peaks_and_windows/Windows_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
    plt.show()
    plt.close()

plt.figure(figsize=(ncols*2, nrows*2))
plt.subplots_adjust(hspace=0.5)
    
for j in range(len(indices_lst_fin)-1):
    
    curr_slice = expt_ln_curr_data[indices_lst_fin[j]+1:indices_lst_fin[j+1]+1]
    num_slice = expt_time_data[indices_lst_fin[j]+1:indices_lst_fin[j+1]+1]
    
    cluster[indices_lst_fin[j]+1:indices_lst_fin[j+1]+1] = j
    
    ax = plt.subplot(nrows,ncols,j+1)
    # ax.set_ylabel('Log Current (log (nA))')
    # ax.set_xlabel('Time (s)')    
    ax.scatter(num_slice,curr_slice,label = f'Window {j}')
    ax.legend(loc='upper center',fontsize=6)
    plt.savefig(f'./STM_automated_filtering/Peaks_and_windows/Windows_log_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
    plt.show()
    plt.close()   

print(len(cluster),len(expt_time_data),cluster)

max_curr_positive['Clusters'] = cluster

# In[9]:

## COARSE FILTERING:FIRST FILTERING STEP...FITTING THE LAST 5 DATAPOINTS IN EACH WINDOW AND CLASSIFIED INTO 'PURE DRIFT' and 'JUNCTION CONTAMINATED WITH DRIFT' FOLDER BASED ON R2 VALUE

dist=[]
mae=[]
#file_index=[]
drift_indices_list =  [] 
junction_indices_list = []
slope_coarse=[]
intercept_coarse=[]
drift_X = []
drift_Y = []
junction_X = []
junction_Y = []
slope_junction = []
intercept_junction = []
window_junction = []
window_drift = []

for w in range(0,int(max(cluster))+1):
    window_df = max_curr_positive.loc[max_curr_positive['Clusters'] == w]
   
    y = window_df['CURRENT'].values
    Y = window_df['Log CURRENT'].values
    X = window_df['TIME INDEX'].values
 
    X_1 = []
    Y_1 = []
    y_1 = []
 
    for i in range(len(X)):
         if X[-1]- X[i] <=5:  ## last 5 datapoints used for fitting        
            X_1.append(X[i])
            Y_1.append(Y[i])
            y_1.append(y[i])
   
    if len(X)>= 5:  
        slope, intercept, r_value, p_value, std_err = linregress(X_1,Y_1) ##last 5 datapoints used for fitting
        distance = Y - (intercept + slope*np.asarray(X))                  
        Y_pred = intercept + slope*np.asarray(X)
        mse_f = np.mean(distance**2)
        mae_f = np.mean(abs(distance)) 
         
        if 0.99 <= r_value <= 1 and slope > 0 and intercept < 0 : ##use r_value such that atleast 2 windows are selected as drift to give significant points for the drift histogram

            print(f'DRIFT - window {w}:',r_value) #,intercept, slope)

            drift_X.append(X)
            drift_Y.append(Y)
            window_drift.append(w)
            dist.extend(distance)
            mae.append(mae_f)
            drift_indices_list.extend(window_df['FILE INDEX'].values)
            slope_coarse.append(slope)
            intercept_coarse.append(intercept)
            
        else:
            junction_indices_list.extend(window_df['FILE INDEX'].values)
            junction_X.append(X)
            junction_Y.append(Y)
            slope_junction.append(slope)
            intercept_junction.append(intercept)
            window_junction.append(w)

            print(f'JUNCTION - window {w}:',r_value) #,intercept, slope)
            
# find minimium required rows given we want 4 columns
ncols = 8
nrows = len(drift_X) // ncols + (len(drift_X) % ncols > 0)

plt.figure(figsize=(ncols*3, nrows*2))
plt.subplots_adjust(hspace=0.5)
# plt.suptitle("Windows", fontsize=16) # y=0.95)

for j in range(len(drift_X)):
    
    curr_slice = drift_Y[j]
    num_slice = drift_X[j]
    slope = slope_coarse[j]
    intercept = intercept_coarse[j]
    window = window_drift[j]
    
    ax = plt.subplot(nrows,ncols,j+1)
    ax = plt.plot(num_slice, curr_slice, 'o',label = f'Window {window}')
    ax = plt.plot(num_slice, intercept + slope*np.asarray(num_slice), 'r')
    # ax = plt.ylabel('Log(Current)',fontsize=14)
    # ax = plt.xlabel('Timestamp',fontsize=14)
    ax = plt.legend(loc='upper center',fontsize=6)
    plt.savefig(f'./STM_automated_filtering/Drift_coarse/Drift_Windows_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
    plt.show()
    plt.close()

# find minimium required rows given we want 4 columns
ncols = 8
nrows = len(junction_X) // ncols + (len(junction_X) % ncols > 0)

plt.figure(figsize=(ncols*3, nrows*2))
plt.subplots_adjust(hspace=0.5)
# plt.suptitle("Windows", fontsize=16) # y=0.95)

for j in range(len(junction_X)):
    
    curr_slice = junction_Y[j]
    num_slice = junction_X[j]
    slope = slope_junction[j]
    intercept = intercept_junction[j]
    window = window_junction[j]
    
    ax = plt.subplot(nrows,ncols,j+1)
    ax = plt.plot(num_slice, curr_slice, 'o',label = f'Window {window}')
    ax = plt.plot(num_slice, intercept + slope*np.asarray(num_slice), 'r')
    # ax = plt.ylabel('Log(Current)',fontsize=14)
    # ax = plt.xlabel('Timestamp',fontsize=14)
    ax = plt.legend(loc='upper center',fontsize=6)
    plt.savefig(f'./STM_automated_filtering/Junction_coarse/Junction_Windows_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
    plt.show()
    plt.close()

# In[10]:

##PLOTTING SLOPE AND INTERCEPT HISTOGRAMS FOR LINE FITS WITH AFTER COARSE FILTERING TO ESTIMATE DRIFT VELOCITY

plt.figure(figsize=(6,6))
sns.histplot(slope_coarse,kde=True,kde_kws={'bw_method' : 0.15},color='green',alpha=0.2)
plt.ylabel('Counts')
plt.xlabel('Slope')
plt.title('Slope',fontsize=16)
plt.savefig(f'./STM_automated_filtering/Slope_drift_coarse/Slope_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
plt.show()
plt.close()
print('SLOPE COARSE',np.mean(slope_coarse))

plt.figure(figsize=(6,6))
sns.histplot(intercept_coarse,kde=True,kde_kws={'bw_method' : 0.15},color='blue',alpha=0.2)
plt.ylabel('Counts')
plt.xlabel('Intercept')
plt.title('Intercept',fontsize=16)
plt.show()
plt.close()
print('INTERCEPT COARSE',np.mean(intercept_coarse))

slope_coarse.sort()
intercept_coarse.sort()
slope_int_df = pd.DataFrame({'Slope':slope_coarse,'intercept': intercept_coarse})
slope_int_df.to_csv(f'./STM_automated_filtering/Slope_drift_coarse/Slope_intecept_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True) ##saved for plotting in OriginLab

# In[11]:

##COMBINED UP AND DOWN SWEEP HISTOGRAM FOR 'PURE DRIFT' FOLDER

folder_names = [f'./CTPR_{ctpr_num}_RUN_{run_num}/'] 

for folder_name in folder_names:
    name_fold = folder_name.split('/')[1]
    print(name_fold)
    location = folder_name 
    dataFolders = natsorted(os.listdir(location  + 'Raw')) 
 
    for folder in dataFolders[:1]:  ## 1: FOR H2O AND :1 FOR D2O
        dataFiles = natsorted(os.listdir(location  + 'Raw' + '/' + folder))   
        print(folder)
        print(len(dataFiles))
        g_up    = []  
        g_down  = []
        g_whole = []
        
        print('NUMBER OF FILES IN PURE DRIFT FOLDER:',len(drift_indices_list))
        
        for i in drift_indices_list:    
            tdms_file = TdmsFile(location  + 'Raw' + '/' + folder +'/' + dataFiles[i])  
            ## bias
            bias = tdms_file['Untitled']['Bias']
            bias_data = bias.data
            bias_data = np.ndarray.tolist(bias_data)
            bias_up = bias_data[: int(len(bias_data)/2)]
            bias_down = bias_data[int(len(bias_data)/2):]
            
            ## current
            current = tdms_file['Untitled']['Current']
            current_data = current.data
            current_data = np.ndarray.tolist(current_data)
            current_up = current_data[: int(len(current_data)/2)]
            current_down = current_data[int(len(current_data)/2):]
            
            slope_up, intercept_up, r_value_up, p_value_up, std_err_up = linregress(bias_up, current_up)
            slope_down, intercept_down, r_value_down, p_value_down, std_err_down = linregress(bias_down, current_down)
            
            if slope_up > 0 and slope_down > 0 :
                g_up.append(slope_up)
                g_down.append(slope_down)
                plt.plot(bias_data,current_data)
                plt.xlabel('Bias')
                plt.ylabel('Current')
                
        log_g_up = np.log10(g_up)
        log_g_down = np.log10(g_down)  
        log_g_whole = np.concatenate((log_g_up,log_g_down))
    
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(1,1) 
        binwidth=0.05
        
        log_g_up.sort()
        log_g_down.sort()
        df = pd.DataFrame({'Log G Up':log_g_up,'Log G Down': log_g_down})
        df.to_csv(f'./STM_automated_filtering/Drift_coarse/Log_G_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True) ##saved for plotting in OriginLab
        
        # the histogram of the data
        ax0 = plt.subplot(gs[0])
        ax0=sns.histplot(log_g_whole,binwidth=binwidth,kde=True,line_kws={'lw':3},
                         kde_kws= {'bw_method' : 0.3},color='blue',edgecolor='black',alpha=0.5) 
        ax0.lines[0].set_color('crimson')
        kdeline = ax0.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        peaks_idx,_ = signal.find_peaks(ys,prominence=0.5)
        print('Peak positions',*xs[peaks_idx],sep = "\n")
        #ax0.plot(xs[peaks_idx], ys[peaks_idx], "x", markersize=12,markerfacecolor='k',markeredgecolor='k',markeredgewidth=3)
        ax0.set_ylabel('Counts')
        ax0.set_xlabel('logG')
        ax0.set_xlim(-2.2,2.2)
        ax0.set_title('DRIFT:Histogram after coarse filtering',fontsize=16)
        plt.savefig(f'./STM_automated_filtering/Drift_coarse/Histogram_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
        plt.show()
        plt.close()
        
d_df = selected_raw_df[selected_raw_df['FILE INDEX'].isin(drift_indices_list)]
plt.figure(figsize=(14,4))
plt.scatter(d_df['TIME INDEX'],d_df['CURRENT'],s=2)
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.savefig(f'./STM_automated_filtering/Drift_coarse/I-T_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
plt.show()
plt.close()

# In[12]:

##COMBINED UP AND DOWN SWEEP HISTOGRAM FOR 'JUNCTION CONTAMINATED WITH DRIFT' FOLDER

folder_names = [f'./CTPR_{ctpr_num}_RUN_{run_num}/'] 

for folder_name in folder_names:
    name_fold = folder_name.split('/')[1]
    print(name_fold)
    location = folder_name 
    dataFolders = natsorted(os.listdir(location  + 'Raw')) 
 
    for folder in dataFolders[:1]:   ## 1: FOR H2O AND :1 FOR D2O
        dataFiles = natsorted(os.listdir(location  + 'Raw' + '/' + folder))   
        print(folder)
        print(len(dataFiles))
        g_up    = []  
        g_down  = []
        g_whole = []
        
        print('NUMBER OF FILES IN JUNCTION CONTAMINATED WITH DRIFT FOLDER:',len(junction_indices_list))
        
        for i in junction_indices_list:  
            tdms_file = TdmsFile(location  + 'Raw' + '/' + folder +'/' + dataFiles[i])  
            
            ## bias
            bias = tdms_file['Untitled']['Bias']
            bias_data = bias.data
            bias_data = np.ndarray.tolist(bias_data)
            bias_up = bias_data[: int(len(bias_data)/2)]
            bias_down = bias_data[int(len(bias_data)/2):]
            
            ## current
            current = tdms_file['Untitled']['Current']
            current_data = current.data
            current_data = np.ndarray.tolist(current_data)
            current_up = current_data[: int(len(current_data)/2)]
            current_down = current_data[int(len(current_data)/2):]
            
            slope_up, intercept_up, r_value_up, p_value_up, std_err_up = linregress(bias_up, current_up)
            slope_down, intercept_down, r_value_down, p_value_down, std_err_down = linregress(bias_down, current_down)
            
            if slope_up > 0 and slope_down > 0 :
                g_up.append(slope_up)
                g_down.append(slope_down)
                plt.plot(bias_data,current_data)
                plt.xlabel('Bias')
                plt.ylabel('Current')

        log_g_up = np.log10(g_up)
        log_g_down = np.log10(g_down)  
        log_g_whole = np.concatenate((log_g_up,log_g_down))
    
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(1,1) 
        binwidth=0.05
        
        log_g_up.sort()
        log_g_down.sort()
        df = pd.DataFrame({'Log G Up':log_g_up,'Log G Down': log_g_down})
        df.to_csv(f'./STM_automated_filtering/Junction_coarse/Log_G_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True) ##saved for plotting in OriginLab
        
        # the histogram of the data
        ax0 = plt.subplot(gs[0]) 
        ax0=sns.histplot(log_g_whole,binwidth=binwidth,kde=True,line_kws={'lw':3},
                 kde_kws= {'bw_method' : 0.3},color='blue',edgecolor='black',alpha=0.5) 
        ax0.lines[0].set_color('crimson')
        kdeline = ax0.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        peaks_idx,_ = signal.find_peaks(ys,prominence=0.5)
        print('Peak positions',*xs[peaks_idx],sep = "\n")
        #ax0.plot(xs[peaks_idx], ys[peaks_idx], "x", markersize=12,markerfacecolor='k',markeredgecolor='k',markeredgewidth=3)
        ax0.set_ylabel('Counts')
        ax0.set_xlabel('logG')
        ax0.set_xlim(-2.2,2.2)
        ax0.set_title('JUNCTION:Histogram after coarse filtering',fontsize=16)
        plt.savefig(f'./STM_automated_filtering/Junction_coarse/Histogram_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
        plt.show()
        plt.close()
    
j_df = selected_raw_df[selected_raw_df['FILE INDEX'].isin(junction_indices_list)]
plt.figure(figsize=(14,4))
plt.scatter(j_df['TIME INDEX'],j_df['CURRENT'],s=2)
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.savefig(f'./STM_automated_filtering/Junction_coarse/I-T_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
plt.show()
plt.close()

# In[13]:

## FINE FILTERING:SECOND FILTERING STEP...CURRENT FOR THE FIRST 5 DATAPOINTS CONSIDERED FOR FITTING AND ALL FITS ARE CONSIDERED FOR ABSOLUTE ERROR/DISTANCE CALCULATION 
## A THRESHOLD OR ERROR IS DECIDED AND ABOVE THE THRESHOLD IS CONSIDERED AS 'PURE JUNCTION' 
## BELOW THE ERROR THRESHOLD IS TAKEN INTO 'DRIFT IN JUNCTION' FOLDER

dist=[]
mae=[]
fin_file_index=[]
slope_all=[]
intercept_all = []
fine_X = []
fine_Y = []
window_fine = []
junction_df = max_curr_positive[max_curr_positive['FILE INDEX'].isin(junction_indices_list)]

for w in range(0,int(max(cluster))+1):
    window_df = junction_df.loc[junction_df['Clusters'] == w]
    
    y = window_df['CURRENT'].values
    Y = window_df['Log CURRENT'].values
    X = window_df['TIME INDEX'].values
    
    X_1 = []
    Y_1 = []
    y_1 = []

    for i in range(len(X)):
        if y[i] < 0.1: ## first 5 datapoints used for fitting, the first 5 points can fall below 0.1 or 0.2 nA amplitude , check that and give as the if condition,tail of the exponential
            X_1.append(X[i])
            Y_1.append(Y[i])
            y_1.append(y[i])

    if len(X_1)> 0:  
        slope, intercept, r_value, p_value, std_err = linregress(X_1,Y_1)  ##first 5 datapoints used for fitting
        distance = Y - (intercept + slope*np.asarray(X))                   
        
        mse_f = np.mean(distance**2)
        mae_f = np.mean(abs(distance))    
        
        if 0 < r_value < 1 :
            
            fine_X.append(X)
            fine_Y.append(Y)
            window_fine.append(w)
            dist.extend(distance)
            mae.append(mae_f)
            slope_all.append(slope)
            intercept_all.append(intercept)
            fin_file_index.extend(window_df['FILE INDEX'].values)

# find minimium required rows given we want 4 columns
ncols = 8
nrows = len(fine_X) // ncols + (len(fine_X) % ncols > 0)

plt.figure(figsize=(ncols*3, nrows*2))
plt.subplots_adjust(hspace=0.5)
# plt.suptitle("Windows", fontsize=16) # y=0.95)

for j in range(len(fine_X)):
    
    curr_slice = fine_Y[j]
    num_slice = fine_X[j]
    slope = slope_all[j]
    intercept = intercept_all[j]
    window = window_fine[j]
    
    ax = plt.subplot(nrows,ncols,j+1)
    ax = plt.plot(num_slice, curr_slice, 'o',label = f'Window {window}')
    ax = plt.plot(num_slice, intercept + slope*np.asarray(num_slice), 'r')
    # ax = plt.ylabel('Log(Current)',fontsize=14)
    # ax = plt.xlabel('Timestamp',fontsize=14)
    ax = plt.legend(loc='upper center',fontsize=6)
    plt.savefig(f'./STM_automated_filtering/Drift_fine/MAE_filter_Windows_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
    plt.show()
    plt.close()
	
# In[14]:

##PLOTTING SLOPE AND INTERCEPT HISTOGRAMS FOR LINE FITS IN FINE FILTERING
plt.figure(figsize=(6,6))
sns.histplot(slope_all,binwidth=0.05,kde=True,kde_kws={'bw_method' : 0.15},color='green',alpha=0.2)
plt.ylabel('Counts')
plt.xlabel('Slope')
plt.title('Slope',fontsize=16)
plt.show()
plt.close()
print('SLOPE FINE', np.mean(slope_all))

plt.figure(figsize=(6,6))
sns.histplot(intercept_all,kde=True,kde_kws={'bw_method' : 0.15},color='blue',alpha=0.2)
plt.ylabel('Counts')
plt.xlabel('Intercept')
plt.title('Intercept',fontsize=16)
plt.show()
plt.close()
print('INTERCEPT FINE', np.mean(intercept_all))

# In[15]:

####ABSOLUTE ERROR PLOTS AND THRESHOLDING FOR THE FINAL STEP OF FINE FILTERING
##
print(len(fin_file_index))
print(len(dist))

thresh = np.mean(abs(np.asarray(dist)))  ##MAE 
print('MAE THRESHOLD',thresh)

just_j_indices_list = []   ##PURE JUNCTION
for i in range(len(dist)):
    if abs(dist[i])>thresh:    ##THRESHOLD VALUE CAN BE MEAN OF ALL ERRORS 
        just_j_indices_list.append(fin_file_index[i])

print(len(just_j_indices_list))

d_in_j_indices_list =  []   ##DRIFT IN JUNCTION
for i in range(len(dist)):
    if abs(dist[i])<=thresh:
        d_in_j_indices_list.append(fin_file_index[i])

print(len(d_in_j_indices_list))


plt.figure(figsize=(16,8))
plt.scatter(fin_file_index,abs(np.asarray(dist)))
plt.hlines(thresh,0,max(fin_file_index)+1,color='red',linestyle= '--',linewidth=2 )
plt.xlabel('File index',fontsize=16)
plt.ylabel('Absolute Error',fontsize=16)
plt.savefig(f'./STM_automated_filtering/Drift_fine/MAE_threshold_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
plt.show()
plt.close()

# In[16]:

##COMBINED UP AND DOWN SWEEP HISTOGRAM FOR 'PURE JUNCTION' FOLDER

folder_names = [f'./CTPR_{ctpr_num}_RUN_{run_num}/']

for folder_name in folder_names:
    name_fold = folder_name.split('/')[1]
    print(name_fold)
    location = folder_name 
    dataFolders = natsorted(os.listdir(location  + 'Raw')) 
 
    for folder in dataFolders[:1]:    ## 1: FOR H2O AND :1 FOR D2O
        dataFiles = natsorted(os.listdir(location  + 'Raw' + '/' + folder))   
        print(folder)
        print(len(dataFiles))
        g_up    = []  
        g_down  = []
        g_whole = []

        print('Number of files that have junction data after fine filtering:',len(just_j_indices_list))
        
        for i in just_j_indices_list:    
            tdms_file = TdmsFile(location  + 'Raw' + '/' + folder +'/' + dataFiles[i])  
            
            ## bias
            bias = tdms_file['Untitled']['Bias']
            bias_data = bias.data
            bias_data = np.ndarray.tolist(bias_data)
            bias_up = bias_data[: int(len(bias_data)/2)]
            bias_down = bias_data[int(len(bias_data)/2):]
            
            ## current
            current = tdms_file['Untitled']['Current']
            current_data = current.data
            current_data = np.ndarray.tolist(current_data)
            current_up = current_data[: int(len(current_data)/2)]
            current_down = current_data[int(len(current_data)/2):]
            
            slope_up, intercept_up, r_value_up, p_value_up, std_err_up = linregress(bias_up, current_up)
            slope_down, intercept_down, r_value_down, p_value_down, std_err_down = linregress(bias_down, current_down)
            
            if slope_up > 0 and slope_down > 0 :
                g_up.append(slope_up)
                g_down.append(slope_down)
                plt.plot(bias_data,current_data)
                plt.xlabel('Bias')
                plt.ylabel('Current')

        log_g_up = np.log10(g_up)
        log_g_down = np.log10(g_down)  
        log_g_whole = np.concatenate((log_g_up,log_g_down))
    
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(1,1) 
        binwidth=0.05  #0.05
        
        log_g_up.sort()
        log_g_down.sort()
        df = pd.DataFrame({'Log G Up':log_g_up,'Log G Down': log_g_down})
        df.to_csv(f'./STM_automated_filtering/Junction_fine/Log_G_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True) ##saved for plotting in OriginLab
        
        # the histogram of the data
        ax0 = plt.subplot(gs[0])
        ax0=sns.histplot(log_g_whole,binwidth=binwidth,kde=True,line_kws={'lw':3},
                         kde_kws= {'bw_method' : 0.3},color='blue',edgecolor='black',alpha=0.5) 
        ax0.lines[0].set_color('crimson')
        kdeline = ax0.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        peaks_idx,_ = signal.find_peaks(ys,prominence=0.5)
        print('Peak positions',*xs[peaks_idx],sep = "\n")
        #ax0.plot(xs[peaks_idx], ys[peaks_idx], "x", markersize=12,markerfacecolor='k',markeredgecolor='k',markeredgewidth=3)
        ax0.set_ylabel('Counts')
        ax0.set_xlabel('logG')
        ax0.set_xlim(-2.2,2.2)
        ax0.set_title('JUNCTION:Histogram after fine filtering',fontsize=16)
        plt.savefig(f'./STM_automated_filtering/Junction_fine/Histogram_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
        plt.show()
        plt.close()
        
        X = log_g_whole.reshape(-1, 1)
        print('Please enter the number of components for gmm ')
        comp_junc = int(input())
        models = GaussianMixture(comp_junc,random_state=101).fit(X)
        M_best = models
        x = np.linspace(min(log_g_whole), max(log_g_whole), len(log_g_whole))
        logprob = M_best.score_samples(x.reshape(-1, 1))
        responsibilities = M_best.predict_proba(x.reshape(-1, 1))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        means_gmm = list(M_best.means_.flatten())
        means_gmm.sort()
        
        pred = models.predict(X)
        #print('gmm:silhouette:',silhouette_score(X,pred))
        
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #binwidth= 0.05
        #ax.hist(X, bins=np.arange(min(d2o_log_g_comb), max(d2o_log_g_comb) + binwidth, binwidth), density=True, histtype='stepfilled', alpha=0.4)
        ax.plot(x, pdf, '-k')
        ax.plot(x, pdf_individual, '--k')
        #for j in range(len(means_gmm)):
        #    ax.text(means_gmm[j], np.max(pdf)+0.1, "{:.2f}".format(means_gmm[j]),fontsize=12)
        #ax.set_ylim(0,np.max(pdf)+0.2)   
        ax.set_xlim(-2,2)
        ax.set_xlabel('Log G (log nS)',fontsize=14)
        ax.set_ylabel('$p(x)$',fontsize=14)
        ax.set_title('Density Estimation using GMM',fontsize=14)
        plt.savefig(f'./STM_automated_filtering/Junction_fine/GMM_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
        
        print('GMM PEAK POSITIONS')
        print('log G (log nS)',M_best.means_)
        print('ln G (ln nS)',np.log(10**(M_best.means_)))   
        print('G (nS)',10**M_best.means_)

        plt.show()
        plt.close()


final_j_df = selected_raw_df[selected_raw_df['FILE INDEX'].isin(just_j_indices_list)]
#final_j_df.to_csv(f'./Junction_fine_I_t_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True)
#final_j_200mV_df = max_curr_selected_df[max_curr_selected_df['FILE INDEX'].isin(just_j_indices_list)]   
#final_j_200mV_df.to_csv(f'./Junction_200mV_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True)
plt.figure(figsize=(14,4))
plt.scatter(final_j_df['TIME INDEX'],final_j_df['CURRENT'],s=2)
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.savefig(f'./STM_automated_filtering/Junction_fine/I-T_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
plt.show()
plt.close()


# In[17]:


##COMBINED UP AND DOWN SWEEP HISTOGRAM FOR 'DRIFT IN JUNCTION' FOLDER

folder_names = [f'./CTPR_{ctpr_num}_RUN_{run_num}/'] 

for folder_name in folder_names:
    name_fold = folder_name.split('/')[1]
    print(name_fold)
    location = folder_name 
    dataFolders = natsorted(os.listdir(location  + 'Raw')) 
 
    for folder in dataFolders[:1]:     ## 1: FOR H2O AND :1 FOR D2O
        dataFiles = natsorted(os.listdir(location  + 'Raw' + '/' + folder))   
        print(folder)
        print(len(dataFiles))
        g_up    = []  
        g_down  = []
        g_whole = []
    
        
        print('Number of files that have drift data after fine filtering:',len(d_in_j_indices_list))
        
        for i in d_in_j_indices_list:    
            tdms_file = TdmsFile(location  + 'Raw' + '/' + folder +'/' + dataFiles[i])  
            
            ## bias
            bias = tdms_file['Untitled']['Bias']
            bias_data = bias.data
            bias_data = np.ndarray.tolist(bias_data)
            bias_up = bias_data[: int(len(bias_data)/2)]
            bias_down = bias_data[int(len(bias_data)/2):]

            ## current
            current = tdms_file['Untitled']['Current']
            current_data = current.data
            current_data = np.ndarray.tolist(current_data)
            current_up = current_data[: int(len(current_data)/2)]
            current_down = current_data[int(len(current_data)/2):]
            
            slope_up, intercept_up, r_value_up, p_value_up, std_err_up = linregress(bias_up, current_up)
            slope_down, intercept_down, r_value_down, p_value_down, std_err_down = linregress(bias_down, current_down)
            
            if slope_up > 0 and slope_down > 0 :
                g_up.append(slope_up)
                g_down.append(slope_down)
                plt.plot(bias_data,current_data)
                plt.xlabel('Bias')
                plt.ylabel('Current')

        log_g_up = np.log10(g_up)
        log_g_down = np.log10(g_down)  
        log_g_whole = np.concatenate((log_g_up,log_g_down))
    
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(1,1) 
        binwidth=0.05  #0.05
        
        log_g_up.sort()
        log_g_down.sort()
        df = pd.DataFrame({'Log G Up':log_g_up,'Log G Down': log_g_down})
        df.to_csv(f'./STM_automated_filtering/Drift_fine/Log_G_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True)##saved for plotting in OriginLab
        # the histogram of the data
        ax0 = plt.subplot(gs[0])
        ax0=sns.histplot(log_g_whole,binwidth=binwidth,kde=True,line_kws={'lw':3},
                 kde_kws= {'bw_method' : 0.3},color='blue',edgecolor='black',alpha=0.5) 
        ax0.lines[0].set_color('crimson')
        kdeline = ax0.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        peaks_idx,_ = signal.find_peaks(ys,prominence=0.5)
        print('Peak positions',*xs[peaks_idx],sep = "\n")
        #ax0.plot(xs[peaks_idx], ys[peaks_idx], "x", markersize=12,markerfacecolor='k',markeredgecolor='k',markeredgewidth=3)
        ax0.set_ylabel('Counts')
        ax0.set_xlabel('logG')
        ax0.set_xlim(-2.2,2.2)
        ax0.set_title('DRIFT:Histogram after fine filtering',fontsize=16)
        plt.savefig(f'./STM_automated_filtering/Drift_fine/Histogram_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
        plt.show()
        plt.close()

        X = log_g_whole.reshape(-1, 1)
        print('Please enter the number of components for gmm ')
        drift_comp = int(input())
        models = GaussianMixture(drift_comp,random_state=101).fit(X)
        M_best = models
        x = np.linspace(min(log_g_whole), max(log_g_whole), len(log_g_whole))
        logprob = M_best.score_samples(x.reshape(-1, 1))
        responsibilities = M_best.predict_proba(x.reshape(-1, 1))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        means_gmm = list(M_best.means_.flatten())
        means_gmm.sort()
        
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #binwidth= 0.05
        #ax.hist(X, bins=np.arange(min(d2o_log_g_comb), max(d2o_log_g_comb) + binwidth, binwidth), density=True, histtype='stepfilled', alpha=0.4)
        ax.plot(x, pdf, '-k')
        ax.plot(x, pdf_individual, '--k')
        #for j in range(len(means_gmm)):
        #    ax.text(means_gmm[j], np.max(pdf)+0.1, "{:.2f}".format(means_gmm[j]),fontsize=12)
        #ax.set_ylim(0,np.max(pdf)+0.2)
        ax.set_xlim(-2,2)
        ax.set_xlabel('Log G (log nS)',fontsize=14)
        ax.set_ylabel('$p(x)$',fontsize=14)
        ax.set_title('Density Estimation using GMM',fontsize=14)
        plt.savefig(f'./STM_automated_filtering/Drift_fine/GMM_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
    
        print('GMM PEAK POSITIONS')
        print('log G (log nS)',M_best.means_)
        print('ln G (ln nS)',np.log(10**(M_best.means_)))   
        print('G (nS)',10**M_best.means_)

        plt.show()
        plt.close()


final_d_df = selected_raw_df[selected_raw_df['FILE INDEX'].isin(d_in_j_indices_list)]
#final_d_df.to_csv(f'./Drift_fine_I_t_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True)
#final_d_200mV_df = max_curr_selected_df[max_curr_selected_df['FILE INDEX'].isin(d_in_j_indices_list)]   
#final_d_200mV_df.to_csv(f'./Drift_200mV_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True)
plt.figure(figsize=(14,4))
plt.scatter(final_d_df['TIME INDEX'],final_d_df['CURRENT'],s=2)
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.savefig(f'./STM_automated_filtering/Drift_fine/I-T_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
plt.show()
plt.close()

# In[18]:
# Histogram of unfiltered data

folder_names = [f'./CTPR_{ctpr_num}_RUN_{run_num}/']

for folder_name in folder_names:
    name_fold = folder_name.split('/')[1]
    print(name_fold)
    location = folder_name 
    dataFolders = natsorted(os.listdir(location  + 'Raw')) 
 
    for folder in dataFolders[:1]:     ## 1: FOR H2O AND :1 FOR D2O
        dataFiles = natsorted(os.listdir(location  + 'Raw' + '/' + folder))   
        print(folder)
        print(len(dataFiles))
        g_up    = []  
        g_down  = []
        g_whole = []

        for i in range(len(dataFiles)):
            tdms_file = TdmsFile(location + 'Raw' + '/' + folder + '/' + dataFiles[i]) 

            # bias
            bias = tdms_file['Untitled']['Bias']
            bias_data = bias.data
            bias_data = np.ndarray.tolist(bias_data)
            bias_up = bias_data[: int(len(bias_data)/2)]
            bias_down = bias_data[int(len(bias_data)/2):]
            
            # current
            current = tdms_file['Untitled']['Current']
            current_data = current.data
            current_data = np.ndarray.tolist(current_data)
            current_up = current_data[: int(len(current_data)/2)]
            current_down = current_data[int(len(current_data)/2):]

            slope_up, intercept_up, r_value_up, p_value_up, std_err_up = linregress(bias_up, current_up)
            slope_down, intercept_down, r_value_down, p_value_down, std_err_down = linregress(bias_down, current_down)

            g_up.append(slope_up)
            g_down.append(slope_down)
            plt.plot(bias_data, current_data)
            plt.xlabel('Bias')
            plt.ylabel('Current')

        log_g_up = np.log10(g_up)
        log_g_down = np.log10(g_down)
        log_g_whole = np.concatenate((log_g_up, log_g_down))

        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(1, 1)
        binwidth = 0.05  
        
        log_g_up.sort()
        log_g_down.sort()
        df = pd.DataFrame({'Log G Up':log_g_up,'Log G Down': log_g_down})
        df.to_csv(f'./STM_automated_filtering/Peaks_and_windows/Unfiltered_Log_G_CTPR_{ctpr_num}_RUN_{run_num}_D2O.csv',header=True) ##saved for plotting in OriginLab
        
        # the histogram of the data
        ax0 = plt.subplot(gs[0])
        ax0 = sns.histplot(log_g_whole, binwidth=binwidth, kde=False, line_kws={'lw': 3},
                           kde_kws={'bw_method': 0.3}, color='blue', edgecolor='black', alpha=0.5)
        ax0.set_ylabel('Counts')
        ax0.set_xlabel('logG')
        ax0.set_xlim(-2.2, 2.2)
        ax0.set_title('Histogram of Unfiltered Data', fontsize=16)
        plt.savefig(f'./STM_automated_filtering/Peaks_and_windows/Raw_Histogram_CTPR_{ctpr_num}_RUN_{run_num}_D2O.png')
        plt.show()
        plt.close()


