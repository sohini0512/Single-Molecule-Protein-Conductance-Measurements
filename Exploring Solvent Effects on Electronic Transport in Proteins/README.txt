Steps:
1. Data folders: './CTPR_{ctpr_num}_RUN_{run_num}/Raw' , with subfolders named 'H2O' and 'D2O'
2. Create I-t traces from the data folders using 'Making raw i_t_dataframes.ipynb' script
3. Create folder for saving analysis results: 'STM_automated_filtering'
4. Create sub-folders: 
    1. Drift_coarse : Drift I-V sweeps after first coarse filtering 
	2. Drift_fine : Drift I-V sweeps after second fine filtering
	3. Junction_coarse : Junction I-V sweeps after first coarse filtering
	4. Junction_fine : Junction I-V sweeps after second fine filtering
	5. Peaks_and_windows : Exponential windows from raw Current vs Time data
	6. Slope_drift_coarse : Inforation on drift velocity from slope
5. Run the script 'STM_AUTOMATED_FILTERING.py' for filtering and saving the output in the sub-folders created in step 4.