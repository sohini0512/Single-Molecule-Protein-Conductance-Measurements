
#IMPORTING LIBRARIES

from nptdms import TdmsFile
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy.signal import chirp, find_peaks, peak_widths
from scipy import io
from scipy import sparse
from scipy.linalg import solveh_banded
from scipy.sparse.linalg import spsolve


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################


def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,
                 max_iters=10, conv_thresh=1e-5, verbose=False):
  '''Computes the asymmetric least squares baseline.
  * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
  smoothness_param: Relative importance of smoothness of the predicted response.
  asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                       Setting p=1 is effectively a hinge loss.
  '''
  smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
  # Rename p for concision.
  p = asymmetry_param
  # Initialize weights.
  w = np.ones(intensities.shape[0])
  for i in range(max_iters):
    z = smoother.smooth(w)
    mask = intensities > z
    new_w = p*mask + (1-p)*(~mask)
    conv = np.linalg.norm(new_w - w)
    if verbose:
      print(i+1, conv)
    if conv < conv_thresh:
      break
    w = new_w
  else:
    print( 'ALS did not converge in %d iterations' % max_iters)
  return z


class WhittakerSmoother(object):
  def __init__(self, signal, smoothness_param, deriv_order=1):
    self.y = signal
    assert deriv_order > 0, 'deriv_order must be an int > 0'
    # Compute the fixed derivative of identity (D).
    d = np.zeros(deriv_order*2 + 1, dtype=int)
    d[deriv_order] = 1
    d = np.diff(d, n=deriv_order)
    n = self.y.shape[0]
    k = len(d)
    s = float(smoothness_param)

    # Here be dragons: essentially we're faking a big banded matrix D,
    # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
    diag_sums = np.vstack([
        np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
        for i in range(1, k+1)])
    upper_bands = np.tile(diag_sums[:,-1:], n)
    upper_bands[:,:k] = diag_sums
    for i,ds in enumerate(diag_sums):
      upper_bands[i,-i-1:] = ds[::-1][:i+1]
    self.upper_bands = upper_bands

  def smooth(self, w):
    foo = self.upper_bands.copy()
    foo[-1] += w  # last row is the diagonal
    return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)




###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################











#LOADING THE TDMS FILE

num_to_plot = 2

for i in range(1,num_to_plot):
	
    tdms_file = TdmsFile("A10C10_dNTP (%s).tdms"%i)
    tdms_groups = tdms_file.groups()
    attributes = tdms_file.object().properties 
    tdms_0 = tdms_file.group_channels("0")
    tdms_1 = tdms_file.group_channels("-0")
    Group_Channel_0_1 = tdms_file.object('0', 'Current')
    Group_Channel_data_0_1 = Group_Channel_0_1.data
    Group_Channel_1_1 = tdms_file.object('-0', 'Current')
    Group_Channel_data_1_1 = Group_Channel_1_1.data
    Group_Channel_Current = np.append(Group_Channel_data_0_1, Group_Channel_data_1_1)*(10/32768)
    print("shape of GCC", Group_Channel_Current.shape)
    x = np.linspace(0,Group_Channel_Current.shape[0],Group_Channel_Current.shape[0]  )*(1/50000)
    plt.plot(x,Group_Channel_Current)
    plt.savefig("Raw_Signal_A10C10_dNTP(%s).png"%i, bbox_inches="tight", dpi=600)
    plt.show()
    narr_curr = np.column_stack((x, Group_Channel_Current))
    io.savemat("Raw_Signal_A10C10_dNTP(%s).mat"%i, {"array": narr_curr})
    plt.close()  
	
	
    #ASYMMETRIC LEAST SQUARES BASELINE TREATMENT
    
    
    
    AsLS_baseline_treatment = als_baseline(Group_Channel_Current, asymmetry_param=0.999, smoothness_param=10e6,max_iters=10, conv_thresh=1e-5, verbose=False) 
    plt.plot(x,AsLS_baseline_treatment)
    plt.savefig('AsLS_treated_signal_A10C10_dNTP (%s).png'%i)
    plt.show()
    narr_AsLS = np.column_stack((x, AsLS_baseline_treatment))
    io.savemat("AsLS_treated_signal_A10C10_dNTP (%s).mat"%i, {"array": narr_AsLS})
    plt.close()
    
    RAW_MINUS_AsLS_treated_signal = np.subtract(Group_Channel_Current,AsLS_baseline_treatment)
    plt.plot(x,RAW_MINUS_AsLS_treated_signal)
    plt.savefig('RAW_MINUS_AsLS_treated_signal_A10C10_dNTP (%s).png'%i)
    plt.show()
    narr_final = np.column_stack((x, RAW_MINUS_AsLS_treated_signal))
    io.savemat("RAW_MINUS_AsLS_treated_signal_A10C10_dNTP (%s).mat"%i, {"array": narr_final})
    plt.close()
    
    #inverting the data
    
    RAW_MINUS_AsLS_treated_signal = RAW_MINUS_AsLS_treated_signal * -1.0
    
    plt.subplots(figsize=(8, 3))
    plt.yticks(size=14);
    plt.xticks(size=14);
    plt.ylabel('current_inv', size=14)
    plt.xlabel('time', size=14)
    plt.plot(x,RAW_MINUS_AsLS_treated_signal, color='#00bbcc', linewidth=1)
    plt.savefig("Main_peaks_AsLS_inverted_A10C10_dNTP (%s).png"%i, bbox_inches="tight", dpi=600)
    plt.show()
    narr_curr_AsLS = np.column_stack((x, RAW_MINUS_AsLS_treated_signal))
    io.savemat("Main_peaks_AsLS_inverted_A10C10_dNTP (%s).mat"%i, {"array": narr_curr_AsLS})
    plt.close()
    
    #peak detection threshold and peak position 
    
    avg_curr_AsLS = np.mean(RAW_MINUS_AsLS_treated_signal)
    
    peaks_AsLS, _ = find_peaks(RAW_MINUS_AsLS_treated_signal, height=avg_curr_AsLS )
    
    
    def pairwiseDifference(arr, n):
        tmp = []
        for j in range(n - 1) : 
            # absolute difference between 
            # consecutive numbers 
            diff = abs(arr[j] - arr[j + 1]) 
            print(diff , end = " ")
            tmp.append(diff)
            tmp = np.asarray(tmp)
            return tmp
    
    
    arr_len_AsLS = peaks_AsLS.shape[0]
    
    peak_diff_AsLS = pairwiseDifference(peaks_AsLS, arr_len_AsLS)
    
    
    results_full = peak_widths(RAW_MINUS_AsLS_treated_signal, peaks_AsLS, rel_height=1)
    results_half = peak_widths(RAW_MINUS_AsLS_treated_signal, peaks_AsLS, rel_height=0.5)
    print(results_full[0])
    plt.subplots(figsize=(8, 3))
    plt.yticks(size=14);
    plt.xticks(size=14);
    plt.ylabel('current', size=14)
    plt.xlabel('time', size=14)
    plt.hlines(*results_half[1:], color="C2")
    plt.hlines(*results_full[1:], color="C3")
    plt.plot(RAW_MINUS_AsLS_treated_signal)
    plt.plot(peaks_AsLS, RAW_MINUS_AsLS_treated_signal[peaks_AsLS], "x")
    plt.savefig("Peak_position_method_AsLS_A10C10_dNTP (%s).png"%i, bbox_inches="tight", dpi=600)
    plt.show()
    plt.close()
    
    #peak distance histogram
    
    plt.yticks(size=14);
    plt.xticks(size=14);
    plt.ylabel('Count', size=14)
    plt.xlabel('Peak_Distance', size=14)
    plt.hist(peak_diff_AsLS, bins=100,edgecolor='black',facecolor='green')
    plt.savefig("Peak_Distance_method_AsLS_A10C10_dNTP (%s).png"%i, bbox_inches="tight", dpi=600)
    plt.show()
    io.savemat("Peak_Distance_method_AsLS_A10C10_dNTP (%s).mat"%i, {"array": peak_diff_AsLS}, oned_as='column')
    plt.close()
    
    #maximum peak width histogram
    
    plt.yticks(size=14);
    plt.xticks(size=14);
    plt.ylabel('Count', size=14)
    plt.xlabel('Base Width', size=14)
    plt.hist(results_full[0], bins=100,edgecolor='black',facecolor='blue')
    plt.savefig("Peak_Full_width_method_AsLS_A10C10_dNTP (%s).png"%i, bbox_inches="tight", dpi=600)
    plt.show()
    results_full_data = results_full[0]
    io.savemat("Peak_Full_width_method_AsLS_A10C10_dNTP (%s).mat"%i, {"array": results_full_data}, oned_as='column')
    plt.close()
    
    #FWHM histogram
    
    plt.yticks(size=14);
    plt.xticks(size=14);
    plt.ylabel('Count', size=14)
    plt.xlabel('FWHM', size=14)
    plt.hist(results_half[0], bins=100,edgecolor='black',facecolor='red')
    plt.savefig("Peak_width_half_maxima_method_AsLS_A10C10_dNTP (%s).png"%i, bbox_inches="tight", dpi=600)
    plt.show()
    results_half_data = results_half[0]
    io.savemat("Peak_width_half_maxima_method_AsLS_A10C10_dNTP (%s).mat"%i, {"array": results_half_data}, oned_as='column')
    plt.close()
