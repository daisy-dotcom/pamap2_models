import os
import numpy as np
from numpy import linalg as LA
from pywt import wavedec
import scipy.stats as stats
import scipy.signal as signal


pamap2_label_names = [
    'lying',
    'sitting',
    'standing',
    'walking',
    'running',
    'cycling',
    'Nordic_walking',
    'ascending stairs',
    'descending stairs',
    'vacuum_cleaning',
    'ironing',
    'rope_jumping'
]

pamap2_label_num = [1,2,3,4,5,6,7,12,13,16,17,24]

pamap2_updated_labels =[
    'lying',
    'sitting',
    'standing',
    'walking',
    'running',
    'stairs',
]

'''
  1 - lying
  2 - sitting
  3 - standing
  4 - walking
  5 - running
  7 - Nordic_walking  -> changed to the same label as walking
  12 - ascending stairs
  13 - descending stairs

  ascending/descending stais -> just stairs now
'''
labels_to_keep = [1,2,3,4,5,7,12,13]

new_labels = [0,1,2,3,4,5]

def mode_labels(labels, window_size=100):
  # only keep the label and not the count
  return [stats.mode(labels[i:i+window_size])[0] for i in range(0,len(labels),window_size) if len(labels[i:i+window_size]) == window_size]

# Source: Yuan et al. 2024
# Link: https://github.com/OxWearables/ssl-wearables/blob/main/downstream_task_evaluation.py#L483
def yuan_feature_eng(batch, samp_rate=100, window_size=1):
  # time domain features
  # x, y, z mean
  mean = np.mean(batch, axis=0)

  # x, y, z standard deviation
  std = np.std(batch, axis=0)

  # x, y, z range
  range = np.ptp(batch, axis=0)

  # division error handling
  with np.errstate(divide='ignore', invalid='ignore'):
    x, y, z = batch.T
    # 3x3 correlation matrix
    corr = np.nan_to_num(np.corrcoef(batch, rowvar=False))
    corr_xy = np.nan_to_num(np.corrcoef(x,y))[0,1]
    corr_xz = np.nan_to_num(np.corrcoef(x,y))[0,1]
    corr_yz = np.nan_to_num(np.corrcoef(y,z))[0,1]


  # Euclidean norm features
  # mean, standard deviation, range, median
  # absolute deviation, kurtosis and skew
  euclid_norm = LA.norm(batch, axis=1)
  euclid_norm_mean = np.mean(euclid_norm)
  euclid_norm_std = np.std(euclid_norm)
  euclid_norm_range = np.ptp(euclid_norm)
  euclid_norm_mad = stats.median_abs_deviation(euclid_norm)

  if euclid_norm_std > 0.01:
    skew = np.nan_to_num(stats.skew(euclid_norm))
    kurtosis = np.nan_to_num(stats.kurtosis(euclid_norm))
  else:
    skew = kurtosis = 0

  # get true power spectrum because detrend=False
  freqs, powers = signal.welch(
      euclid_norm,
      fs=samp_rate,
      nperseg= samp_rate*window_size,
      noverlap=(2 * samp_rate*window_size) // 3,
      detrend=False,
      average='median')

  with np.errstate(divide="ignore", invalid="ignore"):
        pentropy = np.nan_to_num(stats.entropy(powers + 1e-16))

  # get dominant frequencies
  freqs, powers = signal.welch(
    euclid_norm,
    fs=samp_rate,
    nperseg= samp_rate*window_size,
    noverlap=(2 * samp_rate*window_size) // 3,
    detrend='constant',
    average='median')

  with np.errstate(divide="ignore", invalid="ignore"):
        pentropy = np.nan_to_num(stats.entropy(powers + 1e-16))


  peaks, _ = signal.find_peaks(powers)
  peak_powers = powers[peaks]
  peak_freqs = freqs[peaks]
  peak_ranks = np.argsort(peak_powers)[::-1]
  if len(peaks) >= 2:
      f1 = peak_freqs[peak_ranks[0]]
      f2 = peak_freqs[peak_ranks[1]]
  elif len(peaks) == 1:
      f1 = f2 = peak_freqs[peak_ranks[0]]
  else:
      f1 = f2 = 0

  # 21 features
  features = np.concatenate((mean, std, range, corr_xy, corr_xz, corr_yz,
                             euclid_norm_mean, euclid_norm_std, euclid_norm_range, euclid_norm_mad,
                             skew, kurtosis, pentropy, f1, f2),axis=None)

  return features

def yaz_feature_eng(batch):
  # time domain features
  # x, y, z mean
  mean = np.mean(batch, axis=0)

  # x, y, z standard deviation
  std = np.std(batch, axis=0)

  # x, y, z 50th percentile
  fifty_th = np.percentile(batch, 50, axis=0)

  # vector magnitude calculated using Frobenius norm
  vec_mag = LA.norm(batch, axis=1)
  vec_mag_mean = np.mean(vec_mag)
  vec_mag_std = np.std(vec_mag)
  vec_mag_50th = np.percentile(vec_mag, 50)

  # frequency domain features extracted using discrete
  # wavelet transforms
  # x, y, z level 1 wave decomposition
  cA1, cD = wavedec(batch, 'db1', level=1, axis=0)

  # x, y, z level 5 wave decomposition
  cA5, cD5, cD4, cD3, cD2, CD1 = wavedec(batch, 'db1', level=5, axis=0)

  # mean of level 1 and 5 approximation coefficients
  cA1_mean = np.mean(cA1, axis=0)
  cA5_mean = np.mean(cA1, axis=0)

  # mean of level 1 and 5 detail coefficients
  cD_mean = np.mean(cD, axis=0)
  cD5_mean = np.mean(cD5, axis=0)

  x_cA_mean_diff = cA1_mean[0] - cA5_mean[0]
  x_cD_mean_diff = cD_mean[0] - cD5_mean[0]

  y_cA_mean_diff = cA1_mean[1] - cA5_mean[1]
  y_cD_mean_diff = cD_mean[1] - cD5_mean[1]

  # 16 features
  features = np.concatenate((mean, std, fifty_th, vec_mag_mean, vec_mag_std, vec_mag_50th,
                             x_cA_mean_diff, x_cD_mean_diff, y_cA_mean_diff, y_cD_mean_diff),axis=None)

  return features

def yuan_yaz(batch, samp_rate=100, window_size=1):
  # Shared Features
  start_yaz = 2
  # 13 feat yuan + 8
  yuan_features = yuan_feature_eng(batch, samp_rate, window_size)
  yaz_features = yaz_feature_eng(batch)

  features = np.concatenate((yuan_features, yaz_features[start_yaz:]),axis=None)

  return features