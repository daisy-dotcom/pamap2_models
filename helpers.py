import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

from acc_datareader import *

def run(data, f_set='yuan'):
  data.loso_groups()

  result = {
      "accuracy": [],
      "precision": [],
      "recall": [],
      "cohen_kappa": [],
      "f1_score": [],
      "conf_matrix": []
  }

  for i in range(len(data.x_train_idx)):
    data.get_train_test_data(data.x_train_idx[i], data.x_test_idx[i])

    # extract features
    data.get_features(f_set=f_set)

    # get window labels
    data.get_mode_labels()

    # random forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(data.x_train, data.y_train)
    y_pred = rf.predict(data.x_test)

    accuracy, precision, recall, cohen_kappa, f1_score, conf_matrix = performance_eval(data.y_test, y_pred)

    result["accuracy"].append(accuracy)
    result["precision"].append(precision)
    result["recall"].append(recall)
    result["cohen_kappa"].append(cohen_kappa)
    result["f1_score"].append(f1_score)
    result["conf_matrix"].append(conf_matrix)


  result["accuracy"] = np.mean(result["accuracy"])
  result["precision"] = np.mean(result["precision"], axis=0)
  result["recall"] = np.mean(result["recall"], axis=0)
  result["cohen_kappa"] = np.mean(result["cohen_kappa"], axis=0)
  result["f1_score"] = np.mean(result["f1_score"], axis=0)
  return result

def performance_eval(y_true, y_pred):
  accuracy = metrics.accuracy_score(y_true, y_pred)
  precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0.0)
  recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0.0)
  cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)
  f1_score = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0.0)
  conf_matrix = confusion_matrix(y_true, y_pred, labels=new_labels, normalize='true')

  return accuracy, precision, recall, cohen_kappa, f1_score, conf_matrix


def rf_train_all_data(data, f_set='yuan'):
  data.all_data()
  data.get_train_test_data(data.x_train_idx, data.x_test_idx)
  # extract features
  data.get_features(f_set=f_set)

  # get window labels
  data.get_mode_labels()

  # random forest
  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(data.x_train, data.y_train)

  return rf

def save_model(model_name, model):
  model_path = f'models/{model_name}.pkl'
  pickle.dump(model, open(model_path, 'wb'))

def load_model(model_name):
  model_path = f'models/{model_name}.pkl'
  return pickle.load(open(model_path, 'rb'))

def rf_windowing(data, window_size=1, sampling_rate=100, f_set='yuan'):
  window_size= window_size * sampling_rate
  featured_data = np.array(
    [yuan_feature_eng(data[i:i + window_size]) for i in range(0, len(data), window_size) if
     len(data[i:i + window_size]) == window_size])

  return featured_data
