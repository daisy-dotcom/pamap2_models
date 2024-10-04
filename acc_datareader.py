from features import *
from sklearn.model_selection import LeaveOneOut, train_test_split

class AccDataReader:

  def __init__(self,
               data_path,
               data_cols=[4, 5, 6],
               label_col=[1],
               samp_rate=100,
               window_size=1):

    self.data_path = data_path
    self.data_cols = data_cols
    self.label_col = label_col
    self.window_size = window_size * samp_rate

    self.data = []
    self.labels = []

    self.x_train_idx = []
    self.y_train_idx = []

    self.x_test_idx = []
    self.y_test_idx = []

    self.x_train = []
    self.y_train = []
    self.x_test = []
    self.y_test = []

    self.file_list = self._get_file_list(self.data_path)
    self._read_data()

  def _get_file_list(self, data_path):
    self.cwd = os.getcwd()
    try:
      file_list = [ single_file for single_file in os.listdir(os.path.join(self.cwd, self.data_path)) \
                if os.path.isfile(os.path.join(self.cwd, self.data_path, single_file))]
      #return ['subject104.dat','subject107.dat', 'subject109.dat']
      return file_list
    except OSError as e:
      print(f'Error: {e}')
      return []

  def _read_data(self):

    data = []
    labels = []
    no_data = []

    for single_file in self.file_list:
      with open(os.path.join(self.cwd, self.data_path, single_file), 'r') as f:
        individual_data = []
        individual_labels = []
        print(f"Reading {single_file}")
        for line in f.readlines():
          reading = line.split()

          # exclude readings with a label = 0 and x,y,z = not a number (nan)
          label = reading[self.label_col[0]]
          x, y, z = reading[self.data_cols[0]], reading[self.data_cols[1]], reading[self.data_cols[2]]

          if (label != '0') and ('NaN' not in (x,y,z)) and (int(label) in labels_to_keep):
            individual_data.append([x,y,z])
            if label == '7':
              label = '4'
            elif label == '12':
              label = '6'
            elif label == '13':
              label = '6'

            individual_labels.append(f'{int(label)-1}')
      if len(individual_data) > 0 and len(individual_data) > 0 :
        data.append(individual_data)
        labels.append(individual_labels)
      else:
        no_data.append(single_file)

    for data_file in no_data:
      print(f'No data in {data_file}')
      self.file_list.remove(data_file)

    # length of data and label lists = number of subjects
    self.data = data
    self.labels = labels

  def loso_groups(self):
    loo = LeaveOneOut()

    x_train_idx = []
    y_train_idx = []

    x_test_idx = []
    y_test_idx = []

    for i, (train_index, test_index) in enumerate(loo.split(self.data)):
      #print(f"Fold {i}:")
      #print(f"Train: index={train_index}")
      #print(f"Test:  index={test_index}")

      x_train_idx.append(train_index)
      x_test_idx.append(test_index)

    self.x_train_idx = x_train_idx
    self.x_test_idx = x_test_idx

    #return x_train_idx, x_test_idx

  def all_data(self):
    x_train_idx = [i for i in range(len(self.data))]
    x_test_idx = []

    self.x_train_idx = x_train_idx
    self.x_test_idx = x_test_idx

  def get_train_test_data(self, x_train_idx, x_test_idx):
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for idx in x_train_idx:
      x_train += self.data[idx]
      y_train += self.labels[idx]

    for idx in x_test_idx:
      x_test += self.data[idx]
      y_test += self.labels[idx]

    x_train = np.array(x_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int32)

    x_test = np.array(x_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.int32)

    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test

  def get_features(self, f_set='yuan'):
    if f_set == 'yuan':
      self.x_train = np.array([yuan_feature_eng(self.x_train[i:i+self.window_size]) for i in range(0,len(self.x_train),self.window_size) if len(self.x_train[i:i+self.window_size]) == self.window_size])
      self.x_test = np.array([yuan_feature_eng(self.x_test[i:i+self.window_size]) for i in range(0,len(self.x_test),self.window_size) if len(self.x_test[i:i+self.window_size]) == self.window_size])
    elif f_set == 'yaz':
      self.x_train = np.array([yaz_feature_eng(self.x_train[i:i+self.window_size]) for i in range(0,len(self.x_train),self.window_size) if len(self.x_train[i:i+self.window_size]) == self.window_size])
      self.x_test = np.array([yaz_feature_eng(self.x_test[i:i+self.window_size]) for i in range(0,len(self.x_test),self.window_size) if len(self.x_test[i:i+self.window_size]) == self.window_size])
    else:
      self.x_train = np.array([yuan_yaz(self.x_train[i:i+self.window_size]) for i in range(0,len(self.x_train),self.window_size) if len(self.x_train[i:i+self.window_size]) == self.window_size])
      self.x_test = np.array([yuan_yaz(self.x_test[i:i+self.window_size]) for i in range(0,len(self.x_test),self.window_size) if len(self.x_test[i:i+self.window_size]) == self.window_size])


  def get_mode_labels(self):
    self.y_train = np.array(mode_labels(self.y_train, self.window_size))
    self.y_test = np.array(mode_labels(self.y_test, self.window_size))

