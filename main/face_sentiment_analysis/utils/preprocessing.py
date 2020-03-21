import pandas as pd
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read("../config.ini")

BASE_DIR = config["emotion-classify"]["DATA_BASEPATH"]
DATA_CSVPATH = os.path.join(BASE_DIR, "fer2013/fer2013.csv")
WIDTH, HEIGHT = np.repeat(
    config.getint("emotion-classify", "IMG_WIDTH_HEIGHT"), 
    2)

data = pd.read_csv(TRAIN_CSVPATH)
datapoints = data['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
# y = pd.get_dummies(data['emotion']).as_matrix()

# #storing them using numpy
# np.save('fdataX', X)
# np.save('flabels', y)

# print("Preprocessing Done")
# print("Number of Features: "+str(len(X[0])))
# print("Number of Labels: "+ str(len(y[0])))
# print("Number of examples in dataset:"+str(len(X)))
# print("X,y stored in fdataX.npy and flabels.npy respectively")