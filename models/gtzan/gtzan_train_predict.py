import pandas as pd
import json
import kagglehub
import os
import math
import numpy as np
import pickle
import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as skp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import random
import librosa

random.seed(42)
np.random.seed(42)

PRINT = True # set to FALSE to suppress printing progress statements
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
dataset_path = path + "/Data/genres_original"
model_path = "models/gtzan/gtzan_xgboost.pkl"
predictions_path = "gtzan_xgboost.csv"

# # TODO FdH: load dataset
# # data from https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data
# df = pd.read_csv('../../data/genre/genre_3_sec.csv')
# df = df.drop(labels="filename", axis=1)


# data = df.iloc[0:, 1:]
# y = data['label']
# X = data.loc[:, data.columns != 'label']

# # normalize
# cols = X.columns
# min_max_scaler = skp.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(X)
# X = pd.DataFrame(np_scaled, columns = cols)

# # Top 2 pca components
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(X)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

# # concatenate with target label
# finalDf = pd.concat([principalDf, y], axis = 1)

original_set = path
JSON_PATH = "data/gtzan/data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    id_to_label = {}

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            id_to_label[i] = semantic_label
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file

                file_path = os.path.join(dirpath, f)
            
                # there is an issue with this file
                if 'jazz.00054.wav' not in file_path:

                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                
                
                    # process all segments of audio file
                    for d in range(num_segments):

                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # extract mfcc
                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print("{}, segment:{}".format(file_path, d+1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    return id_to_label

id_to_label = save_mfcc(dataset_path, JSON_PATH)

# path to json
def load_data(data_path):

    with open(data_path, "r") as f:
        data = json.load(f)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y

# load data
X, y = load_data(JSON_PATH)

#Reshape X to 2D for XGBoost compatibility
X_reshaped = X.reshape(X.shape[0], -1)

indices = np.arange(X_reshaped.shape[0])
# Split data into training and testing sets
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_reshaped, y, indices, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb_clf.fit(X_train, y_train)

# Predict probabilities
y_pred_prob = xgb_clf.predict_proba(X_test)

# Optional: Get predicted classes
y_pred_class = np.argmax(y_pred_prob, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {accuracy}")

with open(model_path, 'wb') as file:
    pickle.dump(xgb_clf, file)

sm_dict = {f"label_{i}": y_pred_prob[:, i] for i in range(y_pred_prob.shape[1])}
frame = pd.DataFrame({
    "idx": idx_test,
    "label": y_test,
    **sm_dict
})
frame.to_csv("gtzan_scores.csv", index_label='id')

pprint.pprint(id_to_label)
