import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

# Function to extract features from a file given intervals
def extract_features(file_name, intervals):
    y, sr = librosa.load(file_name, sr=16000)
    features = []
    for interval in intervals:
        start_sample = int(interval[0] * sr)
        end_sample = int(interval[1] * sr)
        y_interval = y[start_sample:end_sample]
        mfcc = librosa.feature.mfcc(y=y_interval, sr=sr)
        mfcc_mean = np.mean(mfcc, axis=1)
        features.append(mfcc_mean)
    return features

# Define intervals for each vowel (both short and long)
intervals = {
    'a': [(1.1, 1.34), (0.47, 1.26)],
    'i': [(1.97, 2.2), (1.31, 2.14)],
    'u': [(2.77, 3.01), (2.17, 2.97)],
    'e': [(3.59, 3.82), (3.01, 3.62)],
    'o': [(4.38, 4.62), (3.83, 4.23)]
}

# Labels for vowels
vowel_labels = {'a': 1, 'i': 2, 'u': 3, 'e': 4, 'o': 5}

# Extract features and labels
features = []
labels = []
for vowel, intervals in intervals.items():
    features += extract_features('aiueo.wav', [intervals[0]])  # Short vowel
    features += extract_features('aiueo_.wav', [intervals[1]])  # Long vowel
    labels += [vowel_labels[vowel]] * 2  # Same label for both short and long

# Train a classifier
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(features, labels)

# Save the model
with open('aiueo_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

print("Model trained and saved as aiueo_model.pkl")
