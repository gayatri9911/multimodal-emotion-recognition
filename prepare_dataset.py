import os
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.feature_extraction import extract_features

features = []
labels = []

emotion_dict = {

    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry'
}

dataset_path = "dataset"

for file in os.listdir(dataset_path):

    if file.endswith(".wav"):

        try:

            emotion_code = file.split("-")[2]

            if emotion_code not in emotion_dict:
                continue

            emotion = emotion_dict[
                emotion_code
            ]

            file_path = os.path.join(
                dataset_path,
                file
            )

            feature = extract_features(
                file_path
            )

            if feature is not None:

                features.append(feature)

                labels.append(emotion)

                print("Processed:", file)

        except Exception as e:

            print("Error:", file)

# ==========================
# PAD SEQUENCES
# ==========================

X = pad_sequences(

    features,

    padding='post',

    dtype='float32'
)

y = np.array(labels)

np.save("X.npy", X)
np.save("y.npy", y)

print("\nDataset Prepared Successfully")
print("Shape:", X.shape)