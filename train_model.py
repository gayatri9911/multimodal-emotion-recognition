import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# LOAD DATA
# ==========================

X = np.load("X.npy")
y = np.load("y.npy")

# ==========================
# LABEL ENCODING
# ==========================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.2,

    random_state=42,

    stratify=y_encoded
)

# ==========================
# INPUT SHAPE
# ==========================

input_shape = (

    X_train.shape[1],

    X_train.shape[2]
)

# ==========================
# MODEL
# ==========================

model = Sequential()

# CNN Layer
model.add(Conv1D(

    filters=64,

    kernel_size=3,

    activation='relu',

    input_shape=input_shape
))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.3))

# Second CNN Layer
model.add(Conv1D(

    filters=128,

    kernel_size=3,

    activation='relu'
))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.3))

# LSTM Layer
model.add(LSTM(

    128,

    return_sequences=False
))

model.add(Dropout(0.3))

# Dense Layers
model.add(Dense(

    64,

    activation='relu'
))

model.add(Dropout(0.3))

# Output Layer
model.add(Dense(

    len(np.unique(y_encoded)),

    activation='softmax'
))

# ==========================
# COMPILE MODEL
# ==========================

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)

# ==========================
# EARLY STOPPING
# ==========================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=10,

    restore_best_weights=True
)

# ==========================
# TRAIN
# ==========================

history = model.fit(

    X_train,
    y_train,

    epochs=100,

    batch_size=32,

    validation_data=(
        X_test,
        y_test
    ),

    callbacks=[early_stop]
)

# ==========================
# EVALUATE
# ==========================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nFinal Accuracy:", accuracy)

# ==========================
# SAVE MODEL
# ==========================

model.save(
    "models/emotion_model.h5"
)

print("\nCNN + LSTM Model Saved")