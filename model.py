from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

## DATA INPUT

## Import dummy dataset to test the model

(images_train, labels_train), (images_test, labels_test) = cifar10.load_data()

labels = ['airplane', 
          'automobile',
          'bird',
          'cat',
          'deer',
          'dog',
          'frog',
          'horse',
          'ship',
          'truck']

print(images_train.shape, images_test.shape)
unique, counts = np.unique(labels_train, return_counts=True)
dict(zip(unique, counts))

# Considering only 1/10th of the 50_000 images
reduction_factor = 10

# Choosing the random indices of small train set and small test set
idx_train =  np.random.choice(len(images_train), round(len(images_train)/reduction_factor), replace=False)
idx_test =  np.random.choice(len(images_test), round(len(images_test)/reduction_factor), replace=False)

# Collecting the two subsamples images_train_small and images_test_small from images_train and images_test
images_train_small = images_train[idx_train]
images_test_small = images_test[idx_test]

# and their corresponding labels
labels_train_small = labels_train[idx_train]
labels_test_small = labels_test[idx_test]

unique, counts = np.unique(labels_train_small, return_counts=True)
dict(zip(unique, counts))

##DATA PREPROCESSING

#Here we are going to do a simple preprocessing that should be tested.

#Depending on the model's performance this preprocessing should be improved with image data augmentation

### Normalizing pixels' intensities
X_train_small = images_train_small / 255.
X_test_small = images_test_small / 255.

### Encoding the labels
y_train_small = to_categorical(labels_train_small, 10)
y_test_small = to_categorical(labels_test_small, 10)

#SETTING UP LIBRAI MODEL ARCHITECTURE

def initialize_librai():
    
    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (2, 2), activation = 'relu', padding = 'same'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation = 'softmax'))
    
    return model

# MODEL COMPILING

def compile_librai(model):
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

# MODEL INSTANTIATING, COMPILING AND PREDICTING

model_small = initialize_librai()
model_small = compile_librai(model)

es = EarlyStopping(patience = 5, verbose = 2, restore_best_weights=True)

history_small = model_small.fit(X_train_small, y_train_small, 
                    validation_split = 0.2,
                    callbacks = [es], 
                    epochs = 100, 
                    batch_size = 32)

y_pred = model.predict(X_test_small)

# TRANSLATING OUTPUT

def translate_output(y_pred):
    max_index = np.argmax(y_pred,axis=1)
    translated_output = [labels[idx] for idx in max_index]
    return translated_output