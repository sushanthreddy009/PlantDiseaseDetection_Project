from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configure GPU memory growth
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training Dataset
training_set = train_datagen.flow_from_directory(
    r'C:\Users\Annie3008\Desktop\Plant-Leaf-Disease-Prediction\Dataset\train',
    target_size=(128, 128),
    batch_size=6,
    class_mode='categorical'
)

# Validation Dataset
valid_set = test_datagen.flow_from_directory(
    r'C:\Users\Annie3008\Desktop\Plant-Leaf-Disease-Prediction\Datasetnew\New Plant Diseases Dataset(Augmented)\valid',
    target_size=(128, 128),
    batch_size=3,
    class_mode='categorical'
)

labels = training_set.class_indices
print(labels)

history2 = classifier.fit(training_set, steps_per_epoch=50, epochs=50, validation_data=valid_set)

plt.plot(history2.history['accuracy'], label='accuracy')
plt.plot(history2.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)

# Serialize weights to HDF5
classifier.save_weights("my_model_weights.h5")
classifier.save("model.h5")
print("Saved model to disk")