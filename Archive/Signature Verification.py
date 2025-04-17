import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


#LOAD AND PREPROCESS DATA
data_gen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # Split data into training and validation
)

train_data = data_gen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),  # Resize images
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='training'  # Use for training data
)

val_data = data_gen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Use for validation data
)

#BUILD MODEL
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#TRAIN MODEL
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,          # Stop if no improvement for 5 epochs
    restore_best_weights=True  # Revert to the best model weights
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,  # Maximum number of epochs
    callbacks=[early_stopping]
)

#EVALUATE MODEL
loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy:.2f}")

#TEST MODEL
from tensorflow.keras.preprocessing import image

# model = tf.keras.models.load_model('signature_verification_model.h5')

def predict_signature(image_path):
    img = image.load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return "Authentic" + " " +str(prediction[0][0]) if prediction[0][0] < 0.4 else "Forged" + " " + str(prediction[0][0])

print("1- " + predict_signature('test_signatures/test_signature.jpg'))
print("2- " + predict_signature('test_signatures/test_signature2.jpg'))
print("3- " + predict_signature('test_signatures/test_signature3.jpg'))
print("4- " + predict_signature('test_signatures/test_signature4.jpg'))
print("5- " + predict_signature('test_signatures/test_signature5.jpg'))
print("6- " + predict_signature('test_signatures/test_signature6.jpg'))
print("7- " + predict_signature('test_signatures/test_signature7.jpg'))
print("7- " + predict_signature('test_signatures/forged.jpg'))
# model.save('signature_verification_model.h5')
