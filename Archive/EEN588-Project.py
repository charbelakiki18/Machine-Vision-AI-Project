import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize the images to the range [0, 1]
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0

# # Reshape the images to include a channel dimension (grayscale)
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# # Convert the labels to categorical format (one-hot encoding)
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

# # Build the model
# model = Sequential()

# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# model.save("mnist_model2.h5")

# model = load_model('best_model.h5')

# # Load the image
# img = cv2.imread("Tests/test.jpg", cv2.IMREAD_GRAYSCALE)

# # Resize the image to 28x28 pixels
# img_resized = cv2.resize(img, (28, 28))

# # Normalize the image
# img_resized = img_resized.astype('float32') / 255.0

# # Reshape the image to match the input shape of the model
# img_resized = np.expand_dims(img_resized, axis=-1)  # Shape becomes (28, 28, 1)
# img_resized = np.expand_dims(img_resized, axis=0)   # Shape becomes (1, 28, 28, 1)

# # Predict the digit
# pred = model.predict(img_resized)

# # Get the predicted class (the digit)
# predicted_class = np.argmax(pred)

# # Print the prediction
# print(f"The predicted digit is: {pred}")

# # Optionally, display the image
# plt.imshow(img_resized[0], cmap='gray')
# plt.title(f"Predicted Digit: {predicted_class}")
# plt.show()

img = cv2.imread("Tests/cheque3.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Step 1: Apply Gaussian Blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# # Step 2: Enhance contrast using Adaptive Histogram Equalization (CLAHE)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# contrast_enhanced = clahe.apply(blurred)

# # Step 3: Thresholding to isolate text
# _, binary_thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Step 4: Morphological operations to remove noise and enhance text
# kernel = np.ones((2, 2), np.uint8)
# morphed = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)

# # Step 5: Sharpening to make text edges clearer
# kernel_sharpen = np.array([[-1, -1, -1], 
#                                 [-1, 9, -1], 
#                                 [-1, -1, -1]])
# sharpened = cv2.filter2D(morphed, -1, kernel_sharpen)
text = pytesseract.image_to_string(img)
print("OUTPUT")
print(text)
