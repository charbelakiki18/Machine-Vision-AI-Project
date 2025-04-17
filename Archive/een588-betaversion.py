import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
import numpy as np
import pytesseract
import re
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


############################################################
#PART ZERO: TRAINING SIGNATURE VERIFICATION MODEL USING CNN#
############################################################

#LOAD AND PREPROCESS DATA
data_gen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # Split data into training and validation
)

train_data = data_gen.flow_from_directory(
    'dataset/', #Folder of the dataset
    target_size=(128, 128),  # Resize images
    color_mode='grayscale', #colormode
    batch_size=16, #process 16 samples simultaneously
    class_mode='binary', #binary classification, authentic vs forged
    subset='training'  # Use for training data
)

val_data = data_gen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=16,
    class_mode='binary',
    subset='validation'  # Use for validation data
)

#BUILD MODEL
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)), #32 filters of size 3x3
    layers.MaxPooling2D((2, 2)), #pooling (subsampling)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), #flatten from 2d to 1d
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification, gives a probability of the result, 0 is for Authentic, 1 for forged
])

model.compile(optimizer='adam', #for dynamic weights adjustments
              loss='binary_crossentropy', #evakuates model performance and updates optimizer
              metrics=['accuracy'])


#TRAIN MODEL
early_stopping = EarlyStopping( #To avoid overfitting
    monitor='val_loss',  # Monitor validation loss
    patience=3,          # Stop if no improvement for 3 epochs
    restore_best_weights=True  # Revert to the best model weights
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,  # Maximum number of epochs
    callbacks=[early_stopping]
)

#EVALUATE MODEL
loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy:.2f}")

#TEST MODEL
# model = tf.keras.models.load_model('signature_verification_model.h5')

def predict_signature(image_path):
    if isinstance(image_path, str): #If input is a string (path) load it
        img = image.load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    else: #otherwise directly apply transformations
        actual_image = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(actual_image, (128, 128))
    img_array = image.img_to_array(img) / 255.0 #Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension since model expects batch size with input
    prediction = model.predict(img_array) #model returns (batchsize, prediction 0 or 1)
    return "Valid" + " " +str(prediction[0][0]) if prediction[0][0] < 0.5 else "Invalid" + " " + str(prediction[0][0])

print("1- " + predict_signature('test_signatures/test_signature.jpg')) #Authentic
print("2- " + predict_signature('test_signatures/test_signature2.jpg')) #Forged
print("3- " + predict_signature('test_signatures/test_signature3.jpg')) #Forged
print("4- " + predict_signature('test_signatures/test_signature4.jpg')) #Authentic
print("5- " + predict_signature('test_signatures/test_signature5.jpg')) #Forged
print("6- " + predict_signature('test_signatures/test_signature6.jpg')) #Forged
print("7- " + predict_signature('test_signatures/test_signature7.jpg')) #Forged
print("7- " + predict_signature('test_signatures/test_signature8.jpg')) #Authentic
# # model.save('signature_verification_model.h5')


################################################
#PART ONE: INFO EXTRACTION USING PYTESSERACT OCR
################################################

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def find_check_nb(text):
    lines = text.splitlines()
    for i in range(len(lines)):
        check_nb = lines[-1-i]
        if check_nb.strip(): #.strip() is to make sure the string is neither empty nor full of spaces
            return ''.join(c for c in check_nb if c.isdigit() or c == "O")
    return "No check number found"

def find_check_date(text):
    # Look for words containing exactly two slashes
    two_slash_pattern = r'\b\w*\/\w*\/\w*\b'
    two_slash_words = re.findall(two_slash_pattern, text) #using regex patterns
    
    if two_slash_words:
        return two_slash_words[0]
    
    # If no words with two slashes are found, look for words with one slash
    one_slash_pattern = r'\b\w*\/\w*\b'
    one_slash_words = re.findall(one_slash_pattern, text)
    
    if one_slash_words:
        return one_slash_words[0]
    else:
        return "No Date Found"

def find_check_payee(sentence):
    # Regex pattern for the Axxxx... Bxxxx...
    pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
    
    # Search for the first match in the sentence
    match = re.search(pattern, sentence)
    
    if match:
        return match.group(1), match.group(2)
    return "No Payee Found", ""

def preprocess_image(img):
    img = cv2.resize(img, (1280,720))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding or any other preprocessing steps
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("th",img)
    return thresh

def extract_number_from_sentence(sentence):
    # Mapping of number words to numerical equivalents
    number_words = {
        "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, 
        "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11, "TWELVE": 12, "THIRTEEN": 13, 
        "FOURTEEN": 14, "FIFTEEN": 15, "SIXTEEN": 16, "SEVENTEEN": 17, "EIGHTEEN": 18, 
        "NINETEEN": 19, "TWENTY": 20, "THIRTY": 30, "FORTY": 40, "FIFTY": 50, 
        "SIXTY": 60, "SEVENTY": 70, "EIGHTY": 80, "NINETY": 90, "HUNDRED": 100, 
        "THOUSAND": 1000
    }

    def words_to_number(words):
        """Convert a sequence of number words into an integer."""
        total = 0
        current = 0
        for word in words:
            word_upper = word.upper()
            if word_upper in number_words:
                value = number_words[word_upper]
                if value == 100 or value == 1000:  # Handle multipliers
                    current *= value
                else:
                    current += value
            elif current > 0:
                total += current
                current = 0
        total += current
        return total

    # Split the sentence into words
    words = sentence.split()

    # Filter words that match the number words
    filtered_words = [word for word in words if word.upper() in number_words]

    # Convert the filtered words into a number
    return words_to_number(filtered_words)

def find_check_amount(text):
    chars = "$£"
    lines = text.splitlines()
    numbers = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
 "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN",
 "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY",
 "HUNDRED", "THOUSAND"]

    # Loop through each line
    for i in range(len(lines)):
        line = lines[i]

        # Tokenize the line into words
        words = line.split()

        # Check each word or token in the line
        for word in words:
            if word.upper() in numbers:  # Check if the word matches
                return "$" + str(extract_number_from_sentence(line))
            elif any(c in word for c in chars) and len(word) > 4:  # Check for special chars
                return word

    return "Amount Not Found"

img = cv2.imread("Tests/cheque2.png")

processed = preprocess_image(img)
text = pytesseract.image_to_string(processed)

# print(text)
print("Check Number: " + find_check_nb(text))
print("Date: " + find_check_date(text))
p1, p2 = find_check_payee(text)
print("Payee: " + p1 + " " + p2)
print("Amount: " + find_check_amount(text))

###########################################################
#PART TWO: SIGNATURE DETECTION, EXTRACTION AND VERIFICATION
###########################################################

def extract_signature(source_image):

    # the parameters are used to remove small size connected pixels outliar 
    constant_parameter_1 = 84
    constant_parameter_2 = 250
    constant_parameter_3 = 100

    # the parameter is used to remove big size connected pixels outliar
    constant_parameter_4 = 18
    # read the input image
    img = source_image
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    # image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area/counter)
    # print("the_biggest_component: " + str(the_biggest_component))
    # print("average: " + str(average))

    # experimental-based ratio calculation, modify it for your cases
    # a4_small_size_outliar_constant is used as a threshold value to remove connected outliar connected pixels
    # are smaller than a4_small_size_outliar_constant for A4 size scanned documents
    a4_small_size_outliar_constant = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
    # print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))

    # experimental-based ratio calculation, modify it for your cases
    # a4_big_size_outliar_constant is used as a threshold value to remove outliar connected pixels
    # are bigger than a4_big_size_outliar_constant for A4 size scanned documents
    a4_big_size_outliar_constant = a4_small_size_outliar_constant*constant_parameter_4
    # print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))

    # remove the connected pixels are smaller than a4_small_size_outliar_constant
    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
    # remove the connected pixels are bigger than threshold a4_big_size_outliar_constant 
    # to get rid of undesired connected pixels such as table headers and etc.
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > (a4_big_size_outliar_constant)
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0
    # save the the pre-version which is the image is labelled with colors
    # as considering connected components
    plt.imsave('pre_version.png', pre_version)

    # read the pre-version
    img = cv2.imread('pre_version.png', 0)
    # ensure binary
    img = cv2.threshold(img, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # save the the result
    # cv2.imwrite("output.png", img)
    return img

def draw_signature(img, tried):

    #save the original image
    original = img
    org = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #perform signature extraction
    img = extract_signature(gray)
    height, width = img.shape[:2]

    # Step 3: Set the top-left quadrant to white
    img[:height // 2, :width // 2] = np.ones_like(img[:height // 2, :width // 2]) * 255

    # Step 4: Set the top-right quadrant to white
    img[:height // 2, width // 2:] = np.ones_like(img[:height // 2, width // 2:]) * 255

    # Step 5: Set the bottom-left quadrant to white
    img[height // 2:, :width // 2] = np.ones_like(img[height // 2:, :width // 2]) * 255
    cv2.imshow("Candidate Signatures", img)
    #find contours
    contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    max_bottom_right = (0, 0)
    lower_rightmost_contour = None
    for contour in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        if h < 10 or w < 10:
            continue
        bottom_right = (x , y)  # Bottom-right corner coordinates

        # Check if this contour is further right and lower
        if (bottom_right[0] > max_bottom_right[0] or bottom_right[1] > max_bottom_right[1]):
            max_bottom_right = bottom_right
            lower_rightmost_contour = contour

    x_lr, y_lr, w_lr, h_lr = cv2.boundingRect(lower_rightmost_contour)
    combined_contour = None
    if lower_rightmost_contour is None and tried is False:
        print("No Signature Found on 1st iteration")
    
    else:
        combined_contour = lower_rightmost_contour.copy()
    
        #iterate again over the contours to check for adjacent ones to constitute a larger part of the signature
        for contour in contours:
            if np.array_equal(contour, lower_rightmost_contour):
                continue  # Skip the lower rightmost contour itself

            # Get bounding rectangle of the current contour
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the current contour is within the proximity threshold
            if abs(x - (x_lr + w_lr)) < 15 or abs(y - (y_lr )) < 20:
            # Combine the contours
                combined_contour = np.vstack((combined_contour, contour))

    # Draw the bounding rectangle for total contour
    x, y, w, h = cv2.boundingRect(combined_contour)
    if combined_contour is not None and not(w < 10 or h < 10 or h > 100 or w > 250):
        cv2.putText(original,text='Signature',org=(x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX ,fontScale= 1,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.rectangle(original, (x, y), (x + w, y + h ), (0, 255, 0), 2)
    elif tried == False:
        #resize for better contour detection
        original_height, original_width = original.shape[:2]
        img = cv2.resize(original, (748,406))
        original, x, y, w, h = draw_signature(img, True)
        original = cv2.resize(original, (original_width, original_height))
        scale_w = original_width / 748
        scale_h = original_height / 406
        x = int(x * scale_w)
        y = int(y * scale_h)
        w = int(w * scale_w)
        h = int(h * scale_h)

    return original, x, y, w, h

def addMySignature(base_image, small_image, x, y, w, h):
    # Step 1: Clamp the ROI coordinates to the image boundaries
    y1 = max(0, y - 25)
    y2 = min(base_image.shape[0], y + h + 25)
    x1 = max(0, x - 25)
    x2 = min(base_image.shape[1], x + w + 25)

    # Step 2: Define the ROI
    roi = base_image[y1:y2, x1:x2]

    # Step 3: Resize the smaller image to match the ROI
    small_image_resized = cv2.resize(small_image, (x2 - x1, y2 - y1))

    # Step 4: Overlay the smaller image on the ROI
    if roi.shape[:2] == small_image_resized.shape[:2]:
        base_image[y1:y2, x1:x2] = small_image_resized
    else:
        print("Dimension mismatch between ROI and the smaller image.")

    return base_image

signature_drawn, x, y, w, h = draw_signature(img, False)
cv2.imshow('Detected Signature', signature_drawn)
cv2.imshow('Original Signature', img[y:y+h, x:x+w])
print("Signature: ", predict_signature(img[y:y+h, x:x+w]))

# # Step 1: Read the base image and the smaller image
# my_signature = cv2.imread('Tests/test_signature4.png')  # Replace with your smaller image path

# mysignature_added = addMySignature(img, my_signature, x, y, w, h)
# my_sig = mysignature_added.copy()
# cv2.imshow("Added My Signature", mysignature_added)


# mysignature_drawn, x, y, w, h = draw_signature(mysignature_added, False)
# cv2.imshow('Detected My Signature', mysignature_drawn)
# cv2.imshow("My Signature", my_sig[y:y+h, x:x+w])
# print("After I added my Signature: ", predict_signature(my_sig[y:y+h, x:x+w]))
cv2.waitKey(0)