
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
import numpy as np
import pytesseract
import re

# the parameters are used to remove small size connected pixels outliar 
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100

# the parameter is used to remove big size connected pixels outliar
constant_parameter_4 = 18

def extract_signature(source_image):

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
    print("the_biggest_component: " + str(the_biggest_component))
    print("average: " + str(average))

    # experimental-based ratio calculation, modify it for your cases
    # a4_small_size_outliar_constant is used as a threshold value to remove connected outliar connected pixels
    # are smaller than a4_small_size_outliar_constant for A4 size scanned documents
    a4_small_size_outliar_constant = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
    print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))

    # experimental-based ratio calculation, modify it for your cases
    # a4_big_size_outliar_constant is used as a threshold value to remove outliar connected pixels
    # are bigger than a4_big_size_outliar_constant for A4 size scanned documents
    a4_big_size_outliar_constant = a4_small_size_outliar_constant*constant_parameter_4
    print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))

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


    equalized = cv2.equalizeHist(image)

    # cv2.imshow('Equalized Image', equalized)

    # Apply Gaussian Blur
    denoised = cv2.medianBlur(equalized, 5)

    # cv2.imshow('Denoised with Median Blur', denoised)

    # Assume gray is anything between 1 and 254
    threshold_lower = 29
    threshold_upper = 254

    # Create a mask for gray pixels
    mask = cv2.inRange(denoised, threshold_lower, threshold_upper)

    # Set all pixels matching the mask to white
    denoised[mask > 0] = 255
    return denoised

def draw_signature(img, tried):

    #save the original image
    original = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #perform signature extraction
    img = extract_signature(gray)
    cv2.imshow("Extracted Signatures", img)

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
    if lower_rightmost_contour is None:
        print("No Signature Found")
    
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
        img = cv2.resize(original, (748,406))
        original = draw_signature(img, True)
    return original

def find_check_nb(text):
    lines = text.splitlines()
    for i in range(len(lines)):
        check_nb = lines[-1-i]
        if check_nb.strip():
            return ''.join(c for c in check_nb if c.isdigit() or c == "O")
    return "No check number found"

def find_check_date(text):
    # Look for words containing exactly two slashes
    two_slash_pattern = r'\b\w*\/\w*\/\w*\b'
    two_slash_words = re.findall(two_slash_pattern, text)
    
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
    # Regex pattern for the desired pair of words
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
    cv2.imshow("th",thresh)
    return thresh

def find_check_amount(text):
    chars = "$£"
    for i in text:
        if any(c in i for c in chars):
            return i
    return "Amount Not Found"

#MAIN
#SIGNATURE DETECTION
img = cv2.imread("Tests/cheque2.png")
signature_drawn = draw_signature(img, False)

cv2.imshow('Detected Signature', signature_drawn)

#DATA EXTRACTION
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
processed = preprocess_image(img)
text = pytesseract.image_to_string(processed)
print("\n\nOUTPUT")

print("Check Number: " + find_check_nb(text))
print("Date: " + find_check_date(text))
p1, p2 = find_check_payee(text)
print("Payee: " + p1 + " " + p2)
print("Amount: " + find_check_amount(text))
print(text)
cv2.waitKey(0)