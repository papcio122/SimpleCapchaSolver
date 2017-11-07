

# Standard imports
from PIL import Image
import pytesseract
import cv2
import imutils
import numpy as np
import os

def process_image(im):

    # Detect letters, rotate them and put each one in separate image
    letter_images = detect_letters(im)

    # Concate letters images into one image
    normalized_image = concat_images(letter_images)

    # Save image ready for OCR
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, normalized_image)

    # Read text on image
    image = Image.open(filename)
    text = pytesseract.image_to_string(image)

    # hotfix for some common mistakes
    text = str.replace(text, '4', 'Y')
    text = str.replace(text, '1', 'T')
    text = str.replace(text, '<', 'L')
    text = str.replace(text, '»', 'L')
    text = str.replace(text, ')', 'J')
    text = str.replace(text, '>', 'J')
    text = str.replace(text, '{}', ' ')
    text = str.replace(text, '/', ' ')
    text = str.replace(text, '.', ' ')

    save_name = 'results/result_' + text + '.bmp'
    print(save_name)
    image.save(save_name)
    return text

def detect_letters(im):
    # Reverse colors so there are white letters on black background
    im = cv2.bitwise_not(im)

    # Enlarge image
    im = cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Morphological opening in order to remove tiny connection between letters
    kernel = np.ones((3, 3), np.uint8)
    # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    im = cv2.erode(im, kernel, iterations=2)
    im = cv2.dilate(im, kernel, iterations=1)

    # Tresholding to be sure there are no connection between letters with weak color
    ret, thresh = cv2.threshold(im, 201, 255, cv2.THRESH_BINARY)

    # Detectibg contours(letters)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting contours left-to-right
    contours, boxes = sort_contours(contours)

    # Cut image into smaller ones containing only one letter
    letter_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rotation = cv2.minAreaRect(c)[2]
        if rotation < -30 or rotation > 30:
            rotation = normalizeRotation(rotation)

        subimg = im[y: y + h, x: x + w]

        dst = imutils.rotate_bound(subimg, -rotation)
        dst = add_borders(dst, (158, 158))

        letter_images.append(dst),

    return letter_images

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def normalizeRotation(rot, plus=True):
    if rot < -30 :
        return normalizeRotation(rot + 30, True)
    elif rot > 30 :
        return normalizeRotation(rot - 30, False)
    else :
        if plus :
            return rot + 30
        else :
            return rot - 30

def add_borders(img, new_size):
    x = 128 / img.shape[1]
    y = 128 / img.shape[0]
    img = cv2.resize(img, None, fx=x, fy=y, interpolation=cv2.INTER_CUBIC)

    top = bottom = int((new_size[0] - img.shape[0])/2)
    left = right = int((new_size[1] - img.shape[1])/2)
    dst = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    dst = cv2.resize(dst, new_size, interpolation = cv2.INTER_CUBIC)
    return dst

def concat_images(images):
    im_array = np.array(images)
    result = np.concatenate(im_array, 1)
    return result

file_list = os.listdir("captcha_images")
print(file_list)

for file in file_list:
    # Read image
    im = cv2.imread("captcha_images/" + file, cv2.IMREAD_GRAYSCALE)

    # Read text from image
    result = process_image(im)
cv2.waitKey(0)