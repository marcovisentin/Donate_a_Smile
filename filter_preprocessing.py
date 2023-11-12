import cv2
import numpy as np

def preprocess_filter():
    filter = cv2.imread('dog_filter.png')
    filter_h, filter_w, filter_channles = filter.shape

    # Convert to greyscale 
    filter_gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)

    # create mask and inverse 
    ret, original_mask = cv2.threshold(filter_gray, 40,255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)