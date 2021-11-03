import os
import time

import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Set path to test dataset
TEST_DATASET_PATH = "/medico2021test"

# Set path to output
MASK_OUTUT_PATH  = "/medico2021mask"

# Load Keras model
model = load_model("/submission/model.h5")

# Loop over each image of the test dataset
for image_name in os.listdir(TEST_DATASET_PATH):

    # Load the test image
    image_path = os.path.join(TEST_DATASET_PATH, image_name)
    image = load_img(image_path)
    image = img_to_array(image)

    # Save the original dimensions for resizing after prediction
    original_height, original_width, _ = image.shape

    # Prepare image for model input
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    # Start timer
    start_time = time.time()

    # Perform prediction
    mask_pred = model.predict(image)[0]
    
    # End timer
    end_time = time.time() - start_time

    # Resize prpedicted mask to dimensions of original image
    mask_pred = mask_pred > 0.5
    mask_pred = mask_pred.astype(np.float32)
    mask_pred = mask_pred * 255.0

    mask_pred = cv2.resize(mask_pred, (original_width, original_height))

    # Write mask to output directory
    cv2.imwrite(os.path.join(MASK_OUTUT_PATH, image_name), mask_pred)

    # Print results to stdout
    print("%s;%f" % (image_name, end_time))