from PIL import Image
import numpy as np
import os.path as osp
import os
import cv2
from matplotlib import pyplot as plt
# Load the image
image_path = '~/datasets/ISPRS/Vaihingen/img'
image_path = os.path.expanduser(image_path)
image = os.path.join(image_path, os.listdir(image_path)[14])
print(image)
image = Image.open(image)
image = np.array(image)
# Check if image has 3 channels (assuming it's in Infrared-Red-Green format)
if image.shape[2] == 3:
    # Split the channels
    ir, red, green = cv2.split(image)

    # Create a new image with the channels swapped to simulate natural colors
    # Swap infrared with red, and set red and green channels accordingly
    natural_green = green
    natural_red = red
    natural_blue = np.clip((green + red) // 2, 0, 255)  # Estimate blue using average of red and green

    # Merge back into a new BGR image for visualization
    natural_image = cv2.merge([natural_blue, natural_green, natural_red])

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Converted to Natural Colors")
    plt.show()
else:
    print("Error: The image does not have the expected three channels.")
