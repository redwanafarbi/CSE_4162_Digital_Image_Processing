import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input image
image = cv2.imread("Skull.tif", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(512, 512))

# Function to Decrease intensity level resolution by 1 bit
def decrease_resolution(image, number_of_bits):
    step = 255 / (2 ** number_of_bits - 1)
    height, width = image.shape
    decreased_image = image.copy()

    for r in range(height):
        for c in range(width):
            decreased_image[r, c] = round(image[r, c] / step) * step

    return decreased_image

# Decrease intensity level resolution by 1
decreased_image = image.copy()
plt.figure(figsize = (13, 8))

for k in range(1, 9):
    plt.subplot(2, 4, k)
    number_of_bits = 9 - k
    decreased_image = decrease_resolution(decreased_image, number_of_bits)
    plt.imshow(cv2.cvtColor(decreased_image, cv2.COLOR_BGR2RGB))
    plt.title(f"{number_of_bits}-Bits Image")

plt.show()

