import cv2
import matplotlib.pyplot as plt
import numpy as np

# Take image
image = cv2.imread("Skeleton.tif", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(512, 512))

# Generate Histogram
def generate_histogram(image, histogram_name):
    gray_levels_count = np.zeros(256)
    height, width = image.shape

    for r in range (height):
        for c in range(width):
            gray_levels_count[image[r, c]] += 1

    plt.bar(range(256), gray_levels_count, width = 1.0, color = "gray")
    plt.title(histogram_name)
    plt.show()

# SHow main image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("The Skeleton Image")
plt.show()

# Generate histogram
generate_histogram(image, "The Histogram of the Original Image")

# Make Single Threshold image
threshold_intensity = 27
segmented_image = np.where(image < threshold_intensity, 0, 255)
segmented_image = np.uint8(segmented_image)

plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title("The Segmented Image")
plt.show()

# Generate histogram of segmented image
generate_histogram(segmented_image, "The Histogram of the Segmented Image")