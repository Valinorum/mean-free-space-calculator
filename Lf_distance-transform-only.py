import cv2
import numpy as np
from scipy.ndimage import label
import os

def calculate_free_space_distance_transform(mask):
    """
    Calculates the free space length using the Distance Transform.
    This is a fast, deterministic, and accurate method.

    Args:
        mask (np.array): A binary mask where 1 represents the void space.

    Returns:
        float: The side length of the largest inscribed square in the void space.
    """
    # The mask is already the void space (1s), which is what distTransform requires
    # cv2.DIST_L2 specifies Euclidean distance, and 5 is the mask size for high accuracy.
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # The max value in the transform is the radius of the largest inscribed circle
    max_radius = np.max(dist_transform)
    
    # The side of the largest inscribed square in a circle of radius r is r * sqrt(2)
    # This is a known geometric property.
    side_length = max_radius * np.sqrt(2)
    
    return side_length

def analyze_image(image_path, threshold_value, invert=False):
    """
    Loads an image, applies a threshold, and calculates free space and coverage.

    Args:
        image_path (str): The full path to the grayscale image.
        threshold_value (int): The grayscale threshold value (0-255).
        invert (bool): If True, assumes material is lighter than the void.

    Returns:
        tuple: A tuple containing (mean_free_space_pixels, coverage_fraction).
               Returns (None, None) if the image cannot be processed.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path not found at '{image_path}'")
        return None, None

    # Load the image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Error: Could not read the image file at '{image_path}'")
        return None, None

    # 1. Create the binary image (mask) based on the threshold
    # Pixels >= threshold are considered void (white)
    binary_image = (original_image >= threshold_value).astype(np.uint8) * 255
    if invert:
        binary_image = cv2.bitwise_not(binary_image)

    # 2. Calculate Coverage Fraction
    # In the binary_image, material is black (0).
    material_pixels = np.sum(binary_image == 0)
    coverage_fraction = material_pixels / original_image.size

    # 3. Find the largest connected component (void space)
    # Convert to a 0/1 mask for analysis
    binary_mask = (binary_image / 255).astype(np.uint8)
    labeled_array, num_features = label(binary_mask)

    if num_features == 0:
        # No void space found at this threshold
        return 0, coverage_fraction

    # Isolate the largest component
    component_sizes = np.bincount(labeled_array.ravel())[1:]
    largest_component_label = component_sizes.argmax() + 1
    largest_component_mask = (labeled_array == largest_component_label).astype(np.uint8)

    # 4. Calculate Mean Free Space using Distance Transform
    mean_free_space_pixels = calculate_free_space_distance_transform(largest_component_mask)

    return mean_free_space_pixels, coverage_fraction

if __name__ == '__main__':
    # --- USER INPUTS ---
    # You can change these values to test different images and settings
    
    # IMPORTANT: Replace this with the actual path to your image
    # Example for Windows: "C:\\Users\\YourUser\\Documents\\image.bmp"
    # Example for Mac/Linux: "/home/youruser/images/image.bmp"
    IMAGE_PATH = "path/to/your/image.bmp" 
    
    THRESHOLD = 150
    
    # Set to True if the material you want to measure is lighter than the background
    INVERT_LOGIC = False 

    # --- ANALYSIS ---
    print(f"Analyzing Image: {os.path.basename(IMAGE_PATH)}")
    print(f"Using Threshold: {THRESHOLD}")
    print("-" * 30)

    free_space, coverage = analyze_image(IMAGE_PATH, THRESHOLD, INVERT_LOGIC)

    # --- RESULTS ---
    if free_space is not None and coverage is not None:
        print(f"Mean Free Space: {free_space:.2f} pixels")
        print(f"Coverage Fraction: {coverage:.4f}")
    else:
        print("Analysis could not be completed.")
