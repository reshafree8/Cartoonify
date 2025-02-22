# import cv2
# import numpy as np
# import os

# def cartoonify_image(image_path):
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError("Could not read the image.")

#     # Resize image for consistent processing (optional)
#     img = cv2.resize(img, (800, 800))

#     # Step 1: Convert to grayscale and apply median blur
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)

#     # Step 2: Detect edges using adaptive thresholding
#     edges = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY, 9, 7  # Reduce block size for finer edge detail
#     )

#     # Step 3: Apply bilateral filter for smooth color regions
#     # Keep smoothing subtle for more realism
#     color = cv2.bilateralFilter(img, 9, 150, 150)

#     # Step 4: Perform color quantization to simplify colors
#     Z = color.reshape((-1, 3))  # Reshape the image to a 2D array of pixels
#     K = 12  # Retain more colors for realism
#     _, labels, centers = cv2.kmeans(
#         Z.astype(np.float32), K, None,
#         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,
#         cv2.KMEANS_RANDOM_CENTERS
#     )
#     centers = np.uint8(centers)  # Convert back to uint8
#     quantized = centers[labels.flatten()].reshape(color.shape)

#     # Step 5: Adjust colors for realistic enhancement
#     hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
#     hsv[..., 1] = cv2.add(hsv[..., 1], 30)  # Moderate saturation boost
#     hsv[..., 2] = cv2.add(hsv[..., 2], 20)  # Slightly brighten image
#     enhanced_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#     # Step 6: Combine the cartoon edges with the enhanced color regions
#     cartoon = cv2.bitwise_and(enhanced_color, enhanced_color, mask=edges)

#     # Save the cartoonified image
#     cartoon_path = os.path.splitext(image_path)[0] + '_cartoon_realistic.jpg'
#     cv2.imwrite(cartoon_path, cartoon)
#     return cartoon_path

# import cv2
# import numpy as np
# import os

# def cartoonify_image(image_path):
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError("Could not read the image.")

#     # Resize image for consistent processing (optional)
#     img = cv2.resize(img, (800, 800))

#     # Step 1: Convert to grayscale and apply median blur
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)  # Slightly increased for better edge detection

#     # Step 2: Detect edges using adaptive thresholding
#     edges = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY, 9, 9
#     )

#     # Step 3: Apply bilateral filter for smooth color regions
#     # Further reduced details to give a painted/cartoon-like effect
#     color = cv2.bilateralFilter(img, 10, 300, 300)

#     # Step 4: Perform color quantization to simplify colors
#     Z = color.reshape((-1, 3))  # Reshape the image to a 2D array of pixels
#     K = 8  # Reduced number of colors for a stronger cartoonish effect
#     _, labels, centers = cv2.kmeans(
#         Z.astype(np.float32), K, None,
#         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,
#         cv2.KMEANS_RANDOM_CENTERS
#     )
#     centers = np.uint8(centers)  # Convert back to uint8
#     quantized = centers[labels.flatten()].reshape(color.shape)

#     # Step 5: Enhance colors for a cartoon-like vibrancy
#     hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
#     hsv[..., 1] = cv2.add(hsv[..., 1], 70)  # Boost saturation for vivid colors
#     hsv[..., 2] = cv2.add(hsv[..., 2], 40)  # Increase brightness
#     enhanced_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#     # Step 6: Combine the cartoon edges with the enhanced color regions
#     cartoon = cv2.bitwise_and(enhanced_color, enhanced_color, mask=edges)

#     # Save the cartoonified image
#     cartoon_path = os.path.splitext(image_path)[0] + '_cartoon_strong_effect.jpg'
#     cv2.imwrite(cartoon_path, cartoon)
#     return cartoon_path

# import cv2
# import numpy as np
# import os

# def cartoonify_image(input_path, output_path):
#     """
#     Cartoonify an image and save the result to the specified output path.

#     Parameters:
#     - input_path: Path to the input image.
#     - output_path: Path to save the cartoonified image.
#     """
#     # Read the image
#     img = cv2.imread(input_path)
#     if img is None:
#         raise ValueError("Could not read the image.")

#     # Resize image for consistent processing (optional)
#     img = cv2.resize(img, (800, 800))

#     # Step 1: Convert to grayscale and apply median blur
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 3)  # Reduced blurriness

#     # Step 2: Detect edges using adaptive thresholding
#     edges = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY, 9, 9
#     )

#     # Step 3: Apply bilateral filter for smoother regions
#     # Reduced blurring effect for sharper colors
#     color = cv2.bilateralFilter(img, 7, 200, 200)

#     # Step 4: Perform color quantization with vivid colors
#     Z = color.reshape((-1, 3))  # Reshape the image to a 2D array of pixels
#     K = 12  # Number of colors (higher retains more details)
#     _, labels, centers = cv2.kmeans(
#         Z.astype(np.float32), K, None,
#         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,
#         cv2.KMEANS_RANDOM_CENTERS
#     )
#     centers = np.uint8(centers)  # Convert back to uint8
#     quantized = centers[labels.flatten()].reshape(color.shape)

#     # Step 5: Enhance brightness and saturation
#     hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
#     hsv[..., 1] = cv2.add(hsv[..., 1], 50)  # Increase saturation
#     hsv[..., 2] = cv2.add(hsv[..., 2], 30)  # Increase brightness
#     enhanced_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#     # Step 6: Combine enhanced colors with edges
#     cartoon = cv2.bitwise_and(enhanced_color, enhanced_color, mask=edges)

#     # Save the cartoonified image
#     cv2.imwrite(output_path, cartoon)

import cv2
import numpy as np
import os

def cartoonify_image(input_path, output_path):
    """
    Cartoonify an image and save the result to the specified output path, 
    with enhancements for human faces.

    Parameters:
    - input_path: Path to the input image.
    - output_path: Path to save the cartoonified image.
    """
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not read the image.")

    # Resize image for consistent processing (optional)
    img = cv2.resize(img, (800, 800))

    # Step 1: Convert to grayscale and apply median blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # Increased blur for smoother results

    # Step 2: Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 10  # Slightly higher block size for sharper edges
    )

    # Step 3: Apply bilateral filter for smoother regions and more vivid colors
    # A stronger bilateral filter to create a smooth, cartoonish appearance
    color = cv2.bilateralFilter(img, 9, 300, 300)

    # Step 4: Perform color quantization with more vivid colors (higher K value)
    Z = color.reshape((-1, 3))  # Reshape the image to a 2D array of pixels
    K = 16  # Increased number of colors for more vivid effects
    _, labels, centers = cv2.kmeans(
        Z.astype(np.float32), K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,
        cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)  # Convert back to uint8
    quantized = centers[labels.flatten()].reshape(color.shape)

    # Step 5: Enhance brightness and saturation (to create a more vibrant look)
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.add(hsv[..., 1], 80)  # Increase saturation for a vivid look
    hsv[..., 2] = cv2.add(hsv[..., 2], 40)  # Increase brightness for a more energetic look
    enhanced_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Step 6: Combine enhanced colors with edges to give a cartoon effect
    cartoon = cv2.bitwise_and(enhanced_color, enhanced_color, mask=edges)

    # Save the cartoonified image
    cv2.imwrite(output_path, cartoon)
