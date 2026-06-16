import cv2
import numpy as np

capture = cv2.VideoCapture("movie.avi")

while True:
    ret, frame = capture.read()

    if not ret:
        break

    # Loading the image into the program
    img_max = frame
    # Converting the image to 320p
    bgr_img = cv2.resize(img_max, (320, 240))
    # Converting the image to HSV
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # Creating a black and white version of the same image
    img_grey = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Applying masks to the image showing only the pixels that would be a part of a blue sky, grey sky, dimly lit sky or the sunset / sunrise
    blue_mask = cv2.inRange(img, np.array([95, 25, 120]), np.array([130, 255, 255]))
    grey_mask = cv2.inRange(img, np.array([0, 0, 150]), np.array([180, 70, 255]))

    h, w = bgr_img.shape[:2]
    
    colour_mask = cv2.bitwise_or(blue_mask, grey_mask)

    # Applying the laplacian operation to it to find the edges in the image
    laplacian = cv2.Laplacian(img_grey, cv2.CV_32F, ksize=3)
    # Finding the absolute values of the laplacian results in the image as direction does not matter
    abs_texture = np.abs(laplacian)
    # Removing noise from the image with a Gaussian Blyr
    smooth_texture = cv2.GaussianBlur(abs_texture, (5,5), 0)
    # Normalising and then weighting the image so it can be used as a probability
    normalised_texture = smooth_texture / (smooth_texture.max() + 1e-6)


    # Performing the Canny Edge method
    grey_blur = cv2.GaussianBlur(img_grey, (3,3), 0)
    canny_edges = cv2.Canny(grey_blur, 30, 80)


    # Calculate the hough lines in the image, where the consistent lines are
    hough_lines = cv2.HoughLinesP(canny_edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=80, maxLineGap=10)
    # Create a basic horizon halfway up the image and make a copy of the image
    horizon = img_grey.shape[0] // 2
    hough_line_img = bgr_img.copy()
    horizon_found = False
    # If there is at least one line in the image
    if hough_lines is not None:
        horizontal = []
        # Iterate through every line in the image and add all that are close to horizontal
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10:
                horizontal.append((y1 + y2) // 2)
                #cv2.line(hough_line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Find the median of all horizontal lines in the image and set that to be the horizon
        if len(horizontal) > 3:
            horizon = int(np.median(horizontal))
            horizon_found = True

    horizon_weights = np.zeros((h, w), dtype=np.float32)

    if horizon_found:
        # Every pixel above the horizon gets a weight
        # Pixels further above the horizon get a higher weight
        for y in range(horizon):
            horizon_weights[y, :] = 1.0 - (y / horizon)
    else:
        for y in range(h):
            horizon_weights[y, :] = max(0, 1.0 - (y / (h * 0.6)))


    colour_gate = colour_mask / 255.0
    # Weighting the different sky detection methods
    if horizon_found:
        weighted_horizon = horizon_weights * 0.15
        weighted_texture = ((1.0 - normalised_texture) * 0.1) * colour_gate
        colour_weight = colour_gate * 0.4
        edge_weight = ((1.0 - (canny_edges / 255.0)) * 0.35) * colour_gate
    else:
        weighted_horizon = horizon_weights * 0.05
        weighted_texture = ((1.0 - normalised_texture) * 0.15) * colour_gate
        colour_weight = colour_gate * 0.4
        edge_weight = ((1.0 - (canny_edges / 255.0)) * 0.4) * colour_gate

    # Combining the weights together
    total_weight = colour_weight + weighted_texture + edge_weight + weighted_horizon
    # Displaying the final weighted image
    total_display = (total_weight * 255).astype(np.uint8)
    cv2.imshow("Weights", total_display)

    # Code to display each image
    #display_texture = (weighted_texture / 0.1 * 255).astype(np.uint8)
    #cv2.imshow("texture", display_texture)
    display_horizon = (horizon_weights * 255).astype(np.uint8)
    cv2.imshow("Position Weight", display_horizon)
    cv2.line(hough_line_img, (0, horizon), (img_grey.shape[1], horizon), (0, 0, 255), 1)
    cv2.imshow("Horizon line", hough_line_img)   
    #cv2.imshow("Canny edges", canny_edges)
    cv2.imshow("BGR img", bgr_img)
    #cv2.imshow("colour mask", colour_mask)


    # Converting the probabilistic map into black and white
    _, threshold_img = cv2.threshold(total_display, 140, 255, cv2.THRESH_BINARY)

    # Close small holes within sky regions 
    close_kernel = np.ones((2, 2), np.uint8)
    closed_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, close_kernel)

    # Erosion
    kernel = np.ones((2,2), np.uint8)
    eroded_img = cv2.erode(closed_img, kernel, iterations=1)

    # Dilation
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=2)
    #Tcv2.imshow("Morphological operations", dilated_img)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_img)
    clean_mask = np.zeros((h, w), np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        centroid_y = centroids[label][1]
        # Filter out blobs that are too small
        if area < 150:
            continue
        # Filter out blobs whose centre is in the lower half of the image
        if centroid_y > h * 0.6:
            continue
        # Blob passed both checks, add it to the clean mask
        clean_mask[labels == label] = 255

    cv2.imshow("Sky Detection", clean_mask)

    prob_colour = cv2.applyColorMap(total_display, colormap=cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_img, 0.6, prob_colour, 0.4, 0)
    cv2.imshow("Sky Probability", overlay)

    # Press q to quit early
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()