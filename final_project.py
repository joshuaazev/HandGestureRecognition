import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from matplotlib import pyplot as plt
import sys


def resize_image(image):
    size_max = 100.0
    height, width = image.shape[:2]
    if height > width:
        ratio = height / size_max
        dim = (int(width / ratio), int(size_max))
    else:
        ratio = width / size_max
        dim = (int(size_max), int(height / ratio))

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def open_image(image):
    img = cv2.imread(image)
    return resize_image(img)


def segment_hand2(img):

    ret, bw_img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)

    return bw_img



def boundary_contours(img):
    rows, cols = img.shape
    left, right, top, bottom = cols-1, 0, rows-1, 0
    
    # scan top to bottom, left to right
    for row in range(rows):
        for col in range(cols):
            if img[row, col] == 255:
                left = min(left, col)
                right = max(right, col)
                top = min(top, row)
                break
                
    # scan top to bottom, right to left
    for row in range(rows):
        for col in range(cols-1, -1, -1):
            if img[row, col] == 255:
                right = max(right, col)
                left = min(left, col)
                top = min(top, row)
                break
                
    # scan bottom to top
    for row in range(rows-1, -1, -1):
        for col in range(cols):
            if img[row, col] == 255:
                bottom = max(bottom, row)
                break
                
    return [left, top, right, bottom]

def orientation_detection(bbox, binary_image):
    height, width = binary_image.shape
    length, width_ = bbox[3]-bbox[1], bbox[2]-bbox[0]
    ratio = length/width_
    orientation = None
    
    # Check orientation based on the length-to-width ratio of the bounding box
    if ratio > 1.15:
        orientation = "vertical"
    else:
        orientation = "horizontal"
        
    # Check orientation based on the boundary tracing of the hand in binary image
    x_boundary = [i for i in range(width_) if binary_image[bbox[1]:bbox[3], bbox[0]+i].any()]
    y_boundary = [i for i in range(length) if binary_image[bbox[1]+i, bbox[0]:bbox[2]].any()]
    if x_boundary == [0] and max(y_boundary) == height-1:
        if orientation == "vertical":
            return orientation
        else:
            return "horizontal"
    elif max(x_boundary) == width-1 and y_boundary == [0]:
        if orientation == "horizontal":
            return orientation
        else:
            return "vertical"
    else:
        return orientation

def calculate_centroid(binary_image, bbox):
    M = cv2.moments(binary_image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
    if M["m00"] == 0:
        return None
    x = int(M["m10"]/M["m00"]) + bbox[0]
    y = int(M["m01"]/M["m00"]) + bbox[1]
    return (x, y)


def draw_circle(image, center, radius = 3, color=(0, 0, 255), thickness=1):
    image = cv2.circle(image, center, radius, color, thickness)
    return image

def draw_rectangle(image, bbox, color = (0,0,255)): #BBox in this order: left top right bottom
    return cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)


def detect_thumb(binary_img, bbox, orientation):
    x1, y1, x2, y2 = bbox
    pixel_width = 12 #Width of each green and blue rectangles
    if orientation == "vertical":
        # Crop the bounding box into two regions
        left_box = binary_img[y1:y2, x1:x1 + pixel_width]
        right_box = binary_img[y1:y2, x2 - pixel_width:x2]
        total_white_pixels = np.sum(binary_img == 255)
        left_box_white_pixels = np.sum(left_box == 255)
        right_box_white_pixels = np.sum(right_box == 255)


        left_box_ratio = left_box_white_pixels / total_white_pixels
        right_box_ratio = right_box_white_pixels / total_white_pixels
        # print(left_box_ratio)
        # print(right_box_ratio)
        thumb_detected = False
        if left_box_ratio < 0.07:
            return "left"
        elif right_box_ratio < 0.07:
            return "right"
        else:
            return "no thumb"
    elif orientation == "horizontal":
        # Crop the bounding box into two regions
        top_box = binary_img[y1:y1 + 30, x1:x2]
        bottom_box = binary_img[y2 - 30:y2, x1:x2]
        total_white_pixels = np.sum(binary_img == 255)
        top_box_white_pixels = np.sum(top_box == 255)
        bottom_box_white_pixels = np.sum(bottom_box == 255)
        top_box_ratio = top_box_white_pixels / total_white_pixels
        bottom_box_ratio = bottom_box_white_pixels / total_white_pixels
        thumb_detected = False
        if top_box_ratio < 0.07:
            return "top"
        elif bottom_box_ratio < 0.07:
            return "bottom"
        else:
            return "no thumb"



def highest_white_pixel(binary_image):
    result = []
    for col in range(binary_image.shape[1]):
        highest_row = binary_image.shape[0]
        for row in range(binary_image.shape[0]):
            if binary_image[row][col] != 0 and row < highest_row:
                highest_row = row
        result.append(highest_row)
    return result


def rightmost_white_pixel(binary_image):
    result = []
    for row in range(binary_image.shape[0]):
        rightmost_col = 0
        for col in range(binary_image.shape[1]):
            if binary_image[row][col] != 0:
                rightmost_col = col
        result.append(rightmost_col)
    return result


def get_peaks(binary_img, orientation):
    # Trace the boundary matrices of the hand
    y_coordinates = highest_white_pixel(binary_img)
    x_coordinates = rightmost_white_pixel(binary_img)
    # Initialize peaks array
    peaks = []
    
    # Check orientation and trace coordinates accordingly
    if orientation == "vertical":
        start = False
        for i in range(1, len(y_coordinates)):
            if y_coordinates[i] < y_coordinates[i-1] and not start:
                start = True
            elif y_coordinates[i] > y_coordinates[i-1] and start:
                peaks.append((i,y_coordinates[i]))
                start = False
    elif orientation == "horizontal":
        start = False
        for i in range(1, len(x_coordinates)):
            if x_coordinates[i] > x_coordinates[i-1] and not start:
                start = True
            elif x_coordinates[i] < x_coordinates[i-1] and start:
                peaks.append((x_coordinates[i], i))
                start = False
    else:
        raise ValueError("Invalid orientation value. Use 'vertical' or 'horizontal'.")
    
    return peaks


def euclidean_distance_to_centroid(peaks, centroid):
    distances = []
    for peak in peaks:
        x1, y1 = peak
        x2, y2 = centroid
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        distances.append(distance)
    return distances

def find_significant_peaks(peaks, centroid):
    distances = euclidean_distance_to_centroid(peaks, centroid)
    max_distance = max(distances)
    threshold = max_distance * 0.69
    significant_peaks = [peak for peak, distance in zip(peaks, distances) if distance >= threshold]
    return significant_peaks


def print_circles(img, list, orientation, bbox):
    x1, y1, x2, y2 = bbox
    if orientation == 'vertical':
        for i in range (x1, x2+1):
            img = draw_circle(img, (i, list[i]))
    elif orientation == 'horizontal':
        for i in range (y1, y2+1):
            img = draw_circle(img, (list[i], i))
    return img

def draw_boundary_points(img, binary_ig, orientation, bbox):
    if orientation == 'vertical':
        list = highest_white_pixel(binary_ig)
    elif orientation == 'horizontal':
        list = rightmost_white_pixel(binary_ig)
    return print_circles(img, list, orientation, bbox)

def draw_peak_points(img, list):
    for i in list:
        img = cv2.circle(img, i, 3, (0,0,255), 1)
    return img

def draw_significant_peak_points(img, list, list2, centroid):
    for i in list2:
        if i in list: #If is a significant point, draw a green circle
            img = cv2.circle(img, i, 3, (0,255,0), 1)
        else: #Draw a red circle
            img = cv2.circle(img, i, 3, (0,0,255), 1)
    img = cv2.circle(img, centroid, 3, (255,0,0), 1)
    return img

def draw_green_blue_rectangles(img, orientation, bbox):
    pixel_width = 12
    x1, y1, x2, y2 = bbox
    if orientation == 'vertical':
        img = draw_rectangle(img, [x1, y1, x1+pixel_width, y2], (0,255,0)) #Drawing the green box in the left
        img = draw_rectangle(img, [x2-pixel_width, y1, x2, y2], (255,0,0)) #Drawing the blue box in the right
    elif orientation == 'horizontal':
        img = draw_rectangle(img, [x1, y1, x2, y1+pixel_width], (0,255,0)) #Drawing the green box in the top
        img = draw_rectangle(img, [x1, y2 - pixel_width, x2, y2], (255,0,0)) #Drawing the blue box in the bottom

    return img

def define_fingers_raised(orientation, thumb_detection, list_peaks, list_s_peaks):
    len_peaks = len(list_peaks)
    len_s_peaks =  len(list_s_peaks)
    string = 'Orientation is: '+ orientation + '\nThumb position: '+  thumb_detection + '\n'
    if orientation == 'vertical' or orientation == 'horizontal':
        if thumb_detection == 'no thumb':
            for i in range(len_peaks):
                if list_peaks[i] in list_s_peaks and i == 0:
                    string += 'Index Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 1:
                    string += 'Middle Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 2:
                    string +='Ring Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 3:
                    string += 'Pinkie is raised\n'
        elif thumb_detection == 'right':
            for i in range(len_peaks):
                if list_peaks[i] in list_s_peaks and i == 3:
                    string += 'Index Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 2:
                    string += 'Middle Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 1:
                    string +='Ring Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 0:
                    string += 'Pinkie is raised\n'
        elif thumb_detection == 'left':
            for i in range(len_peaks):
                if list_peaks[i] in list_s_peaks and i == 1:
                    string += 'Index Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 2:
                    string += 'Middle Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 3:
                    string +='Ring Finger is raised\n'
                elif list_peaks[i] in list_s_peaks and i == 4:
                    string += 'Pinkie is raised\n'
    return string
                

        


def classify_hand_image(image):
    original_img = cv2.imread(image) # Reading original image
    original_img = open_image(image)

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) # Converting the original image to grayscale
    

    binary_img = 255 - segment_hand2(gray_img) # Converting the image to binary then inverting it to make the hand white
    # binary_img = 255 - segment_hand3(original_img) # Converting the image to binary then inverting it to make the hand white

    bbox = boundary_contours(binary_img) # Calculating the boundary box that contains the hand
    image_with_bbox = cv2.imread(image)
    image_with_bbox = open_image(image)
    image_with_bbox = draw_rectangle(image_with_bbox, bbox) # Drawing the rectangle in the original image

    orientation = orientation_detection(bbox, binary_img) #Returns the orientation of the hand (Vertical or Horizontal)

    centroid = calculate_centroid(binary_img, bbox)
    image_with_centroid = cv2.imread(image)
    image_with_centroid = open_image(image)
    image_with_centroid = draw_circle(image_with_centroid, centroid) #Drawing the centroid in the original image

    thumb_detection = detect_thumb(binary_img, bbox, orientation) #Detecting where is the thumb
    image_with_boxes = cv2.imread(image)
    image_with_boxes = open_image(image)
    image_with_boxes = draw_green_blue_rectangles(image_with_boxes, orientation, bbox)

    image_with_boundary_points = cv2.imread(image)
    image_with_boundary_points = open_image(image)
    image_with_boundary_points = draw_boundary_points(image_with_boundary_points, binary_img, orientation, bbox) # Drawing original image with points

    list_of_peaks = get_peaks(binary_img, orientation)
    image_with_peaks = cv2.imread(image)
    image_with_peaks = open_image(image)
    image_with_peaks = draw_peak_points(image_with_peaks, list_of_peaks)

    if list_of_peaks:
        significant_peaks = find_significant_peaks(list_of_peaks, centroid)
        image_with_significant_peaks = cv2.imread(image)
        image_with_significant_peaks = open_image(image)
        image_with_significant_peaks = draw_significant_peak_points(image_with_significant_peaks, significant_peaks, list_of_peaks, centroid)

    string = define_fingers_raised(orientation, thumb_detection, list_of_peaks, significant_peaks)

    #All the prints of the images of the process
    # cv2.imshow("original_img", original_img)
    # # cv2.imwrite("original_img2.jpg", original_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("gray_img", gray_img)
    # # cv2.imwrite("gray_img2.jpg", gray_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("binary_img", binary_img)
    # # cv2.imwrite("binary_img2.jpg", binary_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("image_with_bbox", image_with_bbox)
    # # cv2.imwrite("image_with_bbox2.jpg", image_with_bbox)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("image_with_centroid", image_with_centroid)
    # # cv2.imwrite("image_with_centroid2.jpg", image_with_centroid)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("image_with_boxes", image_with_boxes)
    # # cv2.imwrite("image_with_boxes2.jpg", image_with_boxes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("image_with_boundary_points", image_with_boundary_points)
    # # cv2.imwrite("image_with_boundary_points2.jpg", image_with_boundary_points)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("image_with_peaks", image_with_peaks)
    # # cv2.imwrite("image_with_peaks2.jpg", image_with_peaks)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("image_with_significant_peaks", image_with_significant_peaks)
    # # cv2.imwrite("image_with_significant_peaks2.jpg", image_with_significant_peaks)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB)
    image_with_significant_peaks = cv2.cvtColor(image_with_significant_peaks, cv2.COLOR_BGR2RGB)

    

    rows= 2
    columns =2
    fig = plt.figure(figsize=(10,7))
    plt.text(0.5, 1, image, ha='center', va='center', fontsize=20)
    plt.axis('off')
        # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    
    # showing image
    plt.imshow(original_img)
    plt.axis('off')
    plt.title("Original Image")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    
    # showing image
    plt.imshow(binary_img)
    plt.axis('off')
    plt.title("Binary")
    
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    
    
    # showing image
    plt.imshow(image_with_significant_peaks)
    plt.axis('off')
    plt.title("Image with significant peaks")
    
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
    
    # showing image
    plt.text(0.5, 0.5, string, ha='center', va='center', fontsize=20)
    plt.axis('off')
    plt.title("Results")
    plt.show()



for i in range (0,9):
    classify_hand_image("image_test{}.jpg".format(i))

