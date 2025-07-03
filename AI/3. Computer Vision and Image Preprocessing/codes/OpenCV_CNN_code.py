#pip install opencv-python
import cv2
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif')
cv2.imshow('Input image', img)
cv2.waitKey(0)

import cv2
gray_img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Grayscale', gray_img)
cv2.waitKey(0)

cv2.imwrite('sample_output.tif', gray_img)
import os
os.getcwd()

import cv2
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif')
cv2.imwrite('grey_output.png', img)

import cv2
print([x for x in dir(cv2) if x.startswith('COLOR_')]) # RGB, CMYK, YUV, HSV - Hue, Saturation,Value # cyan, magenta, yellow, and key, YUV, Y is the luminance (brightness) component while U and V are the chrominance (color) components

import cv2
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif', cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('Grayscale image', gray_img)
cv2.waitKey(0)
    
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(img)
cv2.imshow('R channel', y)
cv2.imshow('G channel', u)
cv2.imshow('B channel', v)
cv2.waitKey(0)

cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.waitKey(0)

img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif', cv2.IMREAD_COLOR)
g,b,r = cv2.split(img)
gbr_img = cv2.merge((g,b,r))
rbr_img = cv2.merge((r,b,r))
cv2.imshow('Original', img)
cv2.imshow('GRB', gbr_img)
cv2.imshow('RBR', rbr_img)
cv2.waitKey(0)



import cv2
import numpy as np
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif')
num_rows, num_cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 0.7)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey(0)



import cv2 #importing computer vision library
# laoding the img byusing cv2
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif')
cv2.imshow('Original Image', img) # showing the orgibnal image
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation =cv2.INTER_LINEAR) #resizing the img using linear interpolation
cv2.imshow('Scaling - Linear Interpolation', img_scaled)                          # showing the scaled linear interpolation image
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC) #resizing the img using cubic interpolation
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)                             # showing the scaled cubic interpolation image
img_scaled = cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)             # Resize the image to a fixed dimension (450x400) using area interpolation
cv2.imshow('Scaling - Skewed Size', img_scaled)                                   # Show the resized image with area interpolation
# Wait for a key press indefinitely and close all image windows when a key is pressed                               
cv2.waitKey(0)

#Affine Transformations Euclidean Transformation
import cv2 # importing opencv
import numpy as np # for numerical calculation
# laoding the img byusing cv2
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif')
rows, cols = img.shape[:2] # here, taking rows and column
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]]) # taking 3 points into consideration [[0,0] for origin, [cols-1,0] for last column in the forst row, [0,rows-1] first column in the last row]
dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0],[int(0.4*(cols-1)),rows-1]]) # defines three destination points that determine how the original image will be transformed
''' [0, 0] stays the same no shift, [int(0.6*(cols-1)), 0] Moves 40% left from the original position (cols-1, 0).'''
affine_matrix = cv2.getAffineTransform(src_points, dst_points) #  # Compute the affine transformation matrix
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows)) # Apply the affine transformation to the image
cv2.imshow('Input', img) # Display the original input image
cv2.imshow('Output', img_output) # Display the transformed output image
cv2.waitKey(0) # Wait indefinitely until a key is pressed, then close all windows


# Projective transformation
import cv2 # importing opencv
import numpy as np # for numerical calculation
# laoding the img byusing cv2
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/mandrill/mandrill.tif')
rows, cols = img.shape[:2] # here, taking rows and column
# Define four source points representing the corners of the original image
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])

# Define four destination points to apply the perspective transformation
# - The top-left and top-right corners remain unchanged.
# - The bottom-left corner moves right to 33% of the image width.
# - The bottom-right corner moves left to 66% of the image width.
dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1], [int(0.66*cols),rows-1]])

# Compute the perspective transformation matrix using the four source and destination points
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation to the image
img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows))

# Display the original input image
cv2.imshow('Input', img)

# Display the transformed output image with perspective warping
cv2.imshow('Output', img_output)

# Wait indefinitely until a key is pressed, then close all windows
cv2.waitKey(0)


#Blurring is called as low pass filter. Why?
import cv2
import numpy as np
# Load the image
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/plane/plane.bmp')
rows, cols = img.shape[:2] ## Get the number of rows and columns in the image

# Define the Identity kernel (3x3)
# This kernel does not change the image, as it retains the original pixel values.
kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]]) 
kernel_3x3 = np.ones((3,3), np.float32) / 9.0 # Divide by 9 to normalize the kernel
kernel_5x5 = np.ones((5,5), np.float32) / 25.0 # Divide by 25 to normalize the kernel
cv2.imshow('Original', img) # Display the original image
output = cv2.filter2D(img, -1, kernel_identity) # Apply the identity filter (no effect on the image)
cv2.imshow('Identity filter', output) 
output = cv2.filter2D(img, -1, kernel_3x3) # Apply the 3x3 averaging filter (smooths the image slightly)
cv2.imshow('3x3 filter', output)
output = cv2.filter2D(img, -1, kernel_5x5) # Apply the 5x5 averaging filter (produces a stronger smoothing effect)
cv2.imshow('5x5 filter', output)
cv2.waitKey(0) # Wait indefinitely for a key press before continuing
# Apply a built-in OpenCV blur function with a 3x3 kernel
# This is equivalent to the 3x3 averaging filter
output = cv2.blur(img, (3,3)) 
cv2.imshow('blur', output)
cv2.waitKey(0) # Wait indefinitely for a key press before closing all windows




import cv2
import numpy as np
# Load the image in grayscale mode
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/Median filter/Median filter.png', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape # Get the number of rows and columns in the image
# It is used to indicate depth of cv2.CV_64F.
# Apply the Sobel operator for edge detection in the horizontal direction
# - The second argument cv2.CV_64F ensures a high precision float64 output to handle negative values.
# - The third and fourth arguments (1,0) indicate differentiation in the x-direction (horizontal edges).
# - The kernel size (ksize=5) defines the size of the filter, affecting edge sensitivity.
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# Kernel size can be: 1,3,5 or 7.
# Apply the Sobel operator for edge detection in the vertical direction
# - The third and fourth arguments (0,1) indicate differentiation in the y-direction (vertical edges).
# - The kernel size (ksize=5) defines the size of the filter.
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
cv2.imshow('Original', img) # Display the original grayscale image
cv2.imshow('Sobel horizontal', sobel_horizontal) # Display the horizontally detected edges using the Sobel filter
cv2.imshow('Sobel vertical', sobel_vertical) # Display the vertically detected edges using the Sobel filter
cv2.waitKey(0) # Wait indefinitely for a key press before closing all windows



import cv2
import numpy as np
# Load the image in grayscale mode
img = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/plane/plane.bmp', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape # Get the number of rows and columns in the image
# Apply the Canny edge detection algorithm
# - canny edge detection detect the edges both x and y direction by using euclidean distance of sobel(x) and sobel(y)
# - The first threshold (50) is the lower bound for edge detection.
# - The second threshold (240) is the upper bound for strong edge detection.
canny = cv2.Canny(img, 50, 240) 
cv2.imshow('Canny', canny) # Display the resulting edge-detected image
cv2.waitKey(0)# Wait indefinitely for a key press before closing all windows



# Import the OpenCV library
import cv2

# Load the video file from the specified path
# cap = cv2.VideoCapture('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/test2/test2.mp4')
cap = cv2.VideoCapture(0) # primary source
# cap = cv2.VideoCapture(1) # secondary source
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam") # Raise an error if the video cannot be accessed

# Raise an error if the video cannot be accessed
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    # Resize the frame to 50% of its original size using area interpolation
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # Display the resized frame in a window titled 'Input'
    cv2.imshow('Input', frame)
    # Wait for a key press; if the 'Esc' key (ASCII 27) is pressed, exit the loop
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release() # Release the video capture object to free system resources
cv2.destroyAllWindows() # Close all OpenCV windows


# Video color spaces
import cv2
def print_howto():
    print("""
        Change color space of the input video stream using keyboard controls. The control keys are:
            1. Grayscale - press 'g'
            2. YUV - press 'y'
            3. HSV - press 'h'
    """)
if __name__=='__main__':
    # Print the instructions for the user
    print_howto()
    # Open a connection to the default webcam (device index 0)
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam") # Raise an error if the webcam cannot be accessed
    cur_mode = None
    while True:
        # Read the current frame from webcam
        ret, frame = cap.read()
        # Resize the captured image
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
        c = cv2.waitKey(1)
        # Exit the loop if the 'Esc' key (ASCII 27) is pressed
        if c == 27:
            break
        # Update cur_mode only in case it is different and key was pressed
        # In case a key was not pressed during the iteration result is -1 or 255, depending on library versions
        if c != -1 and c != 255 and c != cur_mode:
            cur_mode = c
        
        # Apply the selected color transformation based on the key pressed
        if cur_mode == ord('g'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif cur_mode == ord('y'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif cur_mode == ord('h'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            output = frame
        cv2.imshow('Webcam', output)
    
    # Release the webcam resource
    cap.release()
    cv2.destroyAllWindows() # Close all OpenCV windows


# Import necessary libraries
import cv2  # OpenCV for computer vision tasks
import numpy as np  # NumPy for numerical operations (not used in this code but commonly used with OpenCV)

''' 
Haarcascade is an object detection algorithm used to detect faces, eyes, and other objects in images.
It is based on machine learning and uses a pre-trained XML file containing features for object detection.
'''
# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/opencv_config_files/Day 5/haarcascade_frontalface_alt.xml')
# Open a connection to the default webcam (device index 0)
cap = cv2.VideoCapture(0)

# Define the scaling factor to resize the video frames for faster processing
scaling_factor = 0.5 
# Start the video stream loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    # Resize the frame to reduce computation and improve performance
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # Detect faces in the frame using the Haar Cascade classifier
    # scaleFactor: Specifies how much the image size is reduced at each image scale (1.3 means 30% reduction)
    # minNeighbors: Defines the minimum number of neighboring rectangles required for a region to be considered a face
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    
    # Loop through the detected faces and draw rectangles around them
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    # Display the processed video stream with detected faces
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1) # Exit the loop if 'Esc' is pressed
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()


# Import necessary libraries
import cv2  # OpenCV for computer vision tasks
import numpy as np  # NumPy for numerical operations
# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/opencv_config_files/Day 5/haarcascade_frontalface_alt.xml')

# Load the face mask image (to be applied over detected faces)
face_mask = cv2.imread('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/opencv_config_files/Day 5/plane.bmp')
# Get the height and width of the face mask image
h_mask, w_mask = face_mask.shape[:2]
# Check if the Haar cascade classifier loaded successfully
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
    # Open a connection to the default webcam (device index 0)
cap = cv2.VideoCapture(0) 
scaling_factor = 0.5# Define the scaling factor to resize the video frames for faster processing
# Start the video stream loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3,minNeighbors=3)
    for (x,y,w,h) in face_rects:
        if h <= 0 or w <= 0: pass
        # Adjust the height and weight parameters depending on the sizes and the locations.
        # You need to play around with these to make sure you get it right.
        h, w = int(1.0*h), int(1.0*w)
        y -= int(-0.2*h)
        x = int(x)
        # Extract the region of interest from the image
        frame_roi = frame[y:y+h, x:x+w]
        face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
        # Convert color image to grayscale and threshold it
        gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
        # Create an inverse mask
        mask_inv = cv2.bitwise_not(mask)
        try:
            
            # Use the mask to extract the face mask region of interest
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
            # Use the inverse mask to get the remaining part of the image
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        except cv2.error as e:
            print('Ignoring arithmentic exceptions: '+ str(e))
        # add the two images to get the final output
        frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()




# Eyeball detection
import cv2
import numpy as np
# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/opencv_config_files/Day 5/haarcascade_frontalface_alt.xml')
# Load the Haar cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/opencv_config_files/Day 5/haarcascade_eye.xml')

# Check if the cascade classifiers were loaded properly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')

# Open the webcam for video capture
cap = cv2.VideoCapture(0)
# Define a downscaling factor for resizing the frame
ds_factor = 0.5
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    # Resize the frame to improve processing speed
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,interpolation=cv2.INTER_AREA)
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=1)
    for (x,y,w,h) in faces:
        # Calculate the center and radius for the circle around the eye
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes within the detected face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Loop through detected eyes and draw circles around them
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye)) # Calculate the center of the eye
            radius = int(0.3 * (w_eye + h_eye)) # Define the radius for the circle
            color = (0, 255, 0) # Draw the circle
            thickness = 3
            # Draw a circle around the detected eyes
            cv2.circle(roi_color, center, radius, color, thickness)
        
        # Display the processed frame with detected eyes
        cv2.imshow('Eye Detector', frame)
        # Check for the 'Esc' key (ASCII 27) to exit the loop
        c = cv2.waitKey(1)
        if c == 27:
            break
cap.release()
cv2.destroyAllWindows()


# Loading a video 
# (Known issue with opencv_python package, which does not load the video.
# solution: pip3 uninstall opencv_python
#           pip3 install opencv_python --user
# - Restart kernel after package reinstallation    
import cv2
import numpy as np
cap = cv2.VideoCapture("C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/Datasets-20230301T104158Z-001/Datasets/test2/test2.mp4")
cap.isOpened()
ret, image = cap.read()
while ret:
    cv2.imshow('Video Stream', image)
    cv2.waitKey(20)
    ret, image = cap.read()    
cap.release()
cv2.destroyAllWindows()
print('Video Finished')


#Single Shot Object Detection Model
#(object detection algorithm which will able to detect multiple object, single shot means able to detect bounding boxes and objects same time and it used NMS for remove duplicate bounding boxes)
'''SSD (Single Shot MultiBox Detector) is a deep learning-based object detection algorithm. It is widely used for detecting multiple objects in images or videos in real-time.'''
import cv2
import numpy as np

# Load the pre-trained MobileNetSSD model from Caffe
model = cv2.dnn.readNetFromCaffe('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/opencv_config_files/Day 5/MobileNetSSD_deploy.prototxt',
                                 'C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/opencv_config_files/Day 5/MobileNetSSD_deploy.caffemodel')
'''
.caffemodel file is a pre-trained model file used in Caffe (Convolutional Architecture for Fast Feature Embedding), a deep learning framework. It contains the trained weights (parameters) of a neural network.
The .caffemodel file works alongside a .prototxt file, which defines the model architecture.'''
# Set the confidence threshold for detecting objects
CONF_THR = 0.3 # Objects with confidence > 30% will be considered
# Dictionary mapping class IDs to object labels
LABELS = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
          5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
          10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
          14: 'motorbike', 15: 'person', 16: 'pottedplant',
          17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Open the video file for processing
video = cv2.VideoCapture('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/AI/3. Computer Vision and Image Preprocessing/Data Sets_SDC/opencv_config_files/Day 5/traffic.mp4')
while True:
    # Read a frame from the video
    ret, frame = video.read()
    # If no frame is captured, break the loop (end of video)
    if not ret: break
     # Get frame dimensions (height and width)
    h, w = frame.shape[0:2]
    # Preprocess the frame and create a blob for the model
    # - Resize the image to (300, 300) while maintaining aspect ratio
    # - Normalize pixel values to [-1, 1] (by subtracting 127.5 and dividing by 127.5)
    blob = cv2.dnn.blobFromImage(frame, 1/127.5, (300*w//h,300),(127.5,127.5,127.5), False)
    # Set the input to the model
    model.setInput(blob)
    # Perform forward pass to get detections
    output = model.forward()
    # Iterate through all detected objects
    for i in range(output.shape[2]):
        conf = output[0,0,i,2]# Get the confidence score of the detected object
        # Process only detections with confidence above the threshold
        if conf > CONF_THR:
            label = output[0,0,i,1] # Get class ID
            x0,y0,x1,y1 = (output[0,0,i,3:7] * [w,h,w,h]).astype(int) # Get bounding box coordinates
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2) # Draw bounding box around detected object
            # Add label and confidence score text
            cv2.putText(frame, '{}: {:.2f}'.format(LABELS[label], conf), 
                        (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
    # Display the frame with detections
    cv2.imshow('frame', frame)
    # Wait for a key press (3ms), exit loop if 'Esc' key (27) is pressed
    key = cv2.waitKey(3)
    if key == 27: break
# Release video capture and close all OpenCV windows
cv2.destroyAllWindows() 



