import numpy as  np
import cv2

#get the saved video file as stream
file_video_stream = cv2.VideoCapture(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\5. Object det, seg\Images\video (1).mp4")

#create a while loop 
while (file_video_stream.isOpened):
    #get the current frame from video stream
    ret,current_frame = file_video_stream.read()
    #use the video current frame instead of image
    img_to_detect = current_frame
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]
    
    # convert to blob to pass into model
    # swapRB=True means swap the red and blue channels in the image
    # crop=False means do not crop the image to fit the model input size
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)
    #recommended by yolo authors, scale factor is 0.003922=1/255, width,height of blob is 320,320
    #accepted sizes are 320×320,416×416,608×608. More size means more accuracy but less speed
    
    # set of 80 class labels 
    class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
    
    #Declare List of colors as an array
    #Green, Blue, Red, cyan, yellow, purple
    #Split based on ',' and for every split, change type to int
    #convert that to a numpy array to apply color mask to the image numpy array
    class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"] #list of colors
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors] #convert to numpy array
    class_colors = np.array(class_colors) 
    class_colors = np.tile(class_colors,(16,1)) #repeat the colors 16 times to get 80 colors for 80 classes
    
    # Loading pretrained model 
    # input preprocessed blob into model and pass through the model
    # obtain the detection predictions by the model using forward() method
    yolo_model = cv2.dnn.readNetFromDarknet(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\5. Object det, seg\yolov3\yolov3.cfg",r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\5. Object det, seg\yolov3\yolov3.weights")
    
    # Get all layers from the yolo network
    # Loop and find the last layer (output layer) of the yolo network 
    yolo_layers = yolo_model.getLayerNames() # get all layers from the yolo network
    # get the last layer (output layer) of the yolo network
    yolo_output_layer = [yolo_layers[yolo_layer - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()] 
    
    # input preprocessed blob into model and pass through the model
    yolo_model.setInput(img_blob)
    # obtain the detection layers by forwarding through till the output layer
    obj_detection_layers = yolo_model.forward(yolo_output_layer) # this will return the output of the last layer
    
    
    ############## NMS Change 1 ###############
    # initialization for non-max suppression (NMS) 
    ''' for multiple  bounding box in the same objects, we will only keep the one with the highest confidence score
'''
    # declare list for [class id], [box center, width & height[], [confidences]
    class_ids_list = [] # list of class ids
    boxes_list = [] # list of bounding boxes
    confidences_list = [] # list of confidences
    ############## NMS Change 1 END ###########
    
    
    # loop over each of the layer outputs
    for object_detection_layer in obj_detection_layers:
    	# loop over the detections
        for object_detection in object_detection_layer:
            
            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:] # index of the class max probability score will be stored
            predicted_class_id = np.argmax(all_scores) 
            prediction_confidence = all_scores[predicted_class_id]
        
            # take only predictions with confidence more than 50%
            if prediction_confidence > 0.50:
    
                #obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                
                
                ############## NMS Change 2 ###############
                #save class id, start x, y, width & height, confidences in a list for nms processing
                #make sure to pass confidence as float and width and height as integers
                class_ids_list.append(predicted_class_id) # get the class label
                confidences_list.append(float(prediction_confidence)) # get the confidence score
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)]) # get the bounding box co-ordinates
                ############## NMS Change 2 END ###########
    
    
    ############## NMS Change 3 ###############
    # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes      
    # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    '''NMSBoxes() function takes the following parameters:
    boxes_list: list of bounding boxes
    confidences_list: list of confidences
    confidence_threshold: confidence threshold to filter weak detections
    nms_threshold: threshold for non-maxima suppression we can say iou threshold, i.e. the intersection over union threshold
    (what it does is it takes the bounding boxes and suppresses the ones that have a high overlap with the one with the highest score)'''

    
    # loop through the final set of detections remaining after NMS and draw bounding box and write text
    for max_valueid in max_value_ids:
        max_class_id = max_valueid # get the index of the max value id
        box = boxes_list[max_class_id] # get the bounding box co-ordinates
        start_x_pt = box[0] # x co-ordinate of the top left corner of the bounding box
        start_y_pt = box[1] # y co-ordinate of the top left corner of the bounding box
        box_width = box[2] # width of the bounding box
        box_height = box[3] # height of the bounding box
        
        #get the predicted class id and label
        predicted_class_id = class_ids_list[max_class_id] # get the class label
        predicted_class_label = class_labels[predicted_class_id] # get the class label
        prediction_confidence = confidences_list[max_class_id] # get the confidence score
    ############## NMS Change 3 END ###########
    
        
        #obtain the bounding box end co-oridnates
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height
        
        #get a random mask color from the numpy array of colors
        box_color = class_colors[predicted_class_id]
        
        #convert the color numpy array as a list and apply to text and box
        box_color = [int(c) for c in box_color]
        
        # print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        print("predicted object {}".format(predicted_class_label))
        
        # draw rectangle and text in the image
        cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    
    cv2.imshow("Detection Output", img_to_detect)
    #terminate while loop if 'q' key is pressed
    # 0xFF is a bitwise AND operator to check if the key pressed is 'q'
    # ord is a function to get the unicode code of the character 'q' 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releasing the stream 
#close all opencv windows
file_video_stream.release()
cv2.destroyAllWindows()

'''
Steps in the code:
1. Import the necessary libraries
2. Load the video stream using cv2.VideoCapture() method
3. Read the current frame from the video stream using cv2.VideoCapture().read() method
4. Convert the current frame to a blob using cv2.dnn.blobFromImage() method
5. Pass the blob to the YOLOv3 model using cv2.dnn.blobFromImage() method
6. Get the output layer names of the YOLOv3 model using cv2.dnn.readNetFromDarknet() method
7. Forward the blob through the model to get the detection layers using cv2.dnn.forward() method
8. Loop through the detection layers and get the class id, bounding box coordinates, and confidence score for each detection
9. Apply non-maxima suppression (NMS) to remove duplicate detections using cv2.dnn.NMSBoxes() method
10. Draw the bounding boxes and labels on the image using cv2.rectangle() and cv2.putText() methods
11. Display the output image using cv2.imshow() method
12. Terminate the while loop if 'q' key is pressed using cv2.waitKey() method
13. Release the video stream using cv2.VideoCapture().release() method
14. Close all OpenCV windows using cv2.destroyAllWindows() method
'''