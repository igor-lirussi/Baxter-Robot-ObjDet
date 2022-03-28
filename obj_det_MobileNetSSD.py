import time
import rospy
import baxter
import cv2
import numpy as np

PI = 3.141592
WIDTH = 640
HEIGHT = 400

print("[INFO] loading model...")
PROTO = "./models/MobileNetSSD_deploy.prototxt"
MODEL = "./models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

RESIZED_DIMENSIONS = (300, 300) # Dimensions net was trained on. 
IMG_NORM_RATIO = 0.007843 # In grayscale a pixel can range between 0 and 255
 
#pascal voc classes
classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", 
           "diningtable",  "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]     

print("[INFO] starting robot...")
rospy.init_node("testing")
rospy.sleep(2.0)
robot = baxter.BaxterRobot(rate=100, arm="left")
rospy.sleep(2.0)
robot._set_camera(camera_name="left_hand_camera", state=True, width=WIDTH, height=HEIGHT, fps=30)
robot.set_robot_state(True)


print("[INFO] moving in position...")
print(robot.move_to_neutral())
print(robot.move_to_zero())
print(robot.move_to_joint_position({"left_s0": -PI/4}, timeout=10))
data = np.array(list(robot._cam_image.data), dtype=np.uint8)
middle_point = np.array([WIDTH/2, HEIGHT/2])

print("[INFO] getting image stream and passing to DNN...")
while not rospy.is_shutdown():
    img = np.array(list(robot._cam_image.data), dtype=np.uint8)
    img = img.reshape(int(HEIGHT), int(WIDTH), 4)
    img = img[:, :, :3].copy()

    #Passing image to DNN
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, RESIZED_DIMENSIONS), IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
    net.setInput(blob)
    neural_network_output = net.forward()
    print("Detections: " + str(neural_network_output.shape[2])) if len(neural_network_output)!=0 else print("No Detections")
    
    for i in np.arange(0, neural_network_output.shape[2]):
        confidence = neural_network_output[0, 0, i, 2]
        # Confidence must be at least x%       
        if confidence > 0.40:
            idx = int(neural_network_output[0, 0, i, 1])

            bounding_box = neural_network_output[0, 0, i, 3:7] * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])

            (startX, startY, endX, endY) = bounding_box.astype("int")

            label = "{}: {:.2f}%".format(classes[idx], confidence * 100) 

            cv2.rectangle(img, (startX, startY), (endX, endY), (255,0,0), 2)     

            y = startY - 15 if startY - 15 > 15 else startY + 15    

            cv2.putText(img, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        
    robot._set_display_data(img)
    robot.rate.sleep()

print(robot.move_to_neutral())
robot.set_robot_state(False)
