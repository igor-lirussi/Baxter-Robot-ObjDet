import time
import rospy
import importlib
baxter=importlib.import_module("baxter-python3.baxter")
face=importlib.import_module("baxter-python3.faces")
import cv2
import numpy as np
import math
import argparse
from baxter_core_msgs.msg import EndpointState
# for audio play required playsound
from playsound import playsound
from whisper_mic_igor import WhisperMicIgor #requires a conda env with whisper-mic installed and opencv (opencv-python) and ros (rospkg) 
# https://github.com/mallorbc/whisper_mic/
# if you have problems with wrong libffi (since opencv and ros conflics) conda install libffi==3.3
# do not call your code "whisper.py" will create circular dependency
# if you have a lot of ALSA messages sudo nano /usr/share/alsa/alsa.conf (https://github.com/Uberi/speech_recognition/issues/526) (https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time)
# or set this handler to asound library
from ctypes import *
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
  #print('error received')
  pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so')
# Set error handler of asound libray to the previous one
asound.snd_lib_error_set_handler(c_error_handler)

# create mic module
mic = WhisperMicIgor(model="base",english=True,verbose=False,energy=900,pause=1.8,dynamic_energy=False,save_file=False, model_root="~/.cache/whisper",mic_index=None) #in the console whisper_mic --help for info on usage


PI = 3.141592
WIDTH = 960
HEIGHT = 600
DISPLAY_FACE=True
unreachable_count=0
garabbed=False

model_list = ["yolov4","yolov4-new","yolov4x-mish","yolov4-p6","yolov4-colors"]

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='yolov4-colors', help='Model desired among '+str(model_list))
parser.add_argument('-a', '--arm', type=str, default='left', help='Arm, left or right')
args = parser.parse_args()

OBJECT_DESIRED=""  
print("VOICE: hello")
playsound("./sounds/hello.mp3")

side = args.arm

print("[INFO] loading model...")
if args.model == "yolov4":
    #Load net
    modelConfig  = "./models/yolov4.cfg"
    modelWeigths = "./models/yolov4.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeigths)
    print("Net Loaded: {}".format(args.model))

    with open('./models/coco.names', 'r') as f:
        classes = f.read().splitlines()
    print("Classes: {}".format(len(classes)))

    conf_threshold = 0.1
    nms_threshold = 0.6 #lower=stronger

elif args.model == "yolov4-new":
    #Load net
    modelConfig  = "./models/yolov4_new.cfg"
    modelWeigths = "./models/yolov4_new.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeigths)
    print("Net Loaded: {}".format(args.model))

    with open('./models/coco.names', 'r') as f:
        classes = f.read().splitlines()
    print("Classes: {}".format(len(classes)))

    #suggested
    conf_threshold = 0.35
    nms_threshold = 0.03 #lower=stronger

elif args.model == "yolov4x-mish":
    #Load net
    modelConfig  = "./models/yolov4x-mish.cfg"
    modelWeigths = "./models/yolov4x-mish.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeigths)
    print("Net Loaded: {}".format(args.model))

    with open('./models/coco.names', 'r') as f:
        classes = f.read().splitlines()
    print("Classes: {}".format(len(classes)))

    #suggested
    conf_threshold = 0.35
    nms_threshold = 0.01 #lower=stronger

elif args.model == "yolov4-p6":
    #Load net
    modelConfig  = "./models/yolov4-p6-1280x1280.cfg"
    modelWeigths = "./models/yolov4-p6-1280x1280.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeigths)
    print("Net Loaded: {}".format(args.model))

    with open('./models/coco.names', 'r') as f:
        classes = f.read().splitlines()
    print("Classes: {}".format(len(classes)))

    #suggested
    conf_threshold = 0.35
    nms_threshold = 0.01 #lower=stronger

elif args.model == "yolov4-colors":
    #Load net
    modelConfig  = "./models/yolov4-colors.cfg"
    modelWeigths = "./models/yolov4-colors.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeigths)
    print("Net Loaded: {}".format(args.model))

    with open('./models/colors.names', 'r') as f:
        classes = f.read().splitlines()
    print("Classes: {}".format(len(classes)))

    #suggested
    conf_threshold = 0.35
    nms_threshold = 0.6 #lower=stronger

else:
    print("[Error] Model passed not present, choose between: {}".format(model_list))
    exit()


np.random.seed(42) #to generate the same colors
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
print("Colors generated: "+str(colors.shape[0]))

# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = []
    for i in net.getUnconnectedOutLayers():
        output_layers.append(layer_names[i-1])
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img_yolo, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    # Preparing colour for current bounding box
    color = [int(j) for j in colors[class_id]]
    cv2.rectangle(img_yolo, (x,y), (x_plus_w,y_plus_h), color, 2)
    text_box_current = '{}: {:.2f}'.format(label, confidence)
    if y<5:(x,y)=(x+15, y+30) #label position not out of the image
    cv2.putText(img_yolo, text_box_current, (x-6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2) 
    cv2.putText(img_yolo, text_box_current, (x-5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) 
    

print("[INFO] starting robot...")
np.random.seed()
rospy.init_node("testing"+str(np.random.randint(100))) #random node name so multiple can exist
rospy.sleep(2.0)
robot = baxter.BaxterRobot(rate=100, arm=side)
face._set_look(robot, side, DISPLAY_FACE)
rospy.sleep(2.0)
robot._set_camera(camera_name=side+"_hand_camera", state=True, width=WIDTH, height=HEIGHT, fps=30)
robot.set_robot_state(True)


rospy.sleep(2.0)
print("[INFO] calibrate gripper...")
robot.gripper_calibrate()
rospy.sleep(2.0)
robot.gripper_release()

#display face
face._set_look(robot, side+"_down", DISPLAY_FACE)


print("[INFO] moving in position...")
print(robot.move_to_neutral())
face._set_look(robot, side, DISPLAY_FACE)
print(robot.move_to_zero())
face._set_look(robot, "frontal", DISPLAY_FACE)
data = np.array(list(robot._cam_image.data), dtype=np.uint8)
middle_point = np.array([WIDTH/2, HEIGHT/2])

#move over the table
if side=="left":
    pos_x = 0.8203694373186249
    pos_y = 0.08642622598662506
else:
    pos_x = 0.7456267492841516
    pos_y = -0.18863639477015234
pos_z = 0.28462916699929078
ori_x = 0.011154239796145276
ori_y = 0.9989687054009745
ori_z = -0.006554586552752852
ori_w = 0.06499079561397379
face._set_look(robot, "down", DISPLAY_FACE)
robot.set_cartesian_position([pos_x, pos_y, pos_z], [ori_x, ori_y, ori_z, ori_w])

print("[INFO] getting image stream and passing to DNN...")
while not rospy.is_shutdown():

    if OBJECT_DESIRED=="":  
        print("VOICE: what_obj")
        playsound("./sounds/what_obj.mp3")
        print("Listen")
        result = mic.listen(phrase_time_limit=20)
        print("RESULT: "+result)
        for obj in classes:
            if obj in result:
                OBJECT_DESIRED = obj
                print("class found in string: "+obj)
                print("VOICE: obj_known")
                playsound("./sounds/obj_known.mp3")
    if OBJECT_DESIRED=="":
        print("no class found in string")
        print("VOICE: obj_not_known")
        playsound("./sounds/obj_not_known.mp3")

    img = np.array(list(robot._cam_image.data), dtype=np.uint8)
    img = img.reshape(int(HEIGHT), int(WIDTH), 4)
    img = img[:, :, :3].copy()

    #Passing image to DNN
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create input blob 
    blob = cv2.dnn.blobFromImage(img, 1/255, (640,640), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    # run inference through the network
    # and gather predictions from output layers
    start = time.time()
    outs = net.forward(get_output_layers(net))
    print('\nPrediction took {:.5f} seconds'.format(time.time() - start))
    
    # initialization
    class_ids = []
    confidences = []
    boxes = []

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < conf_threshold)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * WIDTH)
                center_y = int(detection[1] * HEIGHT)
                w = int(detection[2] * WIDTH)
                h = int(detection[3] * HEIGHT)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    #apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print("Detections: "+str(indices.shape[0])) if len(indices)!=0 else print("No Detections")

    #set gripper center, different from image center, should be in between the tips of the gripper, a little down. 
    gripper_delta_x=0
    gripper_delta_y=0
    if side=="left": #cameras in hands can be slightly diffenent angles, so center of gripper in the images is different
        gripper_delta_x=40
        gripper_delta_y=-60
    elif side=="right":
        gripper_delta_x=40
        gripper_delta_y=-100
    #center for the calculations
    center_object_x = round(WIDTH/2)+gripper_delta_x # usually the gripper center is a little on the right of the image of camera
    center_object_y = round(HEIGHT/2)+gripper_delta_y # usually the gripper is a litte up compared to the camera
    
    #object reset: not detected and in the center
    object_present=False
    objects_list=[]
    object_x=0
    object_y=0
    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        #save (X Y) object if present
        if classes[class_ids[i]] == OBJECT_DESIRED:
            object_present=True
            object_x = round(x+(w/2))
            object_y = round(y+(h/2))
            objects_list.append((object_x,object_y))
            print("{} found at: {} {}, size: {} {}".format(classes[class_ids[i]],object_x,object_y, w,h))
            cv2.putText(img, "X", (object_x,object_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 
            cv2.putText(img, "O", (center_object_x,center_object_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 3) 

    #object is selected as the closest one if there are many
    if len(objects_list) > 1:
        print("Many found, finding the closest")
        closest_distance = float('inf')  
        for x, y in objects_list:
            # Calculate Euclidean distance between the current point and the target point
            distance = math.sqrt((x - center_object_x) ** 2 + (y - center_object_y) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                object_x, object_y = (x, y)

    #put different color mark on object selected
    cv2.putText(img, "X", (object_x,object_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2) 
    
    #check infrared distance
    if robot._ir_range.range > robot._ir_range.min_range and robot._ir_range.range < robot._ir_range.max_range:
        current_range = robot._ir_range.range
        distance_str= "Dist: {:0.2f}".format(robot._ir_range.range)
        print(distance_str)
        cv2.putText(img, distance_str, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 
    else:
        current_range = 9999
        print("Range sensor out of limits")
        cv2.putText(img, "Dist: OUT", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 

    #display image
    robot._set_display_data(cv2.resize(img, (1024,600)))

    #if too close, grab
    if current_range < 0.16 and not garabbed:
        print("[info] Gripper CLOSE enough and object present, GRABBING without more movements")
        print("VOICE: Here we go!")
        playsound("./sounds/grasp_moment.mp3")
        OBJECT_DESIRED=""  
        face._set_face(robot, "determined", DISPLAY_FACE)
        #grab
        garabbed = True
        robot.gripper_grip()
        rospy.sleep(2.0)
        #move
        robot.set_cartesian_position([pos_x, pos_y, pos_z], [ori_x, ori_y, ori_z, ori_w])
        robot.move_to_zero()
        rospy.sleep(1.0)
        robot.gripper_release()
        garabbed=False
        rospy.sleep(2.0)
        robot.set_cartesian_position([pos_x, pos_y, pos_z], [ori_x, ori_y, ori_z, ori_w])

    #if present and not close enough: move towards it
    if object_present and not garabbed:
        #get current arm position
        msg = rospy.wait_for_message("/robot/limb/"+side+"/endpoint_state", EndpointState)
        p = msg.pose.position
        q = msg.pose.orientation
        #compute deviation in image
        delta_x_pixel=center_object_x - object_x
        delta_y_pixel=center_object_y - object_y
        print("DELTA PIXELS: {} and {}".format(delta_x_pixel, delta_y_pixel))
        #compute movement robot
        delta_x=0
        delta_y=0
        delta_z=0
        delta_movement=0.05
        if current_range < 0.25: #if close to something move less
            delta_movement = 0.02
        #if it's too on the side in X direction in the image move the robot on Y 
        if delta_x_pixel>40:
            delta_y = delta_movement
        elif delta_x_pixel<-40:
            delta_y = -delta_movement
        #if it's too on the side in Y direction in the image move the robot on X 
        if delta_y_pixel>40:
            delta_x = delta_movement
        elif delta_y_pixel<-40:
            delta_x = -delta_movement
        #if no horizontal movement the obj is centered, move down
        if delta_y==0 and delta_x ==0:
            delta_z = -delta_movement
        #move
        print("DELTA MOVEMENT X:{} Y:{} Z:{}".format(delta_x, delta_y, delta_z))
        movement_valid = robot.set_cartesian_position([p.x+delta_x, p.y+delta_y, p.z+delta_z], [q.x, q.y, q.z, q.w])
        if movement_valid:
            print("[info] Movement OK")
            unreachable_count=0
        elif not movement_valid and unreachable_count<4:
            unreachable_count=unreachable_count+1
            print("[info] Movement Unreachable count: {}".format(unreachable_count))
        elif not movement_valid and unreachable_count>3:
            print("VOICE: out_reach")
            playsound("./sounds/out_reach.mp3")
            face._set_face(robot, "worried", DISPLAY_FACE)
            #set to origin
            print("[info] Moving to Origin")
            robot.set_cartesian_position([pos_x, pos_y, pos_z], [ori_x, ori_y, ori_z, ori_w])





    #else:
    #if enough time passed look around

    #sleep
    robot.rate.sleep()

#out of the cycle
print(robot.move_to_neutral())
robot.set_robot_state(False)
