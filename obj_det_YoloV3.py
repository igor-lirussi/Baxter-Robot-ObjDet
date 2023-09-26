import time
import rospy
import importlib
baxter=importlib.import_module("baxter-python3.baxter")
import cv2
import numpy as np
import argparse

PI = 3.141592
WIDTH = 960
HEIGHT = 600

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='yolov3', help='Model desired')
args = parser.parse_args()
model_list = ["yolov3","yolov3-openimages","yolov3-openimages-spp"]

print("[INFO] loading model...")
if args.model == model_list[0]:
    #Load net
    modelConfig  = "./models/yolov3.cfg"
    modelWeigths = "./models/yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeigths)
    print("Net Loaded: {}".format(args.model))

    with open('./models/coco.names', 'r') as f:
        classes = f.read().splitlines()
    print("Classes: {}".format(len(classes)))

    #suggested
    conf_threshold = 0.1 #confidence threshold
    nms_threshold = 0.40 

elif args.model == model_list[1]:
    #Load net
    modelConfig  = "./models/yolov3-openimages.cfg"
    modelWeigths = "./models/yolov3-openimages.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeigths)
    print("Net Loaded: {}".format(args.model))

    with open('./models/open_images_yolo.names', 'r') as f:
        classes = f.read().splitlines()
    print("Classes: {}".format(len(classes)))

    #suggested
    conf_threshold = 0.1 #confidence threshold
    nms_threshold = 0.40 

elif args.model == model_list[2]:
    #Load net
    modelConfig  = "./models/yolov3-openimages-spp.cfg"
    modelWeigths = "./models/yolov3-openimages-spp.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeigths)
    print("Net Loaded: {}".format(args.model))

    with open('./models/open_images.names', 'r') as f:
        classes = f.read().splitlines()
    print("Classes: {}".format(len(classes)))

    #suggested
    conf_threshold = 0.1 #confidence threshold
    nms_threshold = 0.40 

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
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
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
    # create input blob 
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    # run inference through the network
    # and gather predictions from output layers
    start = time.time()
    outs = net.forward(get_output_layers(net))
    print('Prediction took {:.5f} seconds'.format(time.time() - start))
    
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

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

            
    robot._set_display_data(cv2.resize(img, (1024,600)))
    robot.rate.sleep()

print(robot.move_to_neutral())
robot.set_robot_state(False)
