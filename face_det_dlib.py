import time
import rospy
import baxter
import cv2
import dlib  #requires pip install dlib
import numpy as np

PI = 3.141592
WIDTH = 960
HEIGHT = 600

print("[INFO] loading model...")
detector = dlib.get_frontal_face_detector()

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

    face_found_already = False
    #Passing image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1) 
    
    #to draw faces on image
    for result in faces:
        x1 = result.left()
        y1 = result.top()
        x2 = result.right()
        y2 = result.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #only for the first face detected
        if not face_found_already:
            face_found_already = True
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #move to that face
            current_loc = np.array([x1, y1])
            direction = current_loc - middle_point
            direction = direction / np.array([WIDTH/2, HEIGHT/2])
            robot.set_joint_velocity({"left_s0": -direction[0]/2, "left_s1": direction[1]/2})
        
    robot._set_display_data(img)
    robot.rate.sleep()

print(robot.move_to_neutral())
robot.set_robot_state(False)
