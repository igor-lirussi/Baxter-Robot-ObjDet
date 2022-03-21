import time
import rospy
import baxter
import cv2
import numpy as np

PI = 3.141592
WIDTH = 960
HEIGHT = 600
CASC_PATH = "./models/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASC_PATH)

rospy.init_node("testing")
rospy.sleep(2.0)
robot = baxter.BaxterRobot(rate=100, arm="left")
rospy.sleep(2.0)
robot._set_camera(camera_name="left_hand_camera", state=True, width=WIDTH, height=HEIGHT, fps=30)

robot.set_robot_state(True)
print(robot.move_to_neutral())
print(robot.move_to_zero())
print(robot.move_to_joint_position({"left_s0": -PI/4}, timeout=10))
data = np.array(list(robot._cam_image.data), dtype=np.uint8)
middle_point = np.array([WIDTH/2, HEIGHT/2])

while not rospy.is_shutdown():
    img = np.array(list(robot._cam_image.data), dtype=np.uint8)
    img = img.reshape(int(HEIGHT), int(WIDTH), 4)
    img = img[:, :, :3].copy()

    #cv2.imwrite("./capture.png",img)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(faces)
    if len(faces) > 0:
        x, y, w, h = faces[0]  #get first face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        robot._set_display_data(img)
        current_loc = np.array([x, y])
        direction = current_loc - middle_point
        direction = direction / np.array([WIDTH/2, HEIGHT/2])
        robot.set_joint_velocity({"left_s0": -direction[0]/2, "left_s1": direction[1]/2})
    else:
        robot._set_display_data(img)
    robot.rate.sleep()

print(robot.move_to_neutral())
robot.set_robot_state(False)
