import time
from datetime import datetime
import os
import rospy
import baxter
import cv2
import numpy as np
from baxter_core_msgs.msg import DigitalIOState, EndpointState

directory="./imgs_saved"
interval_sec=3
LAST_PIC_TIME=time.time()
PI = 3.141592
WIDTH = 960
HEIGHT = 600

print("[INFO] Press circle button on the wrist to save image into ", directory)
print("date and time = ", datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
print("pictures min interval: ", interval_sec)
print("camera settings = ", (WIDTH, HEIGHT))

if not os.path.exists(directory):
	os.makedirs(directory)
	print("directory crated ", directory)


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

msg = rospy.wait_for_message("/robot/limb/left/endpoint_state", EndpointState)
p = msg.pose.position
q = msg.pose.orientation
print("Position:")
print(p)
print("Orientation:")
print(q)

while not rospy.is_shutdown():
	#get image from camera
	img = np.array(list(robot._cam_image.data), dtype=np.uint8)
	img = img.reshape(int(HEIGHT), int(WIDTH), 4)
	img = img[:, :, :3].copy()
	
	#calculate time passed from last picture
	time_passed=time.time()-LAST_PIC_TIME
	#check button pressed
	msg = rospy.wait_for_message("/robot/digital_io/left_lower_button/state", DigitalIOState)
	button_pressed = msg.state
	#if button pressed and time passed > interval (not to save image continuously)
	if button_pressed and time_passed > interval_sec:
		img_name= "capture-{}.png".format(datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
		print("Saving image: ", img_name)
		#save
		cv2.imwrite("{}/{}".format(directory,img_name),img)
		#set last pic time
		LAST_PIC_TIME=time.time()
	#if time passed from last pic < interval
	if time_passed < interval_sec:
		#for fading
		#normalize time passed in 0-255
		color_fading = int((1-(time_passed / interval_sec)) * 255)
		#set text on screen
		cv2.putText(img, "[image saved]", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,color_fading), 3)

	robot._set_display_data(cv2.resize(img, (1024,600)))


robot.set_robot_state(False)
