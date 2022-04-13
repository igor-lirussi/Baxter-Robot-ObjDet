import time
import rospy
import baxter
import cv2
import numpy as np
from baxter_core_msgs.msg import EndpointState
from sensor_msgs.msg import Range

rospy.init_node("testing")
rospy.sleep(2.0)
robot = baxter.BaxterRobot(rate=100, arm="left")
rospy.sleep(2.0)
robot.set_robot_state(True)

#You will have to calibrate both grippers before using any of the other commands 
#Check https://sdk.rethinkrobotics.com/wiki/API_Reference#Grippers_.28End-Effectors.29

string = "Calibrated: {} Ready: {} Moving: {} Gripping: {}".format(robot._gripper_state.calibrated, robot._gripper_state.ready, robot._gripper_state.moving, robot._gripper_state.gripping)
print(string)

robot.gripper_calibrate()
rospy.sleep(4.0)
string = "Calibrated: {} Ready: {} Moving: {} Gripping: {}".format(robot._gripper_state.calibrated, robot._gripper_state.ready, robot._gripper_state.moving, robot._gripper_state.gripping)
print(string)


robot.gripper_prepare_to_grip()
rospy.sleep(4.0)
string = "Calibrated: {} Ready: {} Moving: {} Gripping: {}".format(robot._gripper_state.calibrated, robot._gripper_state.ready, robot._gripper_state.moving, robot._gripper_state.gripping)
print(string)

robot.gripper_grip()
rospy.sleep(4.0)
string = "Calibrated: {} Ready: {} Moving: {} Gripping: {}".format(robot._gripper_state.calibrated, robot._gripper_state.ready, robot._gripper_state.moving, robot._gripper_state.gripping)
print(string)

robot.gripper_release()
rospy.sleep(4.0)
string = "Calibrated: {} Ready: {} Moving: {} Gripping: {}".format(robot._gripper_state.calibrated, robot._gripper_state.ready, robot._gripper_state.moving, robot._gripper_state.gripping)
print(string)

robot.gripper_reset()
rospy.sleep(2.0)
string = "Calibrated: {} Ready: {} Moving: {} Gripping: {}".format(robot._gripper_state.calibrated, robot._gripper_state.ready, robot._gripper_state.moving, robot._gripper_state.gripping)
print(string)

while not rospy.is_shutdown():
	string = "Calibrated: {} Ready: {} Moving: {} Gripping: {}".format(robot._gripper_state.calibrated, robot._gripper_state.ready, robot._gripper_state.moving, robot._gripper_state.gripping)
	#print(string)
	#print string on screen
	img = np.full((600,1024,3), 39, np.uint8)
	cv2.putText(img, string, (412,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 3) 
	robot._set_display_data(cv2.resize(img, (1024,600)))


# while not rospy.is_shutdown():
#     robot.rate.sleep()

# print(robot.move_to_neutral())
# robot.set_robot_state(False)
