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

msg = rospy.wait_for_message("/robot/limb/left/endpoint_state", EndpointState)
p = msg.pose.position
q = msg.pose.orientation
print("Position:")
print(p)
print("Orientation:")
print(q.z)

while not rospy.is_shutdown():
	#msg = rospy.wait_for_message("/robot/range/left_hand_range/state", Range)
	#string = "Dist: {:0.2f}".format(msg.range)
	string = "Dist: {}".format(robot._ir_range.range)
	print(string)
	img = np.full((600,1024,3), 39, np.uint8)
	cv2.putText(img, string, (512,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 3) 
	robot._set_display_data(cv2.resize(img, (1024,600)))


# while not rospy.is_shutdown():
#     robot.rate.sleep()

# print(robot.move_to_neutral())
# robot.set_robot_state(False)
