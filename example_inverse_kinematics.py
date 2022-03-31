import time
import rospy
import baxter
import cv2
import numpy as np
from baxter_core_msgs.msg import EndpointState

rospy.init_node("testing")
rospy.sleep(2.0)
robot = baxter.BaxterRobot(rate=100, arm="left")
rospy.sleep(2.0)
# robot._set_camera(camera_name="left_hand_camera", state=True, width=WIDTH, height=HEIGHT, fps=30)

robot.set_robot_state(True)
msg = rospy.wait_for_message("/robot/limb/left/endpoint_state", EndpointState)
p = msg.pose.position
q = msg.pose.orientation
print("Position:")
print(p)
print("Orientation:")
print(q)

delta = 0.1
robot.set_cartesian_position([p.x-delta, p.y-delta, p.z], [q.x, q.y, q.z, q.w])
robot.set_cartesian_position([p.x+delta, p.y-delta, p.z], [q.x, q.y, q.z, q.w])
robot.set_cartesian_position([p.x+delta, p.y+delta, p.z], [q.x, q.y, q.z, q.w])
robot.set_cartesian_position([p.x-delta, p.y+delta, p.z], [q.x, q.y, q.z, q.w])
robot.set_cartesian_position([p.x-delta, p.y-delta, p.z], [q.x, q.y, q.z, q.w])


# while not rospy.is_shutdown():
#     robot.rate.sleep()

# print(robot.move_to_neutral())
# robot.set_robot_state(False)
