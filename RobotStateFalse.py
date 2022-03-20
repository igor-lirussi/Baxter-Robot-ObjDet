import rospy
import baxter

rospy.init_node("testing")
robot = baxter.BaxterRobot(rate=100, arm="left")


print(robot.move_to_neutral())
robot.set_robot_state(False)
