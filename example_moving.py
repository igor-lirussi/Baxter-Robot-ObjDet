import rospy
import baxter
rospy.init_node("example")
rospy.sleep(2.0)
robot = baxter.BaxterRobot(arm="left")
rospy.sleep(2.0)
robot.set_robot_state(True)
robot.move_to_neutral()
robot.move_to_zero()
robot.move_to_joint_position({"left_s0": 1.0})
robot.move_to_joint_position({"left_s0": -1.0})
robot.move_to_neutral()
robot.set_robot_state(False)

