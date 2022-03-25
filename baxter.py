from copy import deepcopy

import rospy
import cv_bridge
from baxter_core_msgs.msg import JointCommand, EndpointState, CameraSettings
from baxter_core_msgs.srv import OpenCamera, CloseCamera
from std_msgs.msg import Bool, Header
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState, Image


class BaxterRobot:

    def __init__(self, arm, rate=100):
        self.rate = rospy.Rate(100)

        self.name = arm
        self._joint_angle = {}
        self._joint_velocity = {}
        self._joint_effort = {}
        self._cartesian_pose = {}
        self._cartesian_velocity = {}
        self._cartesian_effort = {}
        self._joint_names = ["_s0", "_s1", "_e0", "_e1", "_w0", "_w1", "_w2"]
        self._joint_names = [arm+x for x in self._joint_names]
        ns = "/robot/limb/" + arm + "/"

        self._cam_image = Image()

        self._command_msg = JointCommand()

        self._robot_state = rospy.Publisher("/robot/set_super_enable", Bool, queue_size=10)
        self._pub_joint_cmd = rospy.Publisher(ns+"joint_command", JointCommand, queue_size=1)
        self._joint_state_sub = rospy.Subscriber("/robot/joint_states", JointState, self._fill_joint_state)
        self._cam_image_sub = rospy.Subscriber("/cameras/"+arm+"_hand_camera/image", Image, self._fill_image_data)
        self._pub_display = rospy.Publisher("/robot/xdisplay", Image, latch=True, queue_size=1)

    def joint_angle(self):
        return deepcopy(self._joint_angle)

    def set_robot_state(self, state):
        msg = Bool()
        msg.data = state
        self._robot_state.publish(msg)

    def set_joint_position(self, positions):
        self._command_msg.names = list(positions.keys())
        self._command_msg.command = list(positions.values())
        self._command_msg.mode = JointCommand.POSITION_MODE
        self._pub_joint_cmd.publish(self._command_msg)

    def set_joint_velocity(self, velocities):
        self._command_msg.names = list(velocities.keys())
        self._command_msg.command = list(velocities.values())
        self._command_msg.mode = JointCommand.VELOCITY_MODE
        self._pub_joint_cmd.publish(self._command_msg)

    def move_to_joint_position(self, positions, timeout=15.0):
        current_angle = self.joint_angle()
        end_time = rospy.get_time() + timeout
        
        # update the target based on the current location
        # if you use this instead of positions, the jerk
        # will be smaller.
        def current_target():
            for joint in positions:
                current_angle[joint] = 0.012488 * positions[joint] + 0.98751 * current_angle[joint]
            return current_angle

        def difference():
            diffs = []
            for joint in positions:
                diffs.append(abs(positions[joint] - self._joint_angle[joint]))
            return diffs

        while any(diff > 0.008726646 for diff in difference()) and rospy.get_time() < end_time:
            self.set_joint_position(current_target())
            self.rate.sleep()
        return all(diff < 0.008726646 for diff in difference())
    
    def move_to_neutral(self):
        angles = dict(list(zip(self._joint_names, [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0])))
        return self.move_to_joint_position(angles)

    def move_to_zero(self):
        angles = dict(list(zip(self._joint_names, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
        return self.move_to_joint_position(angles)

    def _fill_joint_state(self, msg):
        for idx, name in enumerate(msg.name):
            if name in self._joint_names:
                self._joint_angle[name] = msg.position[idx]
                self._joint_velocity[name] = msg.velocity[idx]
                self._joint_effort[name] = msg.effort[idx]

    def _fill_image_data(self, msg):
        self._cam_image = msg
        # self._pub_display.publish(msg)

    def _set_display_data(self, image):
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
        self._pub_display.publish(msg)

    def _set_camera(self, camera_name, state, width=640, height=400, fps=30):
        if state:
            rospy.wait_for_service("/cameras/open")
            camera_proxy = rospy.ServiceProxy("/cameras/open", OpenCamera)
            settings = CameraSettings()
            settings.width = width
            settings.height = height
            settings.fps = fps
            response = camera_proxy(camera_name, settings)
            return response
        else:
            rospy.wait_for_service("/cameras/close")
            camera_proxy = rospy.ServiceProxy("/cameras/close", CloseCamera)
            response = camera_proxy(camera_name)
            return response

