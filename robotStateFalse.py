import time
import rospy
import baxter
import _thread
import threading

rospy.init_node("testing")
robotL = baxter.BaxterRobot(rate=100, arm="left")
robotR = baxter.BaxterRobot(rate=100, arm="right")
rospy.sleep(2.0)

robotL.set_robot_state(True)

#sequential movements
#print(robotL.move_to_neutral())
#print(robotR.move_to_neutral())


class myThread (threading.Thread):
   def __init__(self, threadID, name, robot):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.robot = robot
   def run(self):
      print ("Starting " + self.name)
      print(self.robot.move_to_neutral())
      print ("Exiting " + self.name)


thread1 = myThread(1, "Thread-1", robotL)
thread2 = myThread(2, "Thread-2", robotR)
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print ("Exiting Main Thread")
robotL.set_robot_state(False)
