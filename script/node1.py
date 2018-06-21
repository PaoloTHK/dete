#!/usr/bin/env python

import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
import neuralnet1 as nrn
import cv2, os, glob, sys, time
from threading import Timer

cap = cv2.VideoCapture(0)
results = []
#r = 0
#i = 0
	
#class GoForwardAvoid():
#	def __init__(self):
#		rospy.init_node('nav_test', anonymous=False)
	#def start(results):		
	        # Publisher to manually control the robot (e.g. to stop it)
	#        cmd_vel_pub = rospy.Publisher('cmd_vel', Twist)

		# Subscribe to the move_base action server
	#	move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)		
		
		#allow up to 5 seconds for the action server to come up
	#	move_base.wait_for_server(rospy.Duration(5))
	#	rospy.loginfo("wait for the action server to come up")		
		
		# A variable to hold the initial pose of the robot to be set by 
		# the user in RViz
	#	initial_pose = PoseWithCovarianceStamped()

		# Make sure we have the initial pose
	#        while initial_pose.header.stamp == "":
	#            rospy.sleep(1)

	#	while initial_pose.header.stamp == "":
        #    		rospy.sleep(1)

while(True):
	for i in range(len(results)):	
	#self.detect_from_cvmat()
	#r=nrn.show_results(results[i][0])

		if nrn.show_results(results[i][5]) >= 20:
			print(resilts[i][0] + '30')
        ret, frame = cap.read()
	nrn.detect_from_cvmat(frame)
				        #T = Timer(10.0, timeout)
				        #if T.start() >= 3:
					#we'll send a goal to the robot to move 3 meters forward
	#				goal = MoveBaseGoal()
					#goal.target_pose.header.frame_id = 'map' # base_link
					#goal.target_pose.header.stamp = rospy.Time.now()
	#				goal.target_pose.pose.position.x = 1.0 #3 meters
	#				goal.target_pose.pose.orientation.w = 1.0 #go forward

					#start moving 
	#				move_base.send_goal(goal)

					#allow Olive up to 60 seconds to complete task
	#				success = move_base.wait_for_result(rospy.Duration(10)) 


	#				if not success:
	#							move_base.cancel_goal()
	#							rospy.loginfo("The base failed to move forward 3 meters for some reason")
	#				else:
						# We made it!
	#					state = move_base.get_state()
	#					if state == GoalStatus.SUCCEEDED:
	#						rospy.loginfo("Hooray, the base moved 3 meters forward")
	if cv2.waitKey(1) & 0xFF == ord('q'):
			  break
#	def shutdown(self):
#		rospy.loginfo("Stop")


#if __name__ == '__main__':
#	try:	
#		GoForwardAvoid()
#	except rospy.ROSInterruptException:
#		rospy.loginfo("Exception thrown")
