#!/usr/bin/env python
import rospy 
from fsd_common_msgs.msg import comb


class RosNode:
    def __init__(self):
        rospy.init_node("trans_node")
        rospy.loginfo("Starting RosNode.")
        subs = rospy.Subscriber("/comb", comb, self.trans)
        self.trans_pub = rospy.Publisher("/trans_comb", comb, queue_size=10)
        # R = rospy.Rate(10)
        
        # while(not rospy.is_shutdown()):
            
        #     R.sleep()
    def trans(self,msg):
        msgs= msg
        msgs.Heading = msg.Heading * 3.14159265 /180

        self.trans_pub.publish(msgs)



if __name__ == "__main__":
    ros_node = RosNode()
    rospy.spin()