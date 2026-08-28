#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


class AmclToOdom(Node):

    def __init__(self):
        super().__init__("amcl_to_odom")

        # AMCL
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_callback,
            10
        )

        # Odometryとして出力
        self.publisher = self.create_publisher(
            Odometry,
            "/amcl_odom",
            10
        )

        self.get_logger().info(
            "AMCL Pose -> Odometry conversion started"
        )

    def amcl_callback(self, msg):

        odom = Odometry()

        # --------------------------------
        # Header
        # --------------------------------

        odom.header = msg.header

        # --------------------------------
        # Frame
        # --------------------------------

        # AMCL poseは通常 map座標系
        odom.header.frame_id = "map"

        # child_frame_idは、
        # AMCLが推定しているロボット座標系
        odom.child_frame_id = "base_link"

        # --------------------------------
        # Position / Orientation
        # --------------------------------

        odom.pose.pose = msg.pose.pose

        # --------------------------------
        # Covariance
        # --------------------------------

        odom.pose.covariance = msg.pose.covariance

        # --------------------------------
        # Twist
        # --------------------------------

        # AMCL Poseには速度情報がないため、
        # ここでは0にする
        odom.twist.twist.linear.x = 0.0
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0

        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        # --------------------------------
        # Publish
        # --------------------------------

        self.publisher.publish(odom)


def main(args=None):

    rclpy.init(args=args)

    node = AmclToOdom()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
