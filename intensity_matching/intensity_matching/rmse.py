#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt


class OdometryRMSE(Node):

    def __init__(self):
        super().__init__('odometry_rmse')

        self.ground_truth = None
        self.estimated = None

        self.errors_x = []
        self.errors_y = []
        self.errors_position = []

        self.rmse_x = []
        self.rmse_y = []
        self.rmse_position = []

        self.time_data = []

        self.start_time = None

        self.gt_sub = self.create_subscription(
            Odometry,
            '/odom/UM982',
            self.gt_callback,
            10
        )

        self.est_sub = self.create_subscription(
            Odometry,
            '/odom_ekf_match',
            self.est_callback,
            10
        )

        self.get_logger().info('Odometry RMSE started')

    def gt_callback(self, msg):
        self.ground_truth = msg

        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds * 1e-9

        self.calculate_rmse()

    def est_callback(self, msg):
        self.estimated = msg

    def calculate_rmse(self):

        if self.ground_truth is None or self.estimated is None:
            return

        gt_x = self.ground_truth.pose.pose.position.x
        gt_y = self.ground_truth.pose.pose.position.y

        est_x = self.estimated.pose.pose.position.x
        est_y = self.estimated.pose.pose.position.y

        error_x = est_x - gt_x
        error_y = est_y - gt_y

        error_position = np.sqrt(
            error_x ** 2 +
            error_y ** 2
        )

        self.errors_x.append(error_x)
        self.errors_y.append(error_y)
        self.errors_position.append(error_position)

        # 累積RMSE
        rmse_x = np.sqrt(
            np.mean(np.array(self.errors_x) ** 2)
        )

        rmse_y = np.sqrt(
            np.mean(np.array(self.errors_y) ** 2)
        )

        rmse_position = np.sqrt(
            np.mean(np.array(self.errors_position) ** 2)
        )

        self.rmse_x.append(rmse_x)
        self.rmse_y.append(rmse_y)
        self.rmse_position.append(rmse_position)

        current_time = self.get_clock().now().nanoseconds * 1e-9

        self.time_data.append(
            current_time - self.start_time
        )

    def plot(self):

        if len(self.errors_position) == 0:
            self.get_logger().warn('No data')
            return

        # 最終RMSE
        final_rmse_x = self.rmse_x[-1]
        final_rmse_y = self.rmse_y[-1]
        final_rmse_position = self.rmse_position[-1]

        print()
        print('==============================')
        print('Odometry RMSE')
        print('==============================')
        print(f'X RMSE        : {final_rmse_x:.4f} m')
        print(f'Y RMSE        : {final_rmse_y:.4f} m')
        print(f'Position RMSE : {final_rmse_position:.4f} m')
        print('==============================')

        # -------------------------
        # RMSE推移
        # -------------------------

        plt.figure()

        plt.plot(
            self.time_data,
            self.rmse_x,
            label='X RMSE'
        )

        plt.plot(
            self.time_data,
            self.rmse_y,
            label='Y RMSE'
        )

        plt.plot(
            self.time_data,
            self.rmse_position,
            label='Position RMSE'
        )

        plt.xlabel('Time [s]')
        plt.ylabel('RMSE [m]')
        plt.title('Odometry RMSE over Time')

        plt.grid()
        plt.legend()

        plt.tight_layout()

        # -------------------------
        # 位置誤差
        # -------------------------

        plt.figure()

        plt.plot(
            self.time_data,
            self.errors_position,
            label='Position Error'
        )

        plt.xlabel('Time [s]')
        plt.ylabel('Position Error [m]')
        plt.title('Position Error over Time')

        plt.grid()
        plt.legend()

        plt.tight_layout()

        # -------------------------
        # X/Y誤差
        # -------------------------

        plt.figure()

        plt.plot(
            self.time_data,
            self.errors_x,
            label='X Error'
        )

        plt.plot(
            self.time_data,
            self.errors_y,
            label='Y Error'
        )

        plt.xlabel('Time [s]')
        plt.ylabel('Error [m]')
        plt.title('Position Error Components')

        plt.grid()
        plt.legend()

        plt.tight_layout()

        plt.show()


def main(args=None):

    rclpy.init(args=args)

    node = OdometryRMSE()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.plot()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
