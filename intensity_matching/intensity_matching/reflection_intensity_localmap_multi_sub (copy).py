# ROS2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
# ROS msgs
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int8MultiArray
import message_filters
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros import Buffer
from tf2_ros import TransformListener
from tf2_ros import TransformException
from rclpy.executors import MultiThreadedExecutor
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
# Python
import numpy as np
import math
import cv2
import yaml
import os
from collections import OrderedDict
import pandas as pd
from cv_bridge import CvBridge
import time

class ObsBayesMap(Node):
    def __init__(self):
        super().__init__('obs_bayes_map')

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        map_qos_profile_sub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth = 1
        )
        # Subscriber
        self.local_odom_sub = self.create_subscription(nav_msgs.Odometry,'/odom/combine',self.get_local_odom, qos_profile_sub)
        self.global_odom_sub = self.create_subscription(nav_msgs.Odometry,'/fusion/odom', self.get_global_odom, qos_profile_sub)
        #self.pcd_ground_sub = message_filters.Subscriber(self, sensor_msgs.PointCloud2, '/pcd_segment_ground')
        #self.pcd_middle_sub = message_filters.Subscriber(self, sensor_msgs.PointCloud2, '/pcd_segment_middle')
        #self.pcd_high_sub = message_filters.Subscriber(self, sensor_msgs.PointCloud2, '/pcd_segment_high')
        self.pcd_sub = self.create_subscription(sensor_msgs.PointCloud2,'/pcd_rotation_merge',self.reflect_map,qos_profile_sub)

        # Publisher
        self.bayes_map_pub = self.create_publisher(OccupancyGrid,'/obs_bayes_map',qos_profile_sub)
        self.static_map_pub = self.create_publisher(OccupancyGrid,'/static_obs_map',qos_profile_sub)
        self.dynamic_map_pub = self.create_publisher(OccupancyGrid,'/dynamic_obs_map',qos_profile_sub)
        self.log_map_pub = self.create_publisher(OccupancyGrid,'/log_map',qos_profile_sub)
        self.pcd_ground_global_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_ground_global', qos_profile) 
        self.reflect_map_ground_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_ground_local', qos_profile_sub)
        self.reflect_map_middle_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_middle_local', qos_profile_sub)
        self.reflect_map_high_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_high_local', qos_profile_sub)
        self.reflect_map_ground_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_ground_global', qos_profile_sub)
        self.reflect_map_middle_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_middle_global', qos_profile_sub)
        self.reflect_map_high_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_high_global', qos_profile_sub)
        self.reflect_map_dynamic_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_dynamic_global', qos_profile_sub)
        self.remove_map_pub = self.create_publisher(OccupancyGrid, 'remove_map', qos_profile_sub)
        # RGB map publisher
        self.rgb_map_local_pub = self.create_publisher(Image, '/rgb_reflect_map_local', qos_profile_sub)
        self.rgb_map_global_pub = self.create_publisher(Image, '/rgb_reflect_map_global', qos_profile_sub)
        self.bridge = CvBridge()

        #timer
        #self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_flag = 0

        # multi subscriber
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.pcd_ground_sub, self.pcd_middle_sub, self.pcd_high_sub], queue_size=1, slop=0.05)
        #self.ts.registerCallback(self.reflect_map)

        #tf
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Parameter
        self.grid_pixel = 1000 / 50.0
        self.MAP_RANGE = 15.0 #[m]
        self.size = int(self.MAP_RANGE * 2 * self.grid_pixel)
        self.RAY_RANGE = 50.0 #[m]
        self.ray_size = int(self.RAY_RANGE * 2 * self.grid_pixel)
        self.bin_count = 720
        #ground 
        self.ground_pixel = self.grid_pixel
        self.MAP_RANGE_GL = 20.0 #[m]
        self.MAP_LIM_X_MIN = -25.0 #[m]
        self.MAP_LIM_X_MAX =  25.0 #[m]
        self.MAP_LIM_Y_MIN = -25.0 #[m]
        self.MAP_LIM_Y_MAX =  25.0 #[m]

        # Odom
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        self.prev_x = 0.0
        self.prev_y = 0.0

        #ekf_odom positon init
        self.ekf_position_x = 0.0 #[m]
        self.ekf_position_y = 0.0 #[m]
        self.ekf_position_z = 0.0 #[m]
        self.ekf_theta_x = 0.0 #[deg]
        self.ekf_theta_y = 0.0 #[deg]
        self.ekf_theta_z = 0.0 #[deg]

        # Weighted Occupancy
        self.logodds = np.zeros((self.size, self.size), dtype=np.float32)
        self.dynamic_count = np.zeros((self.size, self.size), dtype=np.uint8)
        self.shift_residual_x = 0.0
        self.shift_residual_y = 0.0

        #points array init
        self.middle_points = np.zeros((4, 0), dtype=np.float32)

        #mid360 buff
        self.pcd_ground_buff = np.array([[],[],[],[]]);
        self.pcd_middle_buff = np.array([[],[],[],[]]);
        self.pcd_high_buff = np.array([[],[],[],[]]);

        #map position
        self.map_position_x_buff = 0.0 #[m]
        self.map_position_y_buff = 0.0 #[m]
        self.map_position_z_buff = 0.0 #[m]
        self.map_theta_z_buff = 0.0 #[deg]
        self.map_number = 0 # int

        # map flag
        self.map_data = 0
        self.map_data_flag = 0
        self.map_data_gl = 0
        self.map_data_gl_flag = 0
        self.MAKE_GL_MAP_FLAG = 0 # make map
        self.save_dir = os.path.expanduser('~/ros2_ws/src/map/nakaniwa_0520')
        yaml.add_representer(OrderedDict, ordered_dict_representer, Dumper=MyDumper)
        yaml.add_representer(list, list_representer, Dumper=MyDumper)

        self.last_keyframe_x = 0.0
        self.last_keyframe_y = 0.0

        # ==========================================================
        # RGB map / 処理周期
        # ==========================================================

        # RGB mapは最大10 Hz
        self.rgb_map_period = 0.10
        self.last_rgb_map_time = 0.0

        # レイキャスト設定
        # 元:
        # RAY_RANGE = 50.0
        # bin_count = 720
        # step = 0.05
        #
        # かなり重かったので軽量化
        self.RAY_RANGE = 30.0
        self.bin_count = 360
        self.ray_step = 0.10

        # バッファ全体の重複除去は毎回行わない
        # 10回に1回だけ実施
        self.buffer_dedup_interval = 10
        self.buffer_dedup_count = 0

        # デバッグprintを毎回出さない
        self.debug_count = 0

    def timer_callback(self):
        if self.start_flag == 0:
            return
        
        #self.reflect_map(self.t_stamp, self.middle_points, self.dynamic_occ)
        if self.map_data_flag > 0:
            self.reflect_map_ground_local_publisher.publish(self.map_data_ground)
            self.reflect_map_middle_local_publisher.publish(self.map_data_middle)
            self.reflect_map_high_local_publisher.publish(self.map_data_high)
        #gl map
        if self.map_data_gl_flag > 0:
            self.reflect_map_ground_global_publisher.publish(self.map_data_ground_gl) 
            self.reflect_map_middle_global_publisher.publish(self.map_data_middle_gl) 
            self.reflect_map_high_global_publisher.publish(self.map_data_high_gl) 

    def get_local_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.theta_x = 0 #roll /math.pi*180
        self.theta_y = 0 #pitch /math.pi*180
        self.theta_z = yaw /math.pi*180


    def get_global_odom(self, msg):
        self.ekf_position_x = msg.pose.pose.position.x
        self.ekf_position_y = msg.pose.pose.position.y
        self.ekf_position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.ekf_theta_x = 0 #roll /math.pi*180
        self.ekf_theta_y = 0 #pitch /math.pi*180
        self.ekf_theta_z = yaw /math.pi*180

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        #self.tf_broadcaster.sendTransform(t)

    def pointcloud2_to_array(self, msg):
        dtype = np.dtype({
            'names': ['x', 'y', 'z', 'intensity'],
            'formats': ['<f4', '<f4', '<f4', '<f4'],
            'offsets': [0, 4, 8, 12],
            'itemsize': msg.point_step
        })
        data = np.frombuffer(
            msg.data,
            dtype=dtype
        )
        x = data['x']
        y = data['y']
        z = data['z']
        intensity = data['intensity']
        return x, y, z, intensity

    def reflect_map(self, msg):

        # ==========================================================
        # 10 Hz制限
        # ==========================================================

        now = time.perf_counter()

        if now - self.last_rgb_map_time < self.rgb_map_period:
            return

        self.last_rgb_map_time = now

        start_time = now

        # ==========================================================
        # 1. Odometry取得
        # ==========================================================

        local_x = self.position_x
        local_y = self.position_y

        theta_z = math.radians(self.theta_z)

        # ==========================================================
        # 2. pcd_rotation_merge → numpy
        # ==========================================================

        x, y, z, intensity = self.pointcloud2_to_array(msg)

        # ==========================================================
        # 3. TF変換
        #
        # pcd_rotation_merge → odom
        #
        # ここでTFを1回だけ実行する
        # ==========================================================

        try:

            global_msg = self.tf_buffer.transform(
                msg,
                "odom"
            )

        except Exception as e:

            self.get_logger().warn(
                f"TF transform failed: {e}"
            )

            return

        # ==========================================================
        # 4. TF変換後のPointCloud2 → numpy
        # ==========================================================

        gx, gy, gz, gi = self.pointcloud2_to_array(
            global_msg
        )

        # ==========================================================
        # 5. 高さでGround / Middle / Highに分割
        #
        # 元のPcdHeightSegmentationと同じ条件
        #
        # Ground : -15cm ～ 12cm
        # Middle : 50cm ～ 100cm
        # High   : 200cm ～ 400cm
        #
        # ==========================================================

        ground_mask = (
            (gz >= -0.15) &
            (gz <= 0.12)
        )

        middle_mask = (
            (gz >= 0.50) &
            (gz <= 1.00)
        )

        high_mask = (
            (gz >= 2.00) &
            (gz <= 4.00)
        )

        # ==========================================================
        # 6. Ground
        # ==========================================================

        ground_global = np.vstack((
            gx[ground_mask],
            gy[ground_mask],
            gz[ground_mask],
            gi[ground_mask]
        ))

        # ==========================================================
        # 7. Middle
        # ==========================================================

        middle_global = np.vstack((
            gx[middle_mask],
            gy[middle_mask],
            gz[middle_mask],
            gi[middle_mask]
        ))

        # ==========================================================
        # 8. High
        # ==========================================================

        high_global = np.vstack((
            gx[high_mask],
            gy[high_mask],
            gz[high_mask],
            gi[high_mask]
        ))

        # ==========================================================
        # 9. Middle pointをロボット中心基準の座標に変換
        #
        # すでにodomへTF変換済みなので、
        # yaw回転をもう一度行う必要はない
        #
        # global座標 - robot位置
        # ==========================================================

        if middle_global.shape[1] > 0:

            middle_x_local = (
                middle_global[0] - local_x
            )

            middle_y_local = (
                middle_global[1] - local_y
            )

            middle_z_local = middle_global[2]

            middle_i_local = middle_global[3]

        else:

            middle_x_local = np.empty(
                0,
                dtype=np.float32
            )

            middle_y_local = np.empty(
                0,
                dtype=np.float32
            )

            middle_z_local = np.empty(
                0,
                dtype=np.float32
            )

            middle_i_local = np.empty(
                0,
                dtype=np.float32
            )

        # ==========================================================
        # 10. Middle pointの角度・距離
        # ==========================================================

        angle = np.arctan2(
            middle_y_local,
            middle_x_local
        )

        distance = np.sqrt(
            middle_x_local * middle_x_local +
            middle_y_local * middle_y_local
        )

        valid = (
            np.isfinite(angle) &
            np.isfinite(distance) &
            (distance > 0.05) &
            (distance < self.RAY_RANGE)
        )

        angle = angle[valid]
        distance = distance[valid]

        middle_x_local = middle_x_local[valid]
        middle_y_local = middle_y_local[valid]

        # ==========================================================
        # 11. 各角度binで最も近い点だけ使用
        # ==========================================================

        if len(angle) > 0:

            bin_index = (
                (
                    (angle + np.pi) /
                    (2.0 * np.pi) *
                    self.bin_count
                )
                .astype(np.int32)
            )

            bin_index = np.clip(
                bin_index,
                0,
                self.bin_count - 1
            )

            order = np.lexsort(
                (distance, bin_index)
            )

            bin_sorted = bin_index[order]

            _, first_index = np.unique(
                bin_sorted,
                return_index=True
            )

            selected = order[first_index]

            middle_x_occ = (
                middle_x_local[selected]
            )

            middle_y_occ = (
                middle_y_local[selected]
            )

            middle_dist_occ = (
                distance[selected]
            )

        else:

            middle_x_occ = np.empty(
                0,
                dtype=np.float32
            )

            middle_y_occ = np.empty(
                0,
                dtype=np.float32
            )

            middle_dist_occ = np.empty(
                0,
                dtype=np.float32
            )

        # ==========================================================
        # 12. Local occupancy map
        # ==========================================================

        map_size = int(
            self.MAP_RANGE *
            2.0 *
            self.grid_pixel
        )

        center = map_size // 2

        if len(middle_dist_occ) > 0:

            ray_count = int(
                self.RAY_RANGE /
                self.ray_step
            )

            ray_distance = (
                np.arange(
                    ray_count,
                    dtype=np.float32
                ) *
                self.ray_step
            )

            ray_distance = (
                ray_distance[:, None]
            )

            occ_distance = (
                middle_dist_occ[None, :]
            )

            free_mask = (
                ray_distance <
                (occ_distance - self.ray_step)
            )

            direction_x = (
                middle_x_occ /
                middle_dist_occ
            )

            direction_y = (
                middle_y_occ /
                middle_dist_occ
            )

            ray_x = (
                direction_x[None, :] *
                ray_distance
            )

            ray_y = (
                direction_y[None, :] *
                ray_distance
            )

            gx_map = np.round(
                ray_x *
                self.grid_pixel
            ).astype(np.int32) + center

            gy_map = np.round(
                ray_y *
                self.grid_pixel
            ).astype(np.int32) + center

            valid_map = (
                (gx_map >= 0) &
                (gx_map < map_size) &
                (gy_map >= 0) &
                (gy_map < map_size)
            )

            free_mask &= valid_map

            # ------------------------------------------------------
            # free
            # ------------------------------------------------------

            free_x = gx_map[free_mask]
            free_y = gy_map[free_mask]

            if len(free_x) > 0:

                self.logodds[
                    free_y,
                    free_x
                ] -= 0.1

            # ------------------------------------------------------
            # occupied
            # ------------------------------------------------------

            occ_x = np.round(
                middle_x_occ *
                self.grid_pixel
            ).astype(np.int32) + center

            occ_y = np.round(
                middle_y_occ *
                self.grid_pixel
            ).astype(np.int32) + center

            valid_occ = (
                (occ_x >= 0) &
                (occ_x < map_size) &
                (occ_y >= 0) &
                (occ_y < map_size)
            )

            occ_x = occ_x[valid_occ]
            occ_y = occ_y[valid_occ]

            if len(occ_x) > 0:

                self.logodds[
                    occ_y,
                    occ_x
                ] += 0.4

        # ==========================================================
        # 13. Voxel downsample
        # ==========================================================

        ground_global = self.voxel_downsample(
            ground_global,
            self.ground_pixel
        )

        middle_global = self.voxel_downsample(
            middle_global,
            self.ground_pixel
        )

        high_global = self.voxel_downsample(
            high_global,
            self.ground_pixel
        )

        # ==========================================================
        # 14. Global map範囲
        # ==========================================================

        map_x_min = local_x - 25.0
        map_x_max = local_x + 25.0

        map_y_min = local_y - 25.0
        map_y_max = local_y + 25.0

        # ==========================================================
        # 15. 既存bufferを範囲内だけ残す
        # ==========================================================

        if self.pcd_ground_buff.shape[1] > 0:

            mask = self.pcd_serch(
                self.pcd_ground_buff,
                map_x_min,
                map_x_max,
                map_y_min,
                map_y_max
            )

            self.pcd_ground_buff = (
                self.pcd_ground_buff[:, mask]
            )

        if self.pcd_middle_buff.shape[1] > 0:

            mask = self.pcd_serch(
                self.pcd_middle_buff,
                map_x_min,
                map_x_max,
                map_y_min,
                map_y_max
            )

            self.pcd_middle_buff = (
                self.pcd_middle_buff[:, mask]
            )

        if self.pcd_high_buff.shape[1] > 0:

            mask = self.pcd_serch(
                self.pcd_high_buff,
                map_x_min,
                map_x_max,
                map_y_min,
                map_y_max
            )

            self.pcd_high_buff = (
                self.pcd_high_buff[:, mask]
            )

        # ==========================================================
        # 16. bufferへ追加
        # ==========================================================

        if ground_global.shape[1] > 0:

            self.pcd_ground_buff = np.hstack(
                (
                    self.pcd_ground_buff,
                    ground_global
                )
            )

        if middle_global.shape[1] > 0:

            self.pcd_middle_buff = np.hstack(
                (
                    self.pcd_middle_buff,
                    middle_global
                )
            )

        if high_global.shape[1] > 0:

            self.pcd_high_buff = np.hstack(
                (
                    self.pcd_high_buff,
                    high_global
                )
            )

        # ==========================================================
        # 17. buffer全体の重複除去
        # ==========================================================

        self.buffer_dedup_count += 1

        if (
            self.buffer_dedup_count >=
            self.buffer_dedup_interval
        ):

            self.buffer_dedup_count = 0

            if self.pcd_ground_buff.shape[1] > 0:

                self.pcd_ground_buff = (
                    self.voxel_downsample(
                        self.pcd_ground_buff,
                        self.ground_pixel
                    )
                )

            if self.pcd_middle_buff.shape[1] > 0:

                self.pcd_middle_buff = (
                    self.voxel_downsample(
                        self.pcd_middle_buff,
                        self.ground_pixel
                    )
                )

            if self.pcd_high_buff.shape[1] > 0:

                self.pcd_high_buff = (
                    self.voxel_downsample(
                        self.pcd_high_buff,
                        self.ground_pixel
                    )
                )

        # ==========================================================
        # 18. RGB map
        # ==========================================================

        local_map_ground = grid_map_set(
            self.pcd_ground_buff,
            self.ground_pixel,
            local_x,
            local_y,
            self.MAP_RANGE,
            use_intensity=True
        )

        local_map_middle = grid_map_set(
            self.pcd_middle_buff,
            self.ground_pixel,
            local_x,
            local_y,
            self.MAP_RANGE,
            use_intensity=False
        )

        local_map_high = grid_map_set(
            self.pcd_high_buff,
            self.ground_pixel,
            local_x,
            local_y,
            self.MAP_RANGE,
            use_intensity=False
        )

        # ==========================================================
        # 19. RGB画像
        # ==========================================================

        rgb_map = np.zeros(
            (
                local_map_ground.shape[0],
                local_map_ground.shape[1],
                3
            ),
            dtype=np.uint8
        )

        # OpenCVなのでBGR
        rgb_map[:, :, 0] = local_map_high
        rgb_map[:, :, 1] = local_map_middle
        rgb_map[:, :, 2] = local_map_ground

        # ==========================================================
        # 20. ROS Image
        # ==========================================================

        rgb_msg = self.bridge.cv2_to_imgmsg(
            rgb_map,
            encoding="bgr8"
        )

        rgb_msg.header = msg.header

        self.rgb_map_local_pub.publish(
            rgb_msg
        )

        # ==========================================================
        # 21. Global ground point cloud
        # ==========================================================

        if self.pcd_ground_buff.shape[1] > 0:

            ground_cloud_msg = (
                point_cloud_intensity_msg(
                    self.pcd_ground_buff.T,
                    msg.header.stamp,
                    "odom"
                )
            )

            self.pcd_ground_global_publisher.publish(
                ground_cloud_msg
            )

        # ==========================================================
        # debug
        # ==========================================================

        self.debug_count += 1

        if self.debug_count % 10 == 0:

            elapsed = (
                time.perf_counter() -
                start_time
            )

            self.get_logger().info(
                f"RGB map processing: "
                f"{elapsed * 1000:.1f} ms"
            )
        
    def make_ref_map(self, image, position_x, position_y, theta_z, layer):
        #map_pos_diff = math.sqrt((position_x - self.map_position_x_buff)**2 + (position_y - self.map_position_y_buff)**2)
        #map_theta_diff = abs(theta_z -  self.map_theta_z_buff)
        #if ( (map_pos_diff > 10) or ((map_pos_diff > 2) and (map_theta_diff > 40)) ):
        map_number_str = str(self.map_number).zfill(3)
        # 保存ディレクトリの絶対パスを取得
        #save_path = os.path.join(self.save_dir, f'waypoint_map_{map_number_str}')
        pgm_filename = os.path.join(self.save_dir, f'waypoint_map_{layer}_{map_number_str}' + ".pgm")
        pgm_filename_meta = os.path.join(f'waypoint_map_{layer}_{map_number_str}' + ".pgm")
        yaml_filename = os.path.join(self.save_dir, f'waypoint_map_{layer}_{map_number_str}' + ".yaml")
        # ディレクトリが存在するか確認、存在しない場合は作成
        os.makedirs(self.save_dir, exist_ok=True)
        '''
        subprocess.run([
            'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
            '-t', '/reflect_map_global',
            '--occ', '0.13',
            '--free', '0.05',
            '-f', save_path,
            '--ros-args', '-p', 'map_subscribe_transient_local:=true', '-r', '__ns:=/namespace'
        ])
        self.get_logger().info(f'External node executed with argument --arg1 {map_number_str}')
        '''
        # 閾値の設定 
        occ_threshold_param = 0.13 # 占有のしきい値  for save
        occ_threshold = occ_threshold_param * 100 # 占有のしきい値 
        free_threshold_param = 0.05 # 自由空間のしきい値  for save 
        free_threshold = free_threshold_param * 100 # 自由空間のしきい値 
        #image = self.map_data_gl
        #image = np.array(self.map_data_gl.data).reshape((self.map_data_gl.info.height, self.map_data_gl.info.width))
        #print(f"image ={image}")
        # マスクを初期化 
        occupancy_grid = np.zeros_like(image) 
        # 占有空間、自由空間、未確定領域を設定 
        occupancy_grid[image >= occ_threshold] = 255 - 255
        # 占有空間 
        occupancy_grid[image <= free_threshold] = 255 - 0
        # 自由空間 
        occupancy_grid[(image > free_threshold) & (image < occ_threshold)] = 255 - (image[(image > free_threshold) & (image < occ_threshold)])/occ_threshold*100 # 未確定領域は元の値を保持 
        # マップの保存 
        cv2.imwrite(pgm_filename, occupancy_grid)
        
        # メタデータを定義 
        metadata = OrderedDict([ 
            ('image', pgm_filename_meta), 
            ('mode', 'trinary'), 
            ('resolution', 1/self.ground_pixel), 
            ('origin', [round(position_x - self.MAP_RANGE_GL, 1), round(position_y - self.MAP_RANGE_GL, 1), round(0, 1)]), 
            ('negate', 0), ('occupied_thresh', occ_threshold_param), 
            ('free_thresh', free_threshold_param) 
        ])
        
        # YAMLファイルとしてメタデータを保存 
        with open(yaml_filename, 'w') as yaml_file: 
            yaml.dump(metadata, yaml_file, Dumper=MyDumper, default_flow_style=False)
        
        
        #self.map_position_x_buff = position_x #[m]
        #self.map_position_y_buff = position_y #[m]
        #self.map_theta_z_buff = theta_z #[deg]
        #self.map_number += 1

    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max):
        if pointcloud.shape[1] == 0:
            return np.zeros(0, dtype=bool)
        return (
            (pointcloud[0] >= x_min) &
            (pointcloud[0] <= x_max) &
            (pointcloud[1] >= y_min) &
            (pointcloud[1] <= y_max)
        )
    
    def voxel_downsample(self, points, pixel):
        if points.shape[1] == 0:
            return points
        # xyzをグリッド座標へ
        xyz = np.round(
            points[:3, :] * pixel
        ).astype(np.int32)
        # 同じxyzの点を1つにする
        _, index = np.unique(
            xyz.T,
            axis=0,
            return_index=True
        )
        return points[:, index]
    
    def save_rgb_map(self, ground_img, middle_img, high_img, position_x, position_y, theta_z):
        print("ground max =", np.max(ground_img))
        print("middle max =", np.max(middle_img))
        print("high max =", np.max(high_img))
        map_number_str = str(self.map_number).zfill(3)
        png_filename = os.path.join(self.save_dir, f'waypoint_map_rgb_{map_number_str}.png')
        yaml_filename = os.path.join(self.save_dir, f'waypoint_map_rgb_{map_number_str}' + ".yaml")
        os.makedirs(self.save_dir, exist_ok=True)
        # float32化
        ground = ground_img.astype(np.float32)
        middle = middle_img.astype(np.float32)
        high = high_img.astype(np.float32)
        rgb_map = np.zeros((ground.shape[0], ground.shape[1], 3), dtype=np.uint8)
        # BGR(OpenCV)
        rgb_map[:, :, 0] = np.clip(ground * 255.0 / 100.0, 0, 255).astype(np.uint8)
        rgb_map[:, :, 1] = np.clip(middle * 255.0 / 100.0, 0, 255).astype(np.uint8)
        rgb_map[:, :, 2] = np.clip(high * 255.0 / 100.0, 0, 255).astype(np.uint8)
        print("rgb max =", np.max(rgb_map))
        print("rgb dtype =", rgb_map.dtype)
        #mask = np.sum(rgb_map, axis=2) == 0
        #rgb_map[mask] = [30, 30, 30]
        cv2.imwrite(png_filename, rgb_map)
        print("saved:", png_filename)

        occ_threshold_param = 0.13 # 占有のしきい値  for save
        occ_threshold = occ_threshold_param * 100 # 占有のしきい値 
        free_threshold_param = 0.05 # 自由空間のしきい値  for save 
        free_threshold = free_threshold_param * 100 # 自由空間のしきい値 

        metadata = OrderedDict([ 
            ('image', png_filename), 
            ('mode', 'trinary'), 
            ('resolution', 1/self.ground_pixel), 
            ('origin', [round(position_x - self.MAP_RANGE_GL, 1), round(position_y - self.MAP_RANGE_GL, 1), round(0, 1)]), 
            ('negate', 0), ('occupied_thresh', occ_threshold_param), 
            ('free_thresh', free_threshold_param) 
        ])
        
        # YAMLファイルとしてメタデータを保存 
        with open(yaml_filename, 'w') as yaml_file: 
            yaml.dump(metadata, yaml_file, Dumper=MyDumper, default_flow_style=False)
    
    def make_rgb_map(self, ground_img, middle_img, high_img):
        ground = ground_img.astype(np.float32)
        middle = middle_img.astype(np.float32)
        high = high_img.astype(np.float32)
        rgb_map = np.zeros((ground.shape[0], ground.shape[1], 3), dtype=np.uint8)
        # OpenCVはBGR
        rgb_map[:, :, 0] = np.clip(ground * 255.0 / 100.0, 0, 255).astype(np.uint8)
        rgb_map[:, :, 1] = np.clip(middle * 255.0 / 100.0, 0, 255).astype(np.uint8)
        rgb_map[:, :, 2] = np.clip(high * 255.0 / 100.0, 0, 255).astype(np.uint8)
        # 背景を少し明るくする
        #mask = np.sum(rgb_map, axis=2) == 0
        #rgb_map[mask] = [30, 30, 30]
        return rgb_map

# カスタムDumperの設定を追加 
class MyDumper(yaml.Dumper): 
    def increase_indent(self, flow=False, indentless=False): 
        return super(MyDumper, self).increase_indent(flow=flow, indentless=indentless)
def ordered_dict_representer(dumper, data): 
    return dumper.represent_dict(data.items()) 
def list_representer(dumper, data): 
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True) 

def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])
    
    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])
    
    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])
    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    #print(f"rot_matrix ={rot_matrix}")
    #print(f"pointcloud ={pointcloud.shape}")
    rot_pointcloud = rot_matrix.dot(pointcloud)
    return rot_pointcloud, rot_matrix

def quaternion_to_euler(x, y, z, w):
    # クォータニオンから回転行列を計算
    rot_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])

    # 回転行列からオイラー角を抽出
    roll = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
    pitch = np.arctan2(-rot_matrix[2, 0], np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2))
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return roll, pitch, yaw

def point_cloud_intensity_msg(points, t_stamp, parent_frame):
    # In a PointCloud2 message, the point cloud is stored as an byte 
    # array. In order to unpack it, we also include some parameters 
    # which desribes the size of each individual point.
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.
    data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [
            sensor_msgs.PointField(name='x', offset=0, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='y', offset=4, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='z', offset=8, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
        ]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = std_msgs.Header(frame_id=parent_frame, stamp=t_stamp)
    

    return sensor_msgs.PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4), # Every point consists of three float32s.
        row_step=(itemsize * 4 * points.shape[0]), 
        data=data
    )


def make_map_msg(map_data_set, resolution, position, orientation, header_stamp, map_range, frame_id):
    map_data = OccupancyGrid()
    map_data.header.stamp =  header_stamp
    map_data.info.map_load_time = header_stamp
    map_data.header.frame_id = frame_id
    map_data.info.width = map_data_set.shape[0]
    map_data.info.height = map_data_set.shape[1]
    map_data.info.resolution = 1/resolution #50/1000#resolution
    pos_round = np.round(position * resolution) / resolution
    map_data.info.origin.position.x = float(pos_round[0] -map_range) #位置オフセット
    map_data.info.origin.position.y = float(pos_round[1] -map_range)
    map_data.info.origin.position.z = float(0.0) #position[2]
    map_data.info.origin.orientation.w = float(orientation[0])#
    map_data.info.origin.orientation.x = float(orientation[1])
    map_data.info.origin.orientation.y = float(orientation[2])
    map_data.info.origin.orientation.z = float(orientation[3])
    map_data_cv = cv2.flip(map_data_set, 0, dst = None)
    map_data_int8array = [i for row in  map_data_cv.tolist() for i in row]
    map_data.data = Int8MultiArray(data=map_data_int8array).data
    return map_data

'''
フィールド名	内容
image	占有データを含む画像ファイルへのパス。 絶対パス、またはYAMLファイルの場所からの相対パスを設定可能。
resolution	地図の解像度（単位はm/pixel）。
origin	（x、y、yaw）のような地図の左下のピクセルからの2D姿勢で、yawは反時計回りに回転します（yaw = 0は回転しないことを意味します）。現在、システムの多くの部分ではyawを無視しています。
occupied_thresh	この閾値よりも大きい占有確率を持つピクセルは、完全に占有されていると見なされます。
free_thresh	占有確率がこの閾値未満のピクセルは、完全に占有されていないと見なされます。
negate	白/黒について、空き/占有の意味を逆にする必要があるかどうか（閾値の解釈は影響を受けません）
'''

def grid_map_set(
    pointcloud,
    map_pixel,
    position_x,
    position_y,
    map_range,
    use_intensity=False
):

    size = int(
        map_range * 2.0 * map_pixel
    )

    map_data = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    if pointcloud.shape[1] == 0:
        return map_data


    # ==========================================================
    # global座標 → local map pixel
    # ==========================================================

    px = (
        (
            pointcloud[0] -
            (position_x - map_range)
        ) * map_pixel
    ).astype(np.int32)

    py = (
        (
            pointcloud[1] -
            (position_y - map_range)
        ) * map_pixel
    ).astype(np.int32)


    valid = (
        (px >= 0) &
        (px < size) &
        (py >= 0) &
        (py < size)
    )

    px = px[valid]
    py = py[valid]


    if len(px) == 0:
        return map_data


    # intensity
    if use_intensity:

        intensity = pointcloud[3, valid]

        intensity = np.clip(
            intensity,
            0.0,
            255.0
        ).astype(np.uint8)

        map_data[
            size - 1 - py,
            px
        ] = intensity

    else:

        map_data[
            size - 1 - py,
            px
        ] = 100


    return map_data
###################################################################

def main():
    rclpy.init()

    node = ObsBayesMap()

    executor = MultiThreadedExecutor(
        num_threads=4
    )

    executor.add_node(node)

    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
