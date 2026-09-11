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
        self.keyframe_initialized = False

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
        points = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        return x, y, z, intensity

    def reflect_map(self, msg): #(self, t_stamp, ground_points, middle_points, high_points)
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        t0 = time.perf_counter()
        try:
            transform = self.tf_buffer.lookup_transform("odom", "livox_frame", rclpy.time.Time.from_msg(msg.header.stamp), timeout=Duration(seconds=0.1))

        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return      

        global_msg = do_transform_cloud(msg, transform)

        x, y, z, intensity = self.pointcloud2_to_array(msg)
        global_x, global_y, global_z, global_intensity = self.pointcloud2_to_array(global_msg)
        ground_mask = ((z >= -0.15) & (z <= 0.12))
        middle_mask = ((z >= 0.50) & (z <= 1.00))
        high_mask = ((z >= 2.00) & (z <= 4.00))

        ground_points = np.vstack((x[ground_mask], y[ground_mask], z[ground_mask], intensity[ground_mask]))
        middle_points = np.vstack((x[middle_mask], y[middle_mask], z[middle_mask], intensity[middle_mask]))
        high_points = np.vstack((x[high_mask], y[high_mask], z[high_mask], intensity[high_mask]))

        ground_global = np.vstack((global_x[ground_mask], global_y[ground_mask], global_z[ground_mask], global_intensity[ground_mask]), dtype=np.float32)
        middle_global = np.vstack((global_x[middle_mask], global_y[middle_mask], global_z[middle_mask], global_intensity[middle_mask]), dtype=np.float32)
        high_global = np.vstack((global_x[high_mask], global_y[high_mask], global_z[high_mask], global_intensity[high_mask]), dtype=np.float32)
        # ground_points
        #ground_x, ground_y, ground_z, ground_intensity = self.pointcloud2_to_array(ground_msg)
        #ground_points = np.vstack((ground_x, ground_y, ground_z, ground_intensity))

        # high_points
        #high_x, high_y, high_z, high_intensity = self.pointcloud2_to_array(high_msg)
        #high_points = np.vstack((high_x, high_y, high_z, high_intensity))
        
        ############################ log odds ############################
        # PointCloud
        #points = self.pointcloud2_to_array(msg)
        ###middle_x, middle_y, middle_z, middle_intensity = self.pointcloud2_to_array(middle_msg) # pointcloud2点群取得 numpy array
        middle_x = middle_points[0] # pointcloud2点群取得 numpy array
        middle_y = middle_points[1]
        middle_z = middle_points[2]
        middle_intensity = middle_points[3]
        theta = self.theta_z * np.pi / 180.0

        # LiDAR座標を車両座標に変換（回転のみ）
        middle_x_rot = middle_x * np.cos(theta) - middle_y * np.sin(theta)
        middle_y_rot = middle_x * np.sin(theta) + middle_y * np.cos(theta)

        # Odometry取得
        #position set
        position_x=self.position_x; position_y=self.position_y; position_z=self.position_z;
        position = np.array([position_x, position_y, position_z])
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        ekf_position_x=self.ekf_position_x; ekf_position_y=self.ekf_position_y; ekf_position_z=self.ekf_position_z;
        ekf_position = np.array([ekf_position_x, ekf_position_y, ekf_position_z])
        ekf_theta_x=self.ekf_theta_x; ekf_theta_y=self.ekf_theta_y; ekf_theta_z=self.ekf_theta_z;
        prev_x = self.prev_x
        prev_y = self.prev_y

        if not self.keyframe_initialized:
            is_keyframe = True
        else:
            map_pos_diff = np.hypot(position_x - self.last_keyframe_x, position_y - self.last_keyframe_y)
            is_keyframe = map_pos_diff > 0.5

        #map_pos_diff = np.sqrt((ekf_position_x - self.map_position_x_buff)**2 + (ekf_position_y - self.map_position_y_buff)**2)
        #map_theta_diff = abs(ekf_theta_z - self.map_theta_z_buff)

        #is_keyframe = ((map_pos_diff > 0.5)) # or ((map_pos_diff > 0.2) and (map_theta_diff > 40))

        # local map
        middle_x_local = middle_x_rot
        middle_y_local = middle_y_rot

        # Map shift(前フレームからの移動量を計算)
        dx = position_x - prev_x
        dy = position_y - prev_y

        self.shift_residual_x += dx * self.grid_pixel # 移動量をピクセル単位に変換して残差に加算(少数の保持)
        self.shift_residual_y += dy * self.grid_pixel
        shift_x = int(self.shift_residual_x) # 残差から整数のシフト量を取得
        shift_y = int(self.shift_residual_y)
        self.shift_residual_x -= shift_x # シフト量を残差から減算して更新
        self.shift_residual_y -= shift_y

        # Map shift(配列をシフトして更新) roll関数なので平行移動
        self.logodds = np.roll(self.logodds, -shift_x, axis=1)
        self.logodds = np.roll(self.logodds, -shift_y, axis=0)
        self.dynamic_count = np.roll(self.dynamic_count, -shift_x, axis=1)
        self.dynamic_count = np.roll(self.dynamic_count, -shift_y, axis=0)

        # clear edge roll関数は循環しているため、シフト後の端の部分を0でクリアする必要がある
        if shift_x > 0:
            self.logodds[:, -shift_x:] = 0
            self.dynamic_count[:, -shift_x:] = 0
        elif shift_x < 0:
            self.logodds[:, :-shift_x] = 0
            self.dynamic_count[:, :-shift_x] = 0
        if shift_y > 0:
            self.logodds[-shift_y:, :] = 0
            self.dynamic_count[-shift_y:, :] = 0
        elif shift_y < 0:
            self.logodds[:-shift_y, :] = 0
            self.dynamic_count[:-shift_y, :] = 0

        # Polar bin 各点の角度を計算してビンに割り当てる(raycastingのため) 720ビンで-180°から180°までをカバー 0.5°刻み
        bin_count = self.bin_count
        angle_polar = np.arctan2(middle_y_local, middle_x_local)
        angle_bins = np.linspace(-np.pi, np.pi, bin_count)
        
        # Grid
        gx_occ = np.round((middle_y_local + self.MAP_RANGE)* self.grid_pixel).astype(np.int32)
        gy_occ = np.round((middle_x_local + self.MAP_RANGE)* self.grid_pixel).astype(np.int32)
        gx_ray = np.round((middle_y_local + self.RAY_RANGE)* self.grid_pixel).astype(np.int32)
        gy_ray = np.round((middle_x_local + self.RAY_RANGE)* self.grid_pixel).astype(np.int32)
        dist = np.sqrt(middle_x_local**2 + middle_y_local**2)
        occ_mask = dist < self.MAP_RANGE
        ray_mask = dist < self.RAY_RANGE

        # occupancy update用
        gx_occ = gx_occ[occ_mask]
        gy_occ = gy_occ[occ_mask]
        dist_occ = dist[occ_mask]
        angle_occ = angle_polar[occ_mask]

        # raycasting用
        gx_ray = gx_ray[ray_mask]
        gy_ray = gy_ray[ray_mask]
        dist_ray = dist[ray_mask]
        angle_ray = angle_polar[ray_mask]

        valid_ray = ((gx_ray >= 0) & (gx_ray < self.ray_size) & (gy_ray >= 0) & (gy_ray < self.ray_size))
        valid = ((gx_occ >= 0) & (gx_occ < self.size) & (gy_occ >= 0) & (gy_occ < self.size))

        gx_ray = gx_ray[valid_ray]
        gy_ray = gy_ray[valid_ray]
        dist_ray = dist_ray[valid_ray]
        angle_ray = angle_ray[valid_ray]

        middle_x = middle_x[occ_mask][valid]
        middle_y = middle_y[occ_mask][valid]
        middle_z = middle_z[occ_mask][valid]
        middle_intensity = middle_intensity[occ_mask][valid]
        
        # Sensor Origin
        x0 = int(self.MAP_RANGE * self.grid_pixel)
        y0 = int(self.MAP_RANGE * self.grid_pixel)

        # angle -> bin
        bin_idx = ((angle_ray + np.pi) / (2*np.pi) * bin_count).astype(np.int32)

        # 各binで最短距離indexを取得
        sort_idx = np.argsort(dist_ray)
        bin_sorted = bin_idx[sort_idx]

        # unique bin の最初 = 最短距離
        _, unique_first = np.unique(bin_sorted, return_index=True)
        nearest_idx = sort_idx[unique_first]

        # nearest point (0~50m)
        ray_dist_all = dist_ray[nearest_idx]
        ray_angle_all = angle_ray[nearest_idx]

        # occupancy更新は15m以内だけ
        occ_valid = ray_dist_all < self.MAP_RANGE

        # occupancy用
        ray_dist_occ = ray_dist_all[occ_valid]
        ray_angle_occ = ray_angle_all[occ_valid]

        # occupied cell
        ray_occ_x = ray_dist_all[occ_valid] * np.cos(ray_angle_all[occ_valid])
        ray_occ_y = ray_dist_all[occ_valid] * np.sin(ray_angle_all[occ_valid])
        ray_occ_gx = np.round((ray_occ_y + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        ray_occ_gy = np.round((ray_occ_x + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)

        # =====================================================
        # Vectorized Ray Casting
        # =====================================================

        # ray方向ベクトル
        ray_dx = np.cos(ray_angle_all)
        ray_dy = np.sin(ray_angle_all)

        # 各rayの停止距離
        # ・15m以内 → obstacleまで
        # ・15~50m → 15mまで
        ray_limit = np.minimum(ray_dist_all, self.RAY_RANGE)

        # resolution[m]
        step = 1.0 / self.grid_pixel

        # ray上の距離列
        t = np.arange(0.0, self.RAY_RANGE, step)

        # =====================================================
        # 全ray座標生成
        # shape:
        #   [num_steps, num_rays]
        # =====================================================
        xs = t[:, None] * ray_dx[None, :]
        ys = t[:, None] * ray_dy[None, :]

        # =====================================================
        # ray limit判定
        # 各rayで obstacle より手前だけTrue
        # =====================================================
        valid_ray = t[:, None] < ray_limit[None, :]

        # =====================================================
        # map index変換
        # =====================================================
        gx = np.round((ys + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        gy = np.round((xs + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)

        # =====================================================
        # map範囲チェック
        # =====================================================
        valid_map = ((gx >= 0) & (gx < self.size) &(gy >= 0) & (gy < self.size))

        # =====================================================
        # 最終valid
        # =====================================================
        free_valid = valid_ray & valid_map

        # =====================================================
        # free mask生成
        # =====================================================
        free_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        free_mask[gx[free_valid], gy[free_valid]] = 1

        # =====================================================
        # occupied cell除外
        # =====================================================
        occ_valid = ray_dist_all < self.MAP_RANGE
        occ_x = ray_dist_occ * np.cos(ray_angle_occ)
        occ_y = ray_dist_occ * np.sin(ray_angle_occ)
        occ_gx = np.round((occ_y + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        occ_gy = np.round((occ_x + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        valid_occ = ((occ_gx >= 0) & (occ_gx < self.size) & (occ_gy >= 0) & (occ_gy < self.size))
        occ_gx = occ_gx[valid_occ]
        occ_gy = occ_gy[valid_occ]
        free_mask[occ_gx, occ_gy] = 0

        # =====================================================
        # log odds update
        # =====================================================
        # FREE
        self.logodds[free_mask > 0] -= 0.4
        # OCCUPIED
        self.logodds[occ_gx, occ_gy] += 0.8 # 0.6

        # logodds
        self.logodds = np.clip(self.logodds, -5.0, 5.0) # 対数オッズの値を-5から5の範囲に制限することで、確率が極端な値になるのを防止
        prob_log = 1.0 / (1.0 + np.exp(-self.logodds)) # 対数オッズを確率に変換するためのシグモイド関数を適用して、各セルの占有確率を計算
        observed_log = np.abs(self.logodds) > 0.1 # 対数オッズの絶対値が0.1より大きいセルを観測されたセルとみなすためのマスクを作成
        occ_log = (prob_log * 100).astype(np.int8) # 対数オッズから計算された確率を0から100の整数に変換して、占有グリッドマップの値として使用
        occ_log[~observed_log] = -1 # 観測されていないセルは-1に設定して、占有グリッドマップで未観測セルを区別できるようにする

        static_mask = ((prob_log > 0.0)) # & (score > 0.0)) # hit - miss
        #static_mask = (prob_log > 0.65) # log
        #static_mask = (self.static_score > 30.0) 
        dynamic_mask = ((prob_log < 0.25)) # log (prob_log > 0.00) & 
        #dynamic_mask = ((prob > 0.00) & (prob < 0.07)) # average
        self.dynamic_count[dynamic_mask] += 1 # 動的と判断されたセルのカウンタを増加させる
        self.dynamic_count[~dynamic_mask] = 0 # 動的でないと判断されたセルのカウンタをリセットする
        stable_dynamic = self.dynamic_count >= 1 # 動的と判断されたセルのカウンタが10以上のセルを安定した動的セルとみなすためのマスクを作成

        # OccupancyGrid
        static_occ = np.zeros_like(occ_log, dtype=np.int8)
        static_occ[static_mask] = 100

        dynamic_occ = np.zeros_like(occ_log, dtype=np.int8)
        dynamic_occ[stable_dynamic] = 100

        middle_points = np.vstack((middle_x, middle_y, middle_z, middle_intensity))

        print("ray",time.perf_counter()-t0)
        t2 = time.perf_counter()

        ############################  reflection ############################
        #ground global
        #ground_rot, ground_rot_matrix = rotation_xyz(ground_points[[0,1,2],:], theta_x, theta_y, theta_z)
        #ground_x_global = ground_rot[0,:] + position[0]
        #ground_y_global = ground_rot[1,:] + position[1]
        #ground_global = np.vstack((ground_x_global, ground_y_global, ground_rot[2,:], ground_points[3,:]) , dtype=np.float32)
        #ground_global_msg = do_transform_cloud(ground_msg, transform)
        #ground_x_global, ground_y_global, ground_z_global, ground_intensity_global = self.pointcloud2_to_array(ground_global_msg)
        #ground_global = np.vstack((ground_x_global, ground_y_global, ground_z_global, ground_intensity_global), dtype=np.float32)
        
        #middle global
        #middle_rot, middle_rot_matrix = rotation_xyz(middle_points[[0,1,2],:], theta_x, theta_y, theta_z)
        #middle_x_global = middle_rot[0,:] + position_x
        #middle_y_global = middle_rot[1,:] + position_y
        #middle_global = np.vstack((middle_x_global, middle_y_global, middle_rot[2,:], middle_points[3,:]) , dtype=np.float32)
        #middle_global_msg = do_transform_cloud(middle_msg, transform)
        #middle_x_global, middle_y_global, middle_z_global, middle_intensity_global = self.pointcloud2_to_array(middle_global_msg)
        #middle_global = np.vstack((middle_x_global, middle_y_global, middle_z_global, middle_intensity_global), dtype=np.float32)
        
        #high global
        #high_rot, high_rot_matrix = rotation_xyz(high_points[[0,1,2],:], theta_x, theta_y, theta_z)
        #high_x_grobal = high_rot[0,:] + position_x
        #high_y_grobal = high_rot[1,:] + position_y
        #high_global = np.vstack((high_x_grobal, high_y_grobal, high_rot[2,:], high_points[3,:]) , dtype=np.float32)
        #high_global_msg = do_transform_cloud(high_msg, transform)
        #high_x_global,  high_y_global, high_z_global, high_intensity_global = self.pointcloud2_to_array(high_global_msg)
        #high_global = np.vstack((high_x_global, high_y_global, high_z_global, high_intensity_global), dtype=np.float32)

        #map lim set
        map_lim_x_min = position_x + self.MAP_LIM_X_MIN;
        map_lim_x_max = position_x + self.MAP_LIM_X_MAX;
        map_lim_y_min = position_y + self.MAP_LIM_Y_MIN;
        map_lim_y_max = position_y + self.MAP_LIM_Y_MAX;
        map_lim_ground_ind = self.pcd_serch(self.pcd_ground_buff, map_lim_x_min, map_lim_x_max, map_lim_y_min, map_lim_y_max)
        map_lim_middle_ind = self.pcd_serch(self.pcd_middle_buff, map_lim_x_min, map_lim_x_max, map_lim_y_min, map_lim_y_max)
        map_lim_high_ind = self.pcd_serch(self.pcd_high_buff, map_lim_x_min, map_lim_x_max, map_lim_y_min, map_lim_y_max)
        self.pcd_ground_buff = self.pcd_ground_buff[:,map_lim_ground_ind]
        self.pcd_middle_buff= self.pcd_middle_buff[:,map_lim_middle_ind]
        self.pcd_high_buff = self.pcd_high_buff[:,map_lim_high_ind]
        
        #obs round&duplicated ground  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        #pcd_ground_buff = ground_global
        '''
        pcd_ground_buff = np.insert(self.pcd_ground_buff, len(self.pcd_ground_buff[0,:]), ground_global.T, axis=1)
        points_ground_round = np.round(pcd_ground_buff * self.ground_pixel) / self.ground_pixel
        self.pcd_ground_buff =points_ground_round[:,~pd.DataFrame({"x":points_ground_round[0,:], "y":points_ground_round[1,:], "z":points_ground_round[2,:]}).duplicated()]
        '''
        #obs round&duplicated middle  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        #pcd_middle_buff = middle_global
        '''
        pcd_middle_buff = np.insert(self.pcd_middle_buff, len(self.pcd_middle_buff[0,:]), middle_global.T, axis=1)
        points_middle_round = np.round(pcd_middle_buff * self.ground_pixel) / self.ground_pixel
        self.pcd_middle_buff =points_middle_round[:,~pd.DataFrame({"x":points_middle_round[0,:], "y":points_middle_round[1,:], "z":points_middle_round[2,:]}).duplicated()]
        '''
        #obs round&duplicated high  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        #pcd_high_buff = high_global
        '''
        pcd_high_buff = np.insert(self.pcd_high_buff, len(self.pcd_high_buff[0,:]), high_global.T, axis=1)
        points_high_round = np.round(pcd_high_buff * self.ground_pixel) / self.ground_pixel
        self.pcd_high_buff =points_high_round[:,~pd.DataFrame({"x":points_high_round[0,:], "y":points_high_round[1,:], "z":points_high_round[2,:]}).duplicated()]
        '''
        if is_keyframe:
            self.pcd_ground_buff = self.append_unique_points(self.pcd_ground_buff, ground_global)
            self.pcd_middle_buff = self.append_unique_points(self.pcd_middle_buff, middle_global)
            self.pcd_high_buff = self.append_unique_points(self.pcd_high_buff, high_global)
            # keyframe位置を更新
            self.last_keyframe_x = position_x
            self.last_keyframe_y = position_y
            self.keyframe_initialized = True
        print("duplicated",time.perf_counter()-t2)
        
        # remove dynamic obs
        t3=time.perf_counter()
        middle_local_x = self.pcd_middle_buff[0,:] - position_x
        middle_local_y = self.pcd_middle_buff[1,:] - position_y
        middle_global_x = np.round((middle_local_y + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        middle_global_y = np.round((middle_local_x + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        middle_valid = ((middle_global_x >= 0) & (middle_global_x < self.size) & (middle_global_y >= 0) & (middle_global_y < self.size))
        high_local_x = self.pcd_high_buff[0,:] - position_x
        high_local_y = self.pcd_high_buff[1,:] - position_y
        high_global_x = np.round((high_local_y + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        high_global_y = np.round((high_local_x + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        high_valid = ((high_global_x >= 0) & (high_global_x < self.size) & (high_global_y >= 0) & (high_global_y < self.size))

        remove_middle_mask = np.zeros(len(middle_global_x), dtype=bool)
        remove_middle_mask[middle_valid] = (dynamic_occ[middle_global_x[middle_valid], middle_global_y[middle_valid]] > 0)
        removed_middle_points = self.pcd_middle_buff[:, remove_middle_mask]
        #remove_high_mask = np.zeros(len(high_global_x), dtype=bool)
        #remove_high_mask[high_valid] = (dynamic_occ[high_global_x[high_valid], high_global_y[high_valid]] > 0)
        self.pcd_middle_buff = self.pcd_middle_buff[:, ~remove_middle_mask]
        #self.pcd_high_buff = self.pcd_high_buff[:, ~remove_high_mask]
        
        if removed_middle_points.shape[1] > 0:
            # removeされたmiddle点をgrid化
            #remove_x = np.round(removed_middle_points[0,:] * self.ground_pixel).astype(np.int32)
            #remove_y = np.round(removed_middle_points[1,:] * self.ground_pixel).astype(np.int32)
            #remove_cells = set(zip(remove_x, remove_y))
            remove_x = np.round(removed_middle_points[0, :] * self.ground_pixel).astype(np.int64)
            remove_y = np.round(removed_middle_points[1, :] * self.ground_pixel).astype(np.int64)
            KEY_SCALE = 100000
            remove_key = ((remove_x + KEY_SCALE) * 1000000 + (remove_y + KEY_SCALE))

            # ground点をgrid化
            #ground_x = np.round(self.pcd_ground_buff[0,:] * self.ground_pixel).astype(np.int32)
            #ground_y = np.round(self.pcd_ground_buff[1,:] * self.ground_pixel).astype(np.int32)
            ground_x = np.round(self.pcd_ground_buff[0, :] * self.ground_pixel).astype(np.int64)
            ground_y = np.round(self.pcd_ground_buff[1, :] * self.ground_pixel).astype(np.int64)
            ground_key = ((ground_x + KEY_SCALE) * 1000000 + (ground_y + KEY_SCALE))

            # high点をgrid化
            #high_x = np.round(self.pcd_high_buff[0,:] * self.ground_pixel).astype(np.int32)
            #high_y = np.round(self.pcd_high_buff[1,:] * self.ground_pixel).astype(np.int32)

            # ground削除mask
            #ground_remove_mask = np.array([(middle_global_x, middle_global_y) in remove_cells for middle_global_x, middle_global_y in zip(ground_x, ground_y)])
            #self.pcd_ground_buff = self.pcd_ground_buff[:, ~ground_remove_mask]
            ground_remove_mask = np.isin(ground_key, remove_key)
            self.pcd_ground_buff = self.pcd_ground_buff[:, ~ground_remove_mask]

            # high削除mask
            #high_remove_mask = np.array([(middle_global_x, middle_global_y) in remove_cells for middle_global_x, middle_global_y in zip(high_x, high_y)])
            #self.pcd_high_buff = self.pcd_high_buff[:, ~high_remove_mask]
        print("remove points",time.perf_counter()-t3)
        #self.pcd_middle_buff = self.remove_dynamic_points(self.pcd_middle_buff, middle_dynamic_global, radius=0.25)
        #local reflect ground map
        t1=time.perf_counter()
        ground_reflect_conv = self.pcd_ground_buff[3,:]/255*100.0
        map_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        map_data_ground_set = grid_map_set(self.pcd_ground_buff[1,:], self.pcd_ground_buff[0,:], ground_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)

        #local reflect middle map
        '''
        if self.pcd_middle_buff.shape[1] > 0:
            middle_max_intensity = np.max(self.pcd_middle_buff[3,:])
            if middle_max_intensity > 0:
                middle_reflect_conv = np.clip(self.pcd_middle_buff[3,:] / 255 * 100.0, 0, 100).astype(np.int8)
            else:
                middle_reflect_conv = np.zeros(self.pcd_middle_buff.shape[1], dtype=np.int8)
        else:
            middle_reflect_conv = np.zeros(0, dtype=np.int8)
        '''
        if self.pcd_middle_buff.shape[1] > 0:
            middle_reflect_conv = np.full(self.pcd_middle_buff.shape[1], 100, dtype=np.uint8)
        else:
            middle_reflect_conv = np.zeros(0, dtype=np.uint8)
        #middle_reflect_conv = np.clip(self.pcd_middle_buff[3,:] / np.max(self.pcd_middle_buff[3,:]) * 100.0, 0, 100).astype(np.int8)
        #middle_reflect_conv = self.pcd_middle_buff[3,:]/255*100.0
        map_data_middle_set = grid_map_set(self.pcd_middle_buff[1,:], self.pcd_middle_buff[0,:], middle_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)

        #local reflect high map
        #high_reflect_conv = self.pcd_high_buff[3,:]/255*100.0
        '''
        if self.pcd_high_buff.shape[1] > 0:
            high_max_intensity = np.max(self.pcd_high_buff[3,:])
            if high_max_intensity > 0:
                high_reflect_conv = np.clip(self.pcd_high_buff[3,:] / 255 * 100.0, 0, 100).astype(np.int8)
            else:
                high_reflect_conv = np.zeros(self.pcd_high_buff.shape[1], dtype=np.int8)
        else:
            high_reflect_conv = np.zeros(0, dtype=np.int8)
        '''
        if self.pcd_high_buff.shape[1] > 0:
            high_reflect_conv = np.full(self.pcd_high_buff.shape[1], 100, dtype=np.uint8)
        else:
            high_reflect_conv = np.zeros(0, dtype=np.uint8)

        if self.pcd_high_buff.shape[1] == 0:
            map_data_high_set = np.zeros((self.size, self.size), dtype=np.uint8)
        else:
            map_data_high_set = grid_map_set(self.pcd_high_buff[1,:], self.pcd_high_buff[0,:], high_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)
        
        print("local_grid", time.perf_counter()-t1)

        ##ekf pos ground local reflect map
        t5=time.perf_counter()
        '''
        ekf_ground_buff_x = self.pcd_ground_buff[0,:] - position[0]
        ekf_ground_buff_y = self.pcd_ground_buff[1,:] - position[1]
        ekf_ground_buff_z = self.pcd_ground_buff[2,:] - position[2]
        ekf_ground_buff = np.vstack((ekf_ground_buff_x, ekf_ground_buff_y, ekf_ground_buff_z))
        ekf_ground_rot, ekf_ground_rot_matrix = rotation_xyz(ekf_ground_buff, ekf_theta_x-theta_x, ekf_theta_y-theta_y, ekf_theta_z-theta_z)
        ekf_ground_set_x = ekf_ground_rot[0,:] + ekf_position[0]
        ekf_ground_set_y = ekf_ground_rot[1,:] + ekf_position[1]
        ekf_ground_set_z = ekf_ground_rot[2,:] + ekf_position[2]
        ekf_ground_set = np.vstack((ekf_ground_set_x, ekf_ground_set_y, ekf_ground_set_z))
        '''
        #map_data_set_4save = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE)
        #print(f"map_data_ground_set ={map_data_ground_set.shape}")

        ##ekf pos middle local reflect map
        '''
        ekf_middle_buff_x = self.pcd_middle_buff[0,:] - position[0]
        ekf_middle_buff_y = self.pcd_middle_buff[1,:] - position[1]
        ekf_middle_buff_z = self.pcd_middle_buff[2,:] - position[2]
        ekf_middle_buff = np.vstack((ekf_middle_buff_x, ekf_middle_buff_y, ekf_middle_buff_z))
        ekf_middle_rot, ekf_middle_rot_matrix = rotation_xyz(ekf_middle_buff, ekf_theta_x-theta_x, ekf_theta_y-theta_y, ekf_theta_z-theta_z)
        ekf_middle_set_x = ekf_middle_rot[0,:] + ekf_position[0]
        ekf_middle_set_y = ekf_middle_rot[1,:] + ekf_position[1]
        ekf_middle_set_z = ekf_middle_rot[2,:] + ekf_position[2]
        ekf_middle_set = np.vstack((ekf_middle_set_x, ekf_middle_set_y, ekf_middle_set_z))
        '''
        #map_data_set_4save = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE)
        #print(f"map_data_middle_set ={map_data_middle_set.shape}")

        ##ekf pos high local reflect map
        '''
        ekf_high_buff_x = self.pcd_high_buff[0,:] - position[0]
        ekf_high_buff_y = self.pcd_high_buff[1,:] - position[1]
        ekf_high_buff_z = self.pcd_high_buff[2,:] - position[2]
        ekf_high_buff = np.vstack((ekf_high_buff_x, ekf_high_buff_y, ekf_high_buff_z))
        ekf_high_rot, ekf_high_rot_matrix = rotation_xyz(ekf_high_buff, ekf_theta_x-theta_x, ekf_theta_y-theta_y, ekf_theta_z-theta_z)
        ekf_high_set_x = ekf_high_rot[0,:] + ekf_position[0]
        ekf_high_set_y = ekf_high_rot[1,:] + ekf_position[1]
        ekf_high_set_z = ekf_high_rot[2,:] + ekf_position[2]
        ekf_high_set = np.vstack((ekf_high_set_x, ekf_high_set_y, ekf_high_set_z))
        '''
        #map_data_set_4save = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE)
        #print(f"map_data_high_set ={map_data_high_set.shape}")
        print("ekf rotation", time.perf_counter()-t5)
	
        #GL reflect map
        #map_data_gl_set = grid_map_set(self.pcd_ground_buff[1,:], self.pcd_ground_buff[0,:], ground_reflect_conv, position, self.ground_pixel, self.MAP_RANGE_GL)
        t2=time.perf_counter()
        #map_data_ground_gl_set = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        #map_data_middle_gl_set = grid_map_set(ekf_middle_set[1,:], ekf_middle_set[0,:], middle_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        #map_data_high_gl_set = grid_map_set(ekf_high_set[1,:], ekf_high_set[0,:], high_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        #print("global_grid", time.perf_counter()-t2)
        #print("total", time.perf_counter()-t0)

        #if self.pcd_high_buff.shape[1] == 0:
        #    map_data_high_gl_set = np.zeros((self.size, self.size), dtype=np.uint8)
        #else:
        #    map_data_high_gl_set = grid_map_set(ekf_high_set[1,:], ekf_high_set[0,:], high_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        #print(f"map_data_ground_set ={map_data_ground_set.shape}")
	
        
        #publish for rviz2 
        #global ground
        t6=time.perf_counter()
        #ground_global_msg = point_cloud_intensity_msg(self.pcd_ground_buff.T, t_stamp, 'odom')
        #self.pcd_ground_global_publisher.publish(ground_global_msg) 
        print("pcd ground pub", time.perf_counter()-t6)
        t4=time.perf_counter()
        '''
        #local map ground
        self.map_data_ground = make_map_msg(map_data_ground_set, self.ground_pixel, position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        #local map middle
        self.map_data_middle = make_map_msg(map_data_middle_set, self.ground_pixel, position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        #local map high
        self.map_data_high = make_map_msg(map_data_high_set, self.ground_pixel, position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        #self.map_data = make_map_msg(map_data_set_4save, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        self.map_data_flag = 1
        #self.reflect_map_local_publisher.publish(self.map_data)     
        #gl map ground
        self.map_data_ground_gl = make_map_msg(map_data_ground_gl_set, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE_GL, "odom")
        #gl map middle
        self.map_data_middle_gl = make_map_msg(map_data_middle_gl_set, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE_GL, "odom")
        #gl map high
        self.map_data_high_gl = make_map_msg(map_data_high_gl_set, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE_GL, "odom")
        '''
        #print("make_map_msg", time.perf_counter()-t4)
        #self.reflect_map_global_publisher.publish(self.map_data_gl) 
        self.map_data_gl_flag = 1
        rgb_local = self.make_rgb_map(map_data_ground_set, map_data_middle_set, map_data_high_set)
        rgb_local_msg = self.bridge.cv2_to_imgmsg(rgb_local, encoding='bgr8')
        rgb_local_msg.header.stamp = t_stamp
        rgb_local_msg.header.frame_id = "odom"
        self.rgb_map_local_pub.publish(rgb_local_msg)
        #rgb_global = self.make_rgb_map(map_data_ground_gl_set, map_data_middle_gl_set, map_data_high_gl_set)
        #rgb_global_msg = self.bridge.cv2_to_imgmsg(rgb_global, encoding='bgr8')
        #rgb_global_msg.header.stamp = t_stamp
        #rgb_global_msg.header.frame_id = "odom"
        #self.rgb_map_global_pub.publish(rgb_global_msg)
        #print("total", time.perf_counter()-t0)
        print("rgbmap", time.perf_counter()-t4)
        
        if self.MAKE_GL_MAP_FLAG == 1:
            map_pos_diff = math.sqrt((ekf_position_x - self.map_position_x_buff)**2 + (ekf_position_y - self.map_position_y_buff)**2)
            map_theta_diff = abs(ekf_theta_z -  self.map_theta_z_buff)
            
            if ( (map_pos_diff > 5) or ((map_pos_diff > 2) and (map_theta_diff > 40)) ):
                self.save_flag = 1
            else:
                self.save_flag = 0
            #self.make_ref_map(position_x, position_y, theta_z)
            #self.make_ref_map(ekf_position_x, ekf_position_y, ekf_theta_z)

            if self.save_flag == 1:
                #self.make_ref_map(map_data_ground_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z, layer="ground")
                #self.make_ref_map(map_data_middle_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z, layer="middle")
                #self.make_ref_map(map_data_high_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z, layer="high")
                # RGB PNG保存
                self.save_rgb_map(map_data_ground_gl_set, map_data_middle_gl_set, map_data_high_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z)
                self.map_position_x_buff = ekf_position_x
                self.map_position_y_buff = ekf_position_y
                self.map_theta_z_buff = ekf_theta_z
                self.map_number += 1
        
        #remove_vis = np.zeros((self.size,self.size), dtype=np.int8)
        #remove_vis[gx[remove_mask], gy[remove_mask]] = 100
        #remove_vis[dynamic_occ > 0] = 20
        #remove_vis[middle_global_x[remove_middle_mask], middle_global_y[remove_middle_mask]] = 100
        '''
        # Publish
        grid = OccupancyGrid()
        grid.header = middle_msg.header
        grid.header.frame_id = "odom"
        grid.info.resolution = 1.0 / self.grid_pixel
        grid.info.width = self.size
        grid.info.height = self.size
        grid.info.origin.position.x = (self.position_x - self.MAP_RANGE)
        grid.info.origin.position.y = (self.position_y - self.MAP_RANGE)
        grid.info.origin.orientation.w = 1.0        

        static_grid = OccupancyGrid()
        static_grid.header = grid.header
        static_grid.info = grid.info
        static_grid.data = static_occ.flatten().tolist()
        #self.static_map_pub.publish(static_grid)

        dynamic_grid = OccupancyGrid()
        dynamic_grid.header = grid.header
        dynamic_grid.info = grid.info
        dynamic_grid.data = dynamic_occ.flatten().tolist()
        self.dynamic_map_pub.publish(dynamic_grid)

        log_grid = OccupancyGrid()
        log_grid.header = grid.header
        log_grid.info = grid.info
        log_grid.data = occ_log.flatten().tolist()
        self.log_map_pub.publish(log_grid)

        remove_grid = OccupancyGrid()
        remove_grid.header = grid.header
        remove_grid.info = grid.info
        remove_grid.data = remove_vis.flatten().tolist()
        self.remove_map_pub.publish(remove_grid)
        '''
        # Update previous pose
        self.prev_x = self.position_x
        self.prev_y = self.position_y
        
        self.start_flag = 1

        dt = time.perf_counter() - t0
        print(f"reflect_map {dt*1000:.1f} ms")

        #print("ray_dist_all max =", np.max(ray_dist_all))
        #print("ray_dist_all len =", len(ray_dist_all))
        #print("over15 count =", np.sum(ray_dist_all > 15.0))
        #print("over30 count =", np.sum(ray_dist_all > 30.0))
        #print("middle dist max =", np.max(dist))
        #print("NaN dist =", np.sum(np.isnan(dist_ray)))
        #print("INF dist =", np.sum(np.isinf(dist_ray)))
        
    def append_unique_points(self, buff, new_points):
        if new_points.shape[1] == 0:
            return buff

        # 1. 新規点を0.05 mグリッドへ変換
        new_grid = np.round(new_points[:3, :] * self.ground_pixel).astype(np.int32)

        # 2. x,y,zを1個のint64 keyへ変換
        offset = 100000
        new_key = ((new_grid[0].astype(np.int64) + offset) * 10000000000 + (new_grid[1].astype(np.int64) + offset) * 100000 + (new_grid[2].astype(np.int64) + offset))

        # 3. 新規点だけで重複除去
        _, unique_idx = np.unique(new_key, return_index=True)
        new_points = new_points[:, unique_idx]
        new_key = new_key[unique_idx]

        # 4. バッファが空ならそのまま
        if buff.shape[1] == 0:
            return new_points

        # 5. 既存バッファのkey
        buff_grid = np.round(buff[:3, :] * self.ground_pixel).astype(np.int32)

        buff_key = ((buff_grid[0].astype(np.int64) + offset) * 10000000000 + (buff_grid[1].astype(np.int64) + offset) * 100000 + (buff_grid[2].astype(np.int64) + offset))

        # 6. 新規点が既存バッファに存在するか確認
        exists = np.isin(new_key, buff_key)
        new_points = new_points[:, ~exists]

        # 7. 追加する点がなければ終了
        if new_points.shape[1] == 0:
            return buff

        # 8. バッファへ追加
        return np.hstack((buff, new_points))

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
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ))
        return pcd_ind
    
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
def grid_map_set(map_x, map_y, data, position, map_pixel, map_range):
    size = int(2 * map_range * map_pixel)
    # ==================================================
    # map_x → 行方向
    # map_y → 列方向
    # ==================================================
    row = np.round((map_x - position[1] + map_range) * map_pixel).astype(np.int32)
    col = np.round((map_y - position[0] + map_range) * map_pixel).astype(np.int32)
    valid = ((row > 0) & (row < size - 1) & (col > 0) &(col < size - 1))
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[row[valid], col[valid]] = data[valid]

    return np.flipud(grid)
###################################################################

def main():
    rclpy.init()
    node = ObsBayesMap()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
