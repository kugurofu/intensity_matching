# ROS2
import rclpy
from rclpy.node import Node
# ROS msgs
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int8MultiArray
# Python
import numpy as np
import math
import cv2
import yaml
import os
from collections import OrderedDict
import pandas as pd

class ObsBayesMap(Node):
    def __init__(self):
        super().__init__('obs_bayes_map')
        # Subscriber
        self.pcd_ground_sub = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_ground', self.ground_points_callback, 1)
        self.pcd_obs_sub = self.create_subscription(sensor_msgs.PointCloud2,'/pcd_segment_middle',self.publish_map,1)
        self.local_odom_sub = self.create_subscription(nav_msgs.Odometry,'/fusion/odom',self.get_local_odom,1)
        self.global_odom_sub = self.create_subscription(nav_msgs.Odometry,'/fusion/odom', self.get_global_odom, 1)

        # Publisher
        self.bayes_map_pub = self.create_publisher(OccupancyGrid,'/obs_bayes_map',1)
        self.static_map_pub = self.create_publisher(OccupancyGrid,'/static_obs_map',1)
        self.dynamic_map_pub = self.create_publisher(OccupancyGrid,'/dynamic_obs_map',1)
        self.log_map_pub = self.create_publisher(OccupancyGrid,'/log_map',1)
        self.pcd_ground_global_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_ground_global', 1) 
        self.reflect_map_ground_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_ground_local', 1)
        self.reflect_map_middle_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_middle_local', 1)
        self.reflect_map_ground_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_ground_global', 1)
        self.reflect_map_middle_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_middle_global', 1)
        self.reflect_map_dynamic_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_dynamic_global', 1)
        self.remove_map_pub = self.create_publisher(OccupancyGrid, 'remove_map', 1)

        #timer
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_flag = 0

        # Parameter
        self.grid_pixel = 1000 / 50.0
        self.MAP_RANGE = 15.0
        self.size = int(self.MAP_RANGE * 2 * self.grid_pixel)
        #ground 
        self.ground_pixel = self.grid_pixel
        #self.MAP_RANGE = 15.0 #[m]
        
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
        self.hit_score = np.zeros((self.size, self.size), dtype=np.float32)
        self.miss_score = np.zeros((self.size, self.size), dtype=np.float32)
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
        self.save_dir = os.path.expanduser('~/ros2_ws/src/map/test4_waypoint_map')
        yaml.add_representer(OrderedDict, ordered_dict_representer, Dumper=MyDumper)
        yaml.add_representer(list, list_representer, Dumper=MyDumper)

    def timer_callback(self):
        if self.start_flag == 0:
            return
        
        #self.reflect_map(self.t_stamp, self.middle_points, self.dynamic_occ)
        if self.map_data_flag > 0:
            self.reflect_map_ground_local_publisher.publish(self.map_data_ground)
            self.reflect_map_middle_local_publisher.publish(self.map_data_middle)
            #self.reflect_map_high_local_publisher.publish(self.map_data_high)
        #gl map
        if self.map_data_gl_flag > 0:
            self.reflect_map_ground_global_publisher.publish(self.map_data_ground_gl) 
            self.reflect_map_middle_global_publisher.publish(self.map_data_middle_gl) 
            #self.reflect_map_high_global_publisher.publish(self.map_data_high_gl) 
    
    def ground_points_callback(self, msg):
        x, y, z, intensity = self.pointcloud2_to_array(msg)
        points = np.vstack((x, y, z, intensity)) # need mask?
        self.ground_points = points

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

    def pointcloud2_to_array(self, msg):
        points = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        return x, y, z, intensity

    def occ_model(self, d):
        p_max = 0.95
        p_min = 0.45
        return 0.8 #p_min + ((p_max - p_min) * np.exp(-d / 8.0))

    def free_model(self, d):
        free_max = 0.90
        free_min = 0.50
        return 0.5 #free_min + ((free_max - free_min) * np.exp(-d / 10.0))
    
    def remove_dynamic_points(self, static_cloud, dynamic_cloud, radius=0.3):
        
        if static_cloud.shape[1] == 0:
            return static_cloud

        if dynamic_cloud.shape[1] == 0:
            return static_cloud
        '''
        static_xy = static_cloud[0:2, :].T
        dynamic_xy = dynamic_cloud[0:2, :].T

        keep_mask = np.ones(len(static_xy), dtype=bool)

        for dyn_pt in dynamic_xy:
            dist = np.linalg.norm(static_xy - dyn_pt, axis=1)
            keep_mask &= (dist > radius)
        '''
        if dynamic_cloud.shape[1] > 0:
            dynamic_grid_x = np.round(dynamic_cloud[0,:] * self.ground_pixel).astype(np.int32)
            dynamic_grid_y = np.round(dynamic_cloud[1,:] * self.ground_pixel).astype(np.int32)
            # 周囲1セル含める
            dynamic_cells = set()

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cells = zip(dynamic_grid_x + dx, dynamic_grid_y + dy)
                    dynamic_cells.update(cells)

            static_grid_x = np.round(static_cloud[0,:] * self.ground_pixel).astype(np.int32)
            static_grid_y = np.round(static_cloud[1,:] * self.ground_pixel).astype(np.int32)

            keep_mask = np.array([(x, y) not in dynamic_cells for x, y in zip(static_grid_x, static_grid_y)])

            static_cloud = static_cloud[:, keep_mask]

        return static_cloud

    def publish_map(self, msg):
        t_stamp = msg.header.stamp
        self.t_stamp = t_stamp

        # PointCloud
        #points = self.pointcloud2_to_array(msg)
        x, y, z, intensity = self.pointcloud2_to_array(msg) # pointcloud2点群取得 numpy array
        theta = self.theta_z * np.pi / 180.0

        # LiDAR座標を車両座標に変換（回転のみ）
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        # Odometry取得
        position_x = self.position_x
        position_y = self.position_y
        prev_x = self.prev_x
        prev_y = self.prev_y

        # local map
        x_global = x_rot
        y_global = y_rot

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
        self.hit_score = np.roll(self.hit_score, -shift_x, axis=1)
        self.hit_score = np.roll(self.hit_score, -shift_y, axis=0)
        self.miss_score = np.roll(self.miss_score, -shift_x, axis=1)
        self.miss_score = np.roll(self.miss_score, -shift_y, axis=0)
        self.logodds = np.roll(self.logodds, -shift_x, axis=1)
        self.logodds = np.roll(self.logodds, -shift_y, axis=0)
        self.dynamic_count = np.roll(self.dynamic_count, -shift_x, axis=1)
        self.dynamic_count = np.roll(self.dynamic_count, -shift_y, axis=0)

        # clear edge roll関数は循環しているため、シフト後の端の部分を0でクリアする必要がある
        if shift_x > 0:
            self.hit_score[:, -shift_x:] = 0
            self.miss_score[:, -shift_x:] = 0
            self.logodds[:, -shift_x:] = 0
            self.dynamic_count[:, -shift_x:] = 0
        elif shift_x < 0:
            self.hit_score[:, :-shift_x] = 0
            self.miss_score[:, :-shift_x] = 0
            self.logodds[:, :-shift_x] = 0
            self.dynamic_count[:, :-shift_x] = 0
        if shift_y > 0:
            self.hit_score[-shift_y:, :] = 0
            self.miss_score[-shift_y:, :] = 0
            self.logodds[-shift_y:, :] = 0
            self.dynamic_count[-shift_y:, :] = 0
        elif shift_y < 0:
            self.hit_score[:-shift_y, :] = 0
            self.miss_score[:-shift_y, :] = 0
            self.logodds[:-shift_y, :] = 0
            self.dynamic_count[:-shift_y, :] = 0

        # Polar bin 各点の角度を計算してビンに割り当てる(raycastingのため) 720ビンで-180°から180°までをカバー 0.5°刻み
        angle_polar = np.arctan2(y_global, x_global)
        angle_bins = np.linspace(-np.pi, np.pi, 720)
        
        # Grid
        gx = np.round((y_global + self.MAP_RANGE)* self.grid_pixel).astype(np.int32)
        gy = np.round((x_global + self.MAP_RANGE)* self.grid_pixel).astype(np.int32)
        dist = np.sqrt(x_global**2 + y_global**2)
        mask = dist < 15.0

        gx = gx[mask]
        gy = gy[mask]
        dist = dist[mask]
        angle_polar = angle_polar[mask]

        valid = ((gx >= 0) & (gx < self.size) & (gy >= 0) & (gy < self.size))
        gx = gx[valid]
        gy = gy[valid]
        dist = dist[valid]
        angle_polar = angle_polar[valid]

        x = x[mask][valid]
        y = y[mask][valid]
        z = z[mask][valid]
        intensity = intensity[mask][valid]
        
        # Sensor Origin
        x0 = int(self.MAP_RANGE * self.grid_pixel)
        y0 = int(self.MAP_RANGE * self.grid_pixel)

        ray_points = {}

        for i in range(len(dist)): # 各点について、角度に基づいてビンを決定し、最も近い点を保存
            angle = angle_polar[i] # -piからpiの範囲の角度
            count = int((angle + np.pi) / (2*np.pi) * 720) # 0から719のビン番号
            d = dist[i] # 点の距離

            if count not in ray_points:
                ray_points[count] = (gx[i], gy[i], d) # 最初の点を保存
            elif d < ray_points[count][2]:
                ray_points[count] = (gx[i], gy[i], d) # より近い点が見つかった場合は更新

        # Ray casting
        max_range = 15.0 

        for count, (gx_i, gy_i, d) in ray_points.items(): # 各ビンについて、保存された点に対して占有と空きの更新を行う
            angle = angle_bins[count] # ビンの中心角度
            x1 = int(x0 + max_range * np.cos(angle) * self.grid_pixel) # センサからビンの方向に最大距離までの点の座標
            y1 = int(y0 + max_range * np.sin(angle) * self.grid_pixel) # センサからビンの方向に最大距離までの点の座標

            # OCCUPIED HIT
            if count in ray_points: 
                gx_i, gy_i, d = ray_points[count] # ビンに保存された点の座標と距離
                dx = gx_i - x0 # センサから点までのx方向の距離をピクセル単位で計算
                dy = gy_i - y0 # センサから点までのy方向の距離をピクセル単位で計算
                n = int(max(abs(dx), abs(dy))) # 点までの距離をピクセル単位で計算（xとyのどちらか大きい方を採用して線を引くため）
                xs = np.linspace(x0, gx_i, n).astype(np.int32) # センサから点までの線をn分割して、その座標を計算
                ys = np.linspace(y0, gy_i, n).astype(np.int32) # センサから点までの線をn分割して、その座標を計算

                # FREE UPDATE
                if n > 8: # 点までの距離が8ピクセル以上ある場合、点の手前の部分を空きとして更新（点の近くはノイズや誤検出の可能性があるため、最後の8ピクセルは空きとして扱わない）
                    xs_free = xs[:n-8] # 点の手前の部分のx座標
                    ys_free = ys[:n-8] # 点の手前の部分のy座標
                    valid_free = ((xs_free >= 0) & (xs_free < self.size) & (ys_free >= 0) & (ys_free < self.size)) # 空きとして更新する部分がマップの範囲内にあるかを確認するためのマスク
                    xs_free = xs_free[valid_free] # マップの範囲内にある部分のx座標
                    ys_free = ys_free[valid_free] # マップの範囲内にある部分のy座標

                    # 距離
                    k_arr = np.arange(n-8)[valid_free] # 点の手前の部分の距離をピクセル単位で計算（センサから線上の点までの距離をn分割して、そのうち点の手前の部分を採用）
                    d_free = k_arr / self.grid_pixel # 点の手前の部分の距離をメートル単位に変換

                    # confidence
                    free_conf = self.free_model(d_free)

                    # update
                    self.miss_score[xs_free, ys_free] += free_conf * 0.2
                    self.logodds[xs_free, ys_free] += -0.4 # 空きの更新は対数オッズを減少させる方向に更新

                # OCCUPIED UPDATE
                occ_conf = self.occ_model(d)
                self.hit_score[gx_i, gy_i] += occ_conf * 0.1
                self.logodds[gx_i, gy_i] += 0.6 # 占有の更新は対数オッズを増加させる方向に更新

        # Probability
        total = (self.hit_score + self.miss_score)
        score = (self.hit_score - self.miss_score)
        prob = np.zeros_like(total, dtype=np.float32)
        observed = total > 0.01
        prob[observed] = ((score[observed] / total[observed]) + 1.0) * 0.5

        # logodds
        self.logodds = np.clip(self.logodds, -5.0, 5.0) # 対数オッズの値を-5から5の範囲に制限することで、確率が極端な値になるのを防止
        prob_log = 1.0 / (1.0 + np.exp(-self.logodds)) # 対数オッズを確率に変換するためのシグモイド関数を適用して、各セルの占有確率を計算
        observed_log = np.abs(self.logodds) > 0.1 # 対数オッズの絶対値が0.1より大きいセルを観測されたセルとみなすためのマスクを作成
        occ_log = (prob_log * 100).astype(np.int8) # 対数オッズから計算された確率を0から100の整数に変換して、占有グリッドマップの値として使用
        occ_log[~observed_log] = -1 # 観測されていないセルは-1に設定して、占有グリッドマップで未観測セルを区別できるようにする

        static_mask = ((prob_log > 0.0)) # & (score > 0.0)) # hit - miss
        #static_mask = (prob_log > 0.65) # log
        #static_mask = (self.static_score > 30.0) 
        dynamic_mask = ((prob_log > 0.00) & (prob_log < 0.15)) # log
        #dynamic_mask = ((prob > 0.00) & (prob < 0.07)) # average
        self.dynamic_count[dynamic_mask] += 1 # 動的と判断されたセルのカウンタを増加させる
        self.dynamic_count[~dynamic_mask] = 0 # 動的でないと判断されたセルのカウンタをリセットする

        stable_dynamic = self.dynamic_count >= 10 # 動的と判断されたセルのカウンタが10以上のセルを安定した動的セルとみなすためのマスクを作成

        # OccupancyGrid
        occ = (prob * 100).astype(np.int8)
        occ[~observed] = -1

        #static_occ = np.full_like(occ, -1, dtype=np.int8)
        static_occ = np.zeros_like(occ, dtype=np.int8)
        #static_occ[prob_log < 0.2] = 0
        static_occ[static_mask] = 100

        #dynamic_occ = np.full_like(occ, -1, dtype=np.int8)
        dynamic_occ = np.zeros_like(occ_log, dtype=np.int8)
        #dynamic_occ[prob_log < 0.3] = 0
        dynamic_occ[stable_dynamic] = 100

        '''
        static_points_mask = static_mask[gx, gy]
        static_x = x[static_points_mask]
        static_y = y[static_points_mask]
        static_z = z[static_points_mask]
        static_intensity = intensity[static_points_mask]
        if len(static_x) > 0:
            static_points = np.vstack((static_x, static_y, static_z, static_intensity)).astype(np.float32)
        else:
            static_points = np.zeros((4, 0), dtype=np.float32)
        #static_points = np.array(static_points, dtype=np.float32).T
        #self.middle_points = static_points
        '''

        points = np.vstack((x, y, z, intensity))
        middle_points = points
        ground_points = self.ground_points
        '''
        dynamic_points_mask = dynamic_mask[gx, gy]
        dynamic_x = x[dynamic_points_mask]
        dynamic_y = y[dynamic_points_mask]
        dynamic_z = z[dynamic_points_mask]
        dynamic_intensity = intensity[dynamic_points_mask]
        if len(dynamic_x) > 0:
            dynamic_points = np.vstack((dynamic_x, dynamic_y, dynamic_z, dynamic_intensity)).astype(np.float32)
        else:
            dynamic_points = np.zeros((4, 0), dtype=np.float32)
        #static_points = np.array(static_points, dtype=np.float32).T
        #self.middle_points_dynamic = dynamic_points
        '''

        # Publish
        grid = OccupancyGrid()
        grid.header = msg.header
        grid.header.frame_id = "odom"
        grid.info.resolution = 1.0 / self.grid_pixel
        grid.info.width = self.size
        grid.info.height = self.size
        grid.info.origin.position.x = (self.position_x - self.MAP_RANGE)
        grid.info.origin.position.y = (self.position_y - self.MAP_RANGE)
        grid.info.origin.orientation.w = 1.0
        grid.data = occ.flatten().tolist()
        self.bayes_map_pub.publish(grid)

        static_grid = OccupancyGrid()
        static_grid.header = grid.header
        static_grid.info = grid.info
        static_grid.data = static_occ.flatten().tolist()
        self.static_map_pub.publish(static_grid)

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

        #self.dynamic_occ = dynamic_occ.copy()
        self.header = grid.header

        self.reflect_map(self.t_stamp, ground_points, middle_points, dynamic_occ)

        # Update previous pose
        self.prev_x = self.position_x
        self.prev_y = self.position_y

        self.start_flag = 1
 
    ################# original reflection intensity map #################
    def reflect_map(self, t_stamp, ground_points, middle_points, dynamic_occ): #(self, t_stamp, ground_points, middle_points, high_points)
        #print stamp message
        #t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        #points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        #position set
        position_x=self.position_x; position_y=self.position_y; position_z=self.position_z;
        position = np.array([position_x, position_y, position_z])
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        ekf_position_x=self.ekf_position_x; ekf_position_y=self.ekf_position_y; ekf_position_z=self.ekf_position_z;
        ekf_position = np.array([ekf_position_x, ekf_position_y, ekf_position_z])
        ekf_theta_x=self.ekf_theta_x; ekf_theta_y=self.ekf_theta_y; ekf_theta_z=self.ekf_theta_z;
        #ground global
        ground_rot, ground_rot_matrix = rotation_xyz(ground_points[[0,1,2],:], theta_x, theta_y, theta_z)
        ground_x_grobal = ground_rot[0,:] + position_x
        ground_y_grobal = ground_rot[1,:] + position_y
        ground_global = np.vstack((ground_x_grobal, ground_y_grobal, ground_rot[2,:], ground_points[3,:]) , dtype=np.float32)
        
        #middle global
        middle_rot, middle_rot_matrix = rotation_xyz(middle_points[[0,1,2],:], theta_x, theta_y, theta_z)
        middle_x_grobal = middle_rot[0,:] + position_x
        middle_y_grobal = middle_rot[1,:] + position_y
        middle_global = np.vstack((middle_x_grobal, middle_y_grobal, middle_rot[2,:], middle_points[3,:]) , dtype=np.float32)
        
        #high global
        '''
        high_rot, high_rot_matrix = rotation_xyz(high_points[[0,1,2],:], theta_x, theta_y, theta_z)
        high_x_grobal = high_points[0,:] + position_x
        high_y_grobal = high_points[1,:] + position_y
        high_global = np.vstack((high_x_grobal, high_y_grobal, high_rot[2,:], high_points[3,:]) , dtype=np.float32)
        '''
        #map lim set
        map_lim_x_min = position_x + self.MAP_LIM_X_MIN;
        map_lim_x_max = position_x + self.MAP_LIM_X_MAX;
        map_lim_y_min = position_y + self.MAP_LIM_Y_MIN;
        map_lim_y_max = position_y + self.MAP_LIM_Y_MAX;
        map_lim_ground_ind = self.pcd_serch(self.pcd_ground_buff, map_lim_x_min, map_lim_x_max, map_lim_y_min, map_lim_y_max)
        map_lim_middle_ind = self.pcd_serch(self.pcd_middle_buff, map_lim_x_min, map_lim_x_max, map_lim_y_min, map_lim_y_max)
        '''
        map_lim_high_ind = self.pcd_serch(self.pcd_high_buff, map_lim_x_min, map_lim_x_max, map_lim_y_min, map_lim_y_max)
        '''
        self.pcd_ground_buff = self.pcd_ground_buff[:,map_lim_ground_ind]
        self.pcd_middle_buff= self.pcd_middle_buff[:,map_lim_middle_ind]
        '''
        self.pcd_high_buff = self.pcd_high_buff[:,map_lim_high_ind]
        '''
        
        #obs round&duplicated ground  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        pcd_ground_buff = np.insert(self.pcd_ground_buff, len(self.pcd_ground_buff[0,:]), ground_global.T, axis=1)
        points_ground_round = np.round(pcd_ground_buff * self.ground_pixel) / self.ground_pixel
        self.pcd_ground_buff =points_ground_round[:,~pd.DataFrame({"x":points_ground_round[0,:], "y":points_ground_round[1,:], "z":points_ground_round[2,:]}).duplicated()]
        
        #obs round&duplicated middle  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        pcd_middle_buff = np.insert(self.pcd_middle_buff, len(self.pcd_middle_buff[0,:]), middle_global.T, axis=1)
        points_middle_round = np.round(pcd_middle_buff * self.ground_pixel) / self.ground_pixel
        self.pcd_middle_buff =points_middle_round[:,~pd.DataFrame({"x":points_middle_round[0,:], "y":points_middle_round[1,:], "z":points_middle_round[2,:]}).duplicated()]

        # remove dynamic obs
        middle_local_x = self.pcd_middle_buff[0,:] - position_x
        middle_local_y = self.pcd_middle_buff[1,:] - position_y
        middle_global_x = np.round((middle_local_y + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        middle_global_y = np.round((middle_local_x + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        middle_valid = ((middle_global_x >= 0) & (middle_global_x < self.size) & (middle_global_y >= 0) & (middle_global_y < self.size))

        remove_mask = np.zeros(len(middle_global_x), dtype=bool)
        remove_mask[middle_valid] = (dynamic_occ[middle_global_x[middle_valid], middle_global_y[middle_valid]] > 0)
        removed_middle_points = self.pcd_middle_buff[:, remove_mask]
        self.pcd_middle_buff = self.pcd_middle_buff[:, ~remove_mask]

        if removed_middle_points.shape[1] > 0:
            # removeされたmiddle点をgrid化
            remove_x = np.round(removed_middle_points[0,:] * self.ground_pixel).astype(np.int32)
            remove_y = np.round(removed_middle_points[1,:] * self.ground_pixel).astype(np.int32)
            remove_cells = set(zip(remove_x, remove_y))

            # ground点をgrid化
            ground_x = np.round(self.pcd_ground_buff[0,:] * self.ground_pixel).astype(np.int32)
            ground_y = np.round(self.pcd_ground_buff[1,:] * self.ground_pixel).astype(np.int32)

            # ground削除mask
            ground_remove_mask = np.array([(middle_global_x, middle_global_y) in remove_cells for middle_global_x, middle_global_y in zip(ground_x, ground_y)])
            self.pcd_ground_buff = self.pcd_ground_buff[:, ~ground_remove_mask]

        #self.pcd_middle_buff = self.remove_dynamic_points(self.pcd_middle_buff, middle_dynamic_global, radius=0.25)

        #obs round&duplicated high  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        '''
        pcd_high_buff = np.insert(self.pcd_high_buff, len(self.pcd_high_buff[0,:]), high_global.T, axis=1)
        points_high_round = np.round(pcd_high_buff * self.ground_pixel) / self.ground_pixel
        self.pcd_high_buff =points_high_round[:,~pd.DataFrame({"x":points_high_round[0,:], "y":points_high_round[1,:], "z":points_high_round[2,:]}).duplicated()]
        '''
        #local reflect ground map
        ground_reflect_conv = self.pcd_ground_buff[3,:]/255*100.0
        map_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        map_data_ground_set = grid_map_set(self.pcd_ground_buff[1,:], self.pcd_ground_buff[0,:], ground_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)

        #local reflect middle map
        if self.pcd_middle_buff.shape[1] > 0:
            middle_max_intensity = np.max(self.pcd_middle_buff[3,:])
            if middle_max_intensity > 0:
                middle_reflect_conv = np.clip(self.pcd_middle_buff[3,:] / np.max(self.pcd_middle_buff[3,:]) * 100.0, 0, 100).astype(np.int8)
            else:
                middle_reflect_conv = np.zeros(self.pcd_middle_buff.shape[1], dtype=np.int8)
        else:
            middle_reflect_conv = np.zeros(0, dtype=np.int8)
        #middle_reflect_conv = np.clip(self.pcd_middle_buff[3,:] / np.max(self.pcd_middle_buff[3,:]) * 100.0, 0, 100).astype(np.int8)
        #middle_reflect_conv = self.pcd_middle_buff[3,:]/255*100.0
        map_data_middle_set = grid_map_set(self.pcd_middle_buff[1,:], self.pcd_middle_buff[0,:], middle_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)

        #local reflect high map
        '''
        high_reflect_conv = self.pcd_high_buff[3,:]/255*100.0
        map_data_high_set = grid_map_set(self.pcd_high_buff[1,:], self.pcd_high_buff[0,:], high_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)
        '''
        #if self.pcd_high_buff.shape[1] == 0:
        #    map_data_high_set = np.zeros((self.size, self.size), dtype=np.uint8)
        #else:
        #    map_data_high_set = grid_map_set(self.pcd_high_buff[1,:], self.pcd_high_buff[0,:], high_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)

        ##ekf pos ground local reflect map
        ekf_ground_buff_x = self.pcd_ground_buff[0,:] - position[0]
        ekf_ground_buff_y = self.pcd_ground_buff[1,:] - position[1]
        ekf_ground_buff_z = self.pcd_ground_buff[2,:] - position[2]
        ekf_ground_buff = np.vstack((ekf_ground_buff_x, ekf_ground_buff_y, ekf_ground_buff_z))
        ekf_ground_rot, ekf_ground_rot_matrix = rotation_xyz(ekf_ground_buff, ekf_theta_x-theta_x, ekf_theta_y-theta_y, ekf_theta_z-theta_z)
        ekf_ground_set_x = ekf_ground_rot[0,:] + ekf_position[0]
        ekf_ground_set_y = ekf_ground_rot[1,:] + ekf_position[1]
        ekf_ground_set_z = ekf_ground_rot[2,:] + ekf_position[2]
        ekf_ground_set = np.vstack((ekf_ground_set_x, ekf_ground_set_y, ekf_ground_set_z))
        #map_data_set_4save = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE)
        #print(f"map_data_ground_set ={map_data_ground_set.shape}")

        ##ekf pos middle local reflect map
        ekf_middle_buff_x = self.pcd_middle_buff[0,:] - position[0]
        ekf_middle_buff_y = self.pcd_middle_buff[1,:] - position[1]
        ekf_middle_buff_z = self.pcd_middle_buff[2,:] - position[2]
        ekf_middle_buff = np.vstack((ekf_middle_buff_x, ekf_middle_buff_y, ekf_middle_buff_z))
        ekf_middle_rot, ekf_middle_rot_matrix = rotation_xyz(ekf_middle_buff, ekf_theta_x-theta_x, ekf_theta_y-theta_y, ekf_theta_z-theta_z)
        ekf_middle_set_x = ekf_middle_rot[0,:] + ekf_position[0]
        ekf_middle_set_y = ekf_middle_rot[1,:] + ekf_position[1]
        ekf_middle_set_z = ekf_middle_rot[2,:] + ekf_position[2]
        ekf_middle_set = np.vstack((ekf_middle_set_x, ekf_middle_set_y, ekf_middle_set_z))
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
        #map_data_set_4save = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE)
        #print(f"map_data_high_set ={map_data_high_set.shape}")
        '''
	
        #GL reflect map
        #map_data_gl_set = grid_map_set(self.pcd_ground_buff[1,:], self.pcd_ground_buff[0,:], ground_reflect_conv, position, self.ground_pixel, self.MAP_RANGE_GL)
        map_data_ground_gl_set = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        map_data_middle_gl_set = grid_map_set(ekf_middle_set[1,:], ekf_middle_set[0,:], middle_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        '''
        map_data_high_gl_set = grid_map_set(ekf_high_set[1,:], ekf_high_set[0,:], high_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        '''
        #if self.pcd_high_buff.shape[1] == 0:
        #    map_data_high_gl_set = np.zeros((self.size, self.size), dtype=np.uint8)
        #else:
        #    map_data_high_gl_set = grid_map_set(ekf_high_set[1,:], ekf_high_set[0,:], high_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        #print(f"map_data_ground_set ={map_data_ground_set.shape}")
	
        
        #publish for rviz2 
        #global ground
        ground_global_msg = point_cloud_intensity_msg(self.pcd_ground_buff.T, t_stamp, 'odom')
        self.pcd_ground_global_publisher.publish(ground_global_msg) 
        #local map ground
        self.map_data_ground = make_map_msg(map_data_ground_set, self.ground_pixel, position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        #local map middle
        self.map_data_middle = make_map_msg(map_data_middle_set, self.ground_pixel, position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        #local map high
        '''
        self.map_data_high = make_map_msg(map_data_high_set, self.ground_pixel, position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        '''
        #self.map_data = make_map_msg(map_data_set_4save, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        self.map_data_flag = 1
        #self.reflect_map_local_publisher.publish(self.map_data)     
        #gl map ground
        self.map_data_ground_gl = make_map_msg(map_data_ground_gl_set, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE_GL, "odom")
        #gl map middle
        self.map_data_middle_gl = make_map_msg(map_data_middle_gl_set, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE_GL, "odom")
        #gl map high
        '''
        self.map_data_high_gl = make_map_msg(map_data_high_gl_set, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE_GL, "odom")
        '''
        #self.reflect_map_global_publisher.publish(self.map_data_gl) 
        self.map_data_gl_flag = 1
        
        if self.MAKE_GL_MAP_FLAG == 1:
            #self.make_ref_map(position_x, position_y, theta_z)
            #self.make_ref_map(ekf_position_x, ekf_position_y, ekf_theta_z)
            self.make_ref_map(map_data_ground_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z, layer="ground")
            self.make_ref_map(map_data_middle_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z, layer="middle")
            #self.make_ref_map(map_data_high_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z, layer="high")
        
        remove_vis = np.zeros((self.size,self.size), dtype=np.int8)
        #remove_vis[gx[remove_mask], gy[remove_mask]] = 100
        remove_vis[dynamic_occ > 0] = 20
        remove_vis[middle_global_x[remove_mask], middle_global_y[remove_mask]] = 100

        grid = OccupancyGrid()
        grid.header = self.header
        grid.header.frame_id = "odom"
        grid.info.resolution = 1.0 / self.grid_pixel
        grid.info.width = self.size
        grid.info.height = self.size
        grid.info.origin.position.x = (self.position_x - self.MAP_RANGE)
        grid.info.origin.position.y = (self.position_y - self.MAP_RANGE)
        grid.info.origin.orientation.w = 1.0
        grid.data = remove_vis.flatten().tolist()
        self.remove_map_pub.publish(grid)
        
    def make_ref_map(self, image, position_x, position_y, theta_z, layer):
        map_pos_diff = math.sqrt((position_x - self.map_position_x_buff)**2 + (position_y - self.map_position_y_buff)**2)
        map_theta_diff = abs(theta_z -  self.map_theta_z_buff)
        if ( (map_pos_diff > 10) or ((map_pos_diff > 2) and (map_theta_diff > 40)) ):
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
            
            
            self.map_position_x_buff = position_x #[m]
            self.map_position_y_buff = position_y #[m]
            self.map_theta_z_buff = theta_z #[deg]
            self.map_number += 1

    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ))
        return pcd_ind

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
    map_min_x = (-map_range + position[1] ) * map_pixel
    map_max_x = ( map_range + position[1] ) * map_pixel
    map_min_y = (-map_range + position[0] ) * map_pixel
    map_max_y = ( map_range + position[0] ) * map_pixel
    map_ind_px = np.round(map_x * map_pixel )# index
    map_ind_py = np.round(map_y * map_pixel )
    map_px = np.round(map_x * map_pixel -position[1]*map_pixel )#障害物をグリッドサイズで間引き
    map_py = np.round(map_y * map_pixel -position[0]*map_pixel )
    map_ind = (map_min_x +map_pixel < map_ind_px) * (map_ind_px < map_max_x - (1)) * (map_min_y+map_pixel < map_ind_py) * (map_ind_py < map_max_y - (1))#
    
    #0/1 judge
    #map_xy =  np.zeros([int(map_max_x - map_min_x),int(map_max_y - map_min_y)], np.uint8)
    map_xy =  np.zeros([int(2* map_range * map_pixel),int(2* map_range * map_pixel)], np.uint8)
    map_data = map_xy #reflect to map#np.zeros([int(map_max_x - map_min_x),int(map_max_y - map_min_y),1], np.uint8)
    
    print(f"map_xy ={map_xy.shape}")
    print(f"data ={data.shape}")
    print(f"data(map_ind) ={data[map_ind].shape}")
    
    map_data = map_data.reshape(1,len(map_xy[0,:])*len(map_xy[:,0]))
    map_data[:,:] = 0.0
    map_data_x = (map_px[map_ind] - map_range*map_pixel  ) * len(map_xy[0,:])
    map_data_y =  map_py[map_ind] - map_range*map_pixel
    map_data_xy =  list(map(int, map_data_x + map_data_y ) )
    print(f"map_data ={map_data.shape}")
    print(f"map_data_xy ={len(map_data_xy)}")
    print(f"data[map_ind] ={len(data[map_ind])}")
    
    if np.any(map_ind):
        data_max = np.max(data[map_ind])
        map_data_xy_max = np.max(map_data_xy)
    else:
        data_max = 0  # or np.nan
        map_data_xy_max = 0
    
    #data_max = np.max(data[map_ind])
    print(f"data_max ={data_max}")
    #map_data_xy_max = np.max(map_data_xy)
    print(f"map_data_xy_max ={map_data_xy_max}")
    
    
    map_data[0,map_data_xy] = data[map_ind]
    map_data_set = map_data.reshape(len(map_xy[:,0]),len(map_xy[0,:]))
    
    print(f"map_data_set ={map_data_set.shape}")
    
    #map flipud
    #map_xy = np.flipud(map_xy)
    map_xy = np.flipud(map_data_set)
    
    map_xy_max_ind = np.unravel_index(np.argmax(map_xy), map_xy.shape)
    print(f"map_xy_max_ind ={map_xy_max_ind}")
    print(f"map_xy_max ={map_xy[map_xy_max_ind]}")
    
    return map_xy

###################################################################

def main():
    rclpy.init()
    node = ObsBayesMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()