# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
# ROS 2の文字列型を使えるようにimport
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
from livox_ros_driver2.msg import CustomMsg
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd
#import open3d as o3d
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import OccupancyGrid
import cv2
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import time
import matplotlib.pyplot
import struct
import geometry_msgs.msg as geometry_msgs
from collections import OrderedDict

from scipy import interpolate
from std_msgs.msg import Float32MultiArray
import cv2
import subprocess


class ObsBayesMap(Node):
    def __init__(self):
        super().__init__('obs_bayes_map')

        # subscriber
        self.pcd_obs_sub = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_middle', self.publish_map, 10)
        self.local_odom_sub = self.create_subscription(nav_msgs.Odometry,'/fusion/odom', self.get_local_odom, 10)

        # publisher
        self.bayes_map_pub = self.create_publisher(OccupancyGrid, '/obs_bayes_map', 10)

        # parameter
        self.grid_pixel = 1000/50#障害物のグリッドサイズ設定
        #self.resolution = 0.05  # 5cm
        self.MAP_RANGE = 15.0 #[m] global 150?
        self.size = int(self.MAP_RANGE * 2 * self.grid_pixel)
        self.static_prob = 0.6
        self.dynamic_prob = 0.3
        
        #ground 
        #self.ground_pixel = 1000/50#障害物のグリッドサイズ設定
        #self.MAP_RANGE = 15.0 #[m]
        
        #self.MAP_RANGE_GL = 20.0 #[m]
        #self.MAP_LIM_X_MIN = -25.0 #[m]
        #self.MAP_LIM_X_MAX =  25.0 #[m]
        #self.MAP_LIM_Y_MIN = -25.0 #[m]
        #self.MAP_LIM_Y_MAX =  25.0 #[m]
        
        #map position
        #self.map_position_x_buff = 0.0 #[m]
        #self.map_position_y_buff = 0.0 #[m]
        #self.map_position_z_buff = 0.0 #[m]
        #self.map_theta_z_buff = 0.0 #[deg]
        #self.map_number = 0 # int
        
        #odom positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        self.prev_x = 0.0
        self.prev_y = 0.0

        # log-odds
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)

        self.l_occ = 1.2
        self.l_free = -0.1
        self.l_min = -2.0
        self.l_max = 3.5

        self.hit_count = np.zeros((self.size, self.size), dtype=np.int16)
        self.miss_count = np.zeros((self.size, self.size), dtype=np.int16)

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
    
    def pointcloud2_to_array(self, msg):
        # Extract point cloud data
        points = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        
        return x, y, z, intensity
    
    def occ_model(self, d):
        sigma = 3.0
        p0 = 0.95
        p_min = 0.65
        return max(p_min, p0 * np.exp(-(d**2) / (2 * sigma**2)))

    def free_model(self, d):
        p0 = 0.2
        lam = 5.0
        return p0 + (0.5 - p0) * np.exp(-d / lam)

    def logit(self, p):
        p = np.clip(p, 1e-4, 1-1e-4)
        return np.log(p / (1 - p))
    
    def bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0

        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x1, y1))
        return points

    def publish_map(self, msg):
        #self.log_odds.fill(0) # local
        x, y, z, intensity = self.pointcloud2_to_array(msg)
        
        theta = self.theta_z * np.pi / 180.0

        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        # 平行移動（odomへ）
        #x_global = x_rot + self.position_x
        #y_global = y_rot + self.position_y  
        x_global = x_rot # local
        y_global = y_rot # local
        dx = self.position_x - self.prev_x
        dy = self.position_y - self.prev_y
        shift_x = int(dx * self.grid_pixel)
        shift_y = int(dy * self.grid_pixel)
        self.log_odds = np.roll(self.log_odds, -shift_x, axis=1)
        self.log_odds = np.roll(self.log_odds, -shift_y, axis=0)
        self.hit_count = np.roll(self.hit_count, -shift_x, axis=1)
        self.hit_count = np.roll(self.hit_count, -shift_y, axis=0)
        self.miss_count = np.roll(self.miss_count, -shift_x, axis=1)
        self.miss_count = np.roll(self.miss_count, -shift_y, axis=0)

        if shift_x > 0:
            self.log_odds[:, -shift_x:] = 0
            self.hit_count[:, -shift_x:] = 0
            self.miss_count[:, -shift_x:] = 0
        elif shift_x < 0:
            self.log_odds[:, :-shift_x] = 0
            self.hit_count[:, :-shift_x] = 0
            self.miss_count[:, :-shift_x] = 0

        if shift_y > 0:
            self.log_odds[-shift_y:, :] = 0
            self.hit_count[-shift_y:, :] = 0
            self.miss_count[-shift_y:, :] = 0
        elif shift_y < 0:
            self.log_odds[:-shift_y, :] = 0
            self.hit_count[:-shift_y, :] = 0
            self.miss_count[:-shift_y, :] = 0

        # ===== グリッド化 =====
        gx = np.floor((y_global + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        gy = np.floor((x_global + self.MAP_RANGE) * self.grid_pixel).astype(np.int32)
        dist = np.sqrt(x_global**2 + y_global**2)
        mask = dist < 15.0

        gx = gx[mask]
        gy = gy[mask]
        dist = dist[mask]

        valid = (gx >= 0) & (gx < self.size) & (gy >= 0) & (gy < self.size)

        gx = gx[valid]
        gy = gy[valid]
        dist = dist[valid]
           
        # ===== ベイズ更新 =====
        #self.log_odds *= 0.99 # local
        #unique_idx = np.unique(np.stack((gx, gy), axis=1), axis=0)
        #self.log_odds[unique_idx[:, 0], unique_idx[:, 1]] += self.l_occ
        #obs_mask = np.zeros_like(self.log_odds, dtype=bool)
        #obs_mask[unique_idx[:,0], unique_idx[:,1]] = True

        #x0 = int((self.position_x + self.MAP_RANGE) * self.grid_pixel)
        #y0 = int((self.position_y + self.MAP_RANGE) * self.grid_pixel)
        x0 = int((self.MAP_RANGE) * self.grid_pixel) # local
        y0 = int((self.MAP_RANGE) * self.grid_pixel) # local

        obs_set = set(zip(gx, gy))
        obs_array = np.zeros((self.size, self.size), dtype=np.uint8)
        #obs_array[gx, gy] = 1
        #obs_array = cv2.dilate(obs_array, np.ones((3,3), np.uint8))
        angle_polar = np.arctan2(y_global, x_global)
        angle_bins = np.linspace(-np.pi/3, np.pi/3, 400)  # 視野角
        bin_idx = np.digitize(angle_polar, angle_bins)
        max_range = 15.0
        min_dist = {}
        min_points = {}

        for i in range(len(dist)):
            b = bin_idx[i]
            if b not in min_dist or dist[i] < min_dist[b]:
                min_dist[b] = dist[i]
                min_points[b] = (gx[i], gy[i], dist[i])
        
        filterd_points = list(min_points.values())

        '''
        for angle in angle_bins:
            global_angle = theta + angle
            x1 = int(x0 + max_range * np.cos(global_angle) * self.grid_pixel)
            y1 = int(y0 + max_range * np.sin(global_angle) * self.grid_pixel)
            
            line = self.bresenham(x0, y0, x1, y1)

            for (cy, cx) in line:
                if not (0 <= cx < self.size and 0 <= cy < self.size):
                    break

                self.log_odds[cx, cy] += self.l_free * 0.5
        '''
        for b in range(len(angle_bins)):

            angle = angle_bins[b]
            global_angle = theta + angle

            x0 = int(self.MAP_RANGE * self.grid_pixel)
            y0 = int(self.MAP_RANGE * self.grid_pixel)

            # =========================
            # 🔴 ヒットあり
            # =========================
            if b in min_points:

                gx_i, gy_i, d = min_points[b]

                line = self.bresenham(x0, y0, gx_i, gy_i)

                # free（手前）
                for k, (cy, cx) in enumerate(line[:-3]):
                    if 0 <= cx < self.size and 0 <= cy < self.size:
                        d = k / self.grid_pixel
                        p_free = self.free_model(d)
                        self.log_odds[cx, cy] += 0.3 * (self.logit(p_free) - self.logit(0.5))

                # occ（最後）
                #d = len(line) / self.grid_pixel
                p_occ = self.occ_model(d)
                #self.log_odds[gx_i, gy_i] += self.l_occ
                self.log_odds[gx_i, gy_i] += 3.5 * (self.logit(p_occ) - self.logit(0.5))

            # =========================
            # 🔵 ヒットなし
            # =========================
            else:
                #self.log_odds[cx, cy] += -0.02
                #self.log_odds[cx, cy] *= 0.98
                
                x1 = int(x0 + max_range * np.cos(global_angle) * self.grid_pixel)
                y1 = int(y0 + max_range * np.sin(global_angle) * self.grid_pixel)

                line = self.bresenham(x0, y0, x1, y1)

                for k, (cy, cx) in enumerate(line):
                    if 0 <= cx < self.size and 0 <= cy < self.size:
                        d = k / self.grid_pixel
                        p_free = self.free_model(d)

                        # 弱め更新（重要）
                        self.log_odds[cx, cy] += 0.2 * (self.logit(p_free) - self.logit(0.5))
                
        '''
        for (gx_i, gy_i) in filterd_points:
            obs_array[gx_i, gy_i] = 1
            line = self.bresenham(x0, y0, gx_i, gy_i)
            # free更新（手前）
            for (cy, cx) in line[:-1]:
                if 0 <= cx < self.size and 0 <= cy < self.size:
                    self.log_odds[cx, cy] += self.l_free

            # 最後だけoccupied
            if 0 <= gx_i < self.size and 0 <= gy_i < self.size:
                self.log_odds[gx_i, gy_i] += self.l_occ
        
        obs_array = cv2.dilate(obs_array, np.ones((3,3), np.uint8))
        '''
        '''
        for angle in angle_bins:
            global_angle = theta + angle
            x1 = int(x0 + max_range * np.cos(global_angle) * self.grid_pixel)
            y1 = int(y0 + max_range * np.sin(global_angle) * self.grid_pixel)
            
            line = self.bresenham(x0, y0, x1, y1)

            for (cy, cx) in line:
                if not (0 <= cx < self.size and 0 <= cy < self.size):
                    break

                # 障害物にヒット
                if obs_array[cx, cy] == 1:
                #if (cx, cy) in obs_set:
                    self.log_odds[cx, cy] += self.l_occ
                    break
                
                else:
                    self.log_odds[cx, cy] += self.l_free

                # それまで free
                #if self.log_odds[cx, cy] < 0.5:
                #self.log_odds[cx, cy] += self.l_free

            #for (cx, cy) in line:
            #    if 0 <= cx < self.size and 0 <= cy < self.size:
            #        if self.log_odds[cx, cy] < 0.5:
            #            self.log_odds[cx, cy] += self.l_free
        '''
        #for i in range(0, len(gx), 5):
        #    x1 = gx[i]
        #    y1 = gy[i]

            # Bresenhamなどで直線生成
            #line = self.bresenham(x0, y0, x1, y1)

            # free更新（最後以外）
            #for (cx, cy) in line[:-1]:
            #        if self.log_odds[cx, cy] < 0.5:
            #            self.log_odds[cx, cy] += self.l_free
            
            # hit（occupied）
            #if 0 <= x1 < self.size and 0 <= y1 < self.size:
            #    if self.log_odds[x1, y1] > -0.5:
            #        self.log_odds[x1, y1] += self.l_occ

        self.log_odds = np.clip(self.log_odds, self.l_min, self.l_max)

        # ===== 確率変換 =====
        prob = 1.0 / (1.0 + np.exp(-self.log_odds)) # original
        static_mask = (prob > self.static_prob) #& (self.hit_count > 1) # static obs map
        dynamic_mask = (prob < self.dynamic_prob) #& (self.miss_count > 1) # dynamic obs map
        unknown_mask = (prob >= self.dynamic_prob) & (prob <= self.static_prob)
        #observed_mask = (self.log_odds != 0)
        observed_mask = np.abs(self.log_odds) > 0.01 # 0.01

        # ===== OccupancyGrid形式 =====
        #occ = np.full_like(prob, -1, dtype=np.int8)  # unknown = -1
        occ = (prob * 100).astype(np.int8)
        occ[~observed_mask] = -1

        #occ[static_mask] = 100   # 黒（障害物）
        #occ[dynamic_mask] = 0     # 白（自由空間）

        #occ[observed_mask & static_mask] = 100
        #occ[observed_mask & dynamic_mask] = 0

        # ===== 前方領域の切り出し =====
        forward_min = 0.0
        forward_max = 10.0
        left_min = -5.0
        left_max = 5.0

        gx_min = int(x0 + forward_min * self.grid_pixel)
        gx_max = int(x0 + forward_max * self.grid_pixel)
        gy_min = int(y0 + left_min * self.grid_pixel)
        gy_max = int(y0 + left_max * self.grid_pixel)

        # 範囲チェック（重要）
        gx_min = max(0, gx_min)
        gx_max = min(self.size, gx_max)
        gy_min = max(0, gy_min)
        gy_max = min(self.size, gy_max)

        # 切り出し
        local_occ = occ[gx_min:gx_max, gy_min:gy_max]
        free_mask = (local_occ == 0)
        free_space = np.where(free_mask, 1, 0)

        self.prev_x = self.position_x
        self.prev_y = self.position_y
        self.hit_count = np.clip(self.hit_count, 0, 100)
        self.miss_count = np.clip(self.miss_count, 0, 100)

        # ===== メッセージ作成 =====
        grid = OccupancyGrid()
        grid.header = msg.header
        grid.header.frame_id = "odom"

        grid.info.resolution = 1 / self.grid_pixel
        grid.info.width = self.size
        grid.info.height = self.size

        grid.info.origin.position.x = self.position_x -self.MAP_RANGE
        grid.info.origin.position.y = self.position_y -self.MAP_RANGE
        #grid.info.origin.position.x = -self.MAP_RANGE
        #grid.info.origin.position.y = -self.MAP_RANGE
        grid.info.origin.orientation.w = 1.0

        grid.data = occ.flatten().tolist()

        self.bayes_map_pub.publish(grid)

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
    

def main():
    rclpy.init()
    node = ObsBayesMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
