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
        self.pcd_obs_sub = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_obs', self.publish_map, 10)
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

        self.l_occ = 1.0
        self.l_free = -0.3
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

        valid = (gx >= 0) & (gx < self.size) & (gy >= 0) & (gy < self.size)

        gx = gx[valid]
        gy = gy[valid]
           
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

        #self.log_odds[obs_mask] += self.l_occ
        #self.log_odds[~obs_mask] += self.l_free
        #i=0
                
        for (x1, y1) in zip(gx, gy):     
            #x0 = int((self.position_x + self.MAP_RANGE) * self.grid_pixel)
            #y0 = int((self.position_y + self.MAP_RANGE) * self.grid_pixel)

            #x1 = gx[i]
            #y1 = gy[i]
            #if 0 <= x1 < self.size and 0 <= y1 < self.size:
            #    self.log_odds[x1, y1] += self.l_occ
            #    self.hit_count[x1, y1] += 1

            # Bresenhamなどで直線生成
            line = self.bresenham(x0, y0, x1, y1)

            # free更新（最後以外）
            for (cx, cy) in line[:-1]:
                if 0 <= cx < self.size and 0 <= cy < self.size:
                    #self.log_odds[cx, cy] += self.l_free
                    self.miss_count[cx, cy] += 1
            
            # hit（occupied）
            if 0 <= x1 < self.size and 0 <= y1 < self.size:
                self.hit_count[x1, y1] += 2

            # occupied更新
            #self.log_odds[x1, y1] += self.l_occ
            #i = i + 1
        
        #obs_mask = np.zeros_like(self.log_odds, dtype=bool)
        #obs_mask[gx, gy] = True
        #self.log_odds[obs_mask] += self.l_occ
        #self.log_odds[~obs_mask] += self.l_free

        self.log_odds = np.clip(self.log_odds, self.l_min, self.l_max)

        # ===== 確率変換 =====
        #prob = 1.0 / (1.0 + np.exp(-self.log_odds)) # original
        total = self.hit_count + self.miss_count + 1e-6
        prob = self.hit_count / total
        #static_mask = prob > self.static_prob # static obs map
        static_mask = (prob > self.static_prob) & (self.hit_count > 1) # static obs map
        dynamic_mask = (prob < self.dynamic_prob) & (self.miss_count > 1) # dynamic obs map
        unknown_mask = (prob >= self.dynamic_prob) & (prob <= self.static_prob)
        #unknown_mask = (self.hit_count < 1)

        # ===== OccupancyGrid形式 =====
        #occ = (prob * 100).astype(np.int8)
        occ = np.zeros_like(prob, dtype=np.int8)
        occ[static_mask] = 100 # black
        occ[dynamic_mask] = 0 # white
        occ[unknown_mask] = 50 # gray

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
