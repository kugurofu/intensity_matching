# ROS2
import rclpy
from rclpy.node import Node
# ROS msgs
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
from nav_msgs.msg import OccupancyGrid
# Python
import numpy as np
import math
import cv2


class ObsBayesMap(Node):
    def __init__(self):
        super().__init__('obs_bayes_map')
        # Subscriber
        self.pcd_obs_sub = self.create_subscription(sensor_msgs.PointCloud2,'/pcd_segment_middle',self.publish_map,10)

        self.local_odom_sub = self.create_subscription(nav_msgs.Odometry,'/fusion/odom',self.get_local_odom,10)

        # Publisher
        self.bayes_map_pub = self.create_publisher(OccupancyGrid,'/obs_bayes_map',10)
        self.static_map_pub = self.create_publisher(OccupancyGrid,'/static_obs_map',10)
        self.dynamic_map_pub = self.create_publisher(OccupancyGrid,'/dynamic_obs_map',10)
        self.log_map_pub = self.create_publisher(OccupancyGrid,'/log_map',10)

        # Parameter
        self.grid_pixel = 1000 / 50.0
        self.MAP_RANGE = 15.0
        self.size = int(self.MAP_RANGE * 2 * self.grid_pixel)

        # Odom
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.theta_z = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0

        # Weighted Occupancy
        self.hit_score = np.zeros((self.size, self.size), dtype=np.float32)
        self.miss_score = np.zeros((self.size, self.size), dtype=np.float32)
        self.logodds = np.zeros((self.size, self.size), dtype=np.float32)

        self.shift_residual_x = 0.0
        self.shift_residual_y = 0.0

        # forgetting factor
        self.decay = 0.995

    def get_local_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        _, _, yaw = quaternion_to_euler(qx, qy, qz, qw)
        self.theta_z = yaw

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

    def publish_map(self, msg):
        # Forgetting
        #self.hit_score *= self.decay
        #self.miss_score *= 0.999
        #self.static_score *= 0.98

        # PointCloud
        x, y, z, intensity = self.pointcloud2_to_array(msg)
        theta = self.theta_z

        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        # Odometry
        position_x = self.position_x
        position_y = self.position_y
        prev_x = self.prev_x
        prev_y = self.prev_y

        # local map
        x_global = x_rot
        y_global = y_rot

        # Map shift
        dx = position_x - prev_x
        dy = position_y - prev_y

        self.shift_residual_x += dx * self.grid_pixel
        self.shift_residual_y += dy * self.grid_pixel

        shift_x = int(self.shift_residual_x)
        shift_y = int(self.shift_residual_y)

        self.shift_residual_x -= shift_x
        self.shift_residual_y -= shift_y

        #shift_x = round(dx * self.grid_pixel)
        #shift_y = round(dy * self.grid_pixel)

        self.hit_score = np.roll(self.hit_score, -shift_x, axis=1)
        self.hit_score = np.roll(self.hit_score, -shift_y, axis=0)
        self.miss_score = np.roll(self.miss_score, -shift_x, axis=1)
        self.miss_score = np.roll(self.miss_score, -shift_y, axis=0)
        self.logodds = np.roll(self.logodds, -shift_x, axis=1)
        self.logodds = np.roll(self.logodds, -shift_y, axis=0)

        # clear edge
        if shift_x > 0:
            self.hit_score[:, -shift_x:] = 0
            self.miss_score[:, -shift_x:] = 0
            self.logodds[:, -shift_x:] = 0
        elif shift_x < 0:
            self.hit_score[:, :-shift_x] = 0
            self.miss_score[:, :-shift_x] = 0
            self.logodds[:, :-shift_x] = 0
        if shift_y > 0:
            self.hit_score[-shift_y:, :] = 0
            self.miss_score[-shift_y:, :] = 0
            self.logodds[-shift_y:, :] = 0
        elif shift_y < 0:
            self.hit_score[:-shift_y, :] = 0
            self.miss_score[:-shift_y, :] = 0
            self.logodds[:-shift_y, :] = 0

        # Polar bin
        angle_polar = np.arctan2(y_global, x_global)
        angle_bins = np.linspace(-np.pi, np.pi, 720)
        bin_idx = np.digitize(angle_polar, angle_bins) - 1
        
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

        # Sensor Origin
        x0 = int(self.MAP_RANGE * self.grid_pixel)
        y0 = int(self.MAP_RANGE * self.grid_pixel)
        
        '''
        min_dist = {}
        min_points = {}

        for i in range(len(dist)):
            count = bin_idx[i]
            if (count not in min_dist or dist[i] < min_dist[count]):
                min_dist[count] = dist[i]
                min_points[count] = (gx[i], gy[i], dist[i])
        '''

        ray_points = {}

        for i in range(len(dist)):
            angle = angle_polar[i]
            count = int((angle + np.pi) / (2*np.pi) * 720)
            d = dist[i]

            if count not in ray_points:
                ray_points[count] = (gx[i], gy[i], d)
            elif d < ray_points[count][2]:
                ray_points[count] = (gx[i], gy[i], d)

        # Ray casting
        max_range = 15.0

        #for count in range(len(angle_bins)):
        for count, (gx_i, gy_i, d) in ray_points.items():
            angle = angle_bins[count]
            x1 = int(x0 + max_range * np.cos(angle) * self.grid_pixel)
            y1 = int(y0 + max_range * np.sin(angle) * self.grid_pixel)

            # OCCUPIED HIT
            if count in ray_points:
                gx_i, gy_i, d = ray_points[count]
                dx = gx_i - x0
                dy = gy_i - y0
                n = int(max(abs(dx), abs(dy)))
                xs = np.linspace(x0, gx_i, n).astype(np.int32)
                ys = np.linspace(y0, gy_i, n).astype(np.int32)
                
                # FREE UPDATE
                for k in range(n-3):
                    cx = xs[k]
                    cy = ys[k]
                    if 0 <= cx < self.size and 0 <= cy < self.size:
                        d_free = k / self.grid_pixel
                        free_conf = self.free_model(d_free)
                        self.miss_score[cx, cy] += free_conf * 0.2
                        self.logodds[cx, cy] += -0.4
                        #self.miss_score[xs[:-8], ys[:-8]] += free_conf * 0.1

                # OCCUPIED UPDATE
                occ_conf = self.occ_model(d)
                self.hit_score[gx_i, gy_i] += occ_conf * 0.1
                self.logodds[gx_i, gy_i] += 0.6

        # Probability
        total = (self.hit_score + self.miss_score)
        score = (self.hit_score - self.miss_score)
        prob = np.zeros_like(total, dtype=np.float32)
        observed = total > 0.01
        prob[observed] = ((score[observed] / total[observed]) + 1.0) * 0.5

        # logodds
        self.logodds = np.clip(self.logodds, -5.0, 5.0)
        prob_log = 1.0 / (1.0 + np.exp(-self.logodds))
        observed_log = np.abs(self.logodds) > 0.1
        occ_log = (prob_log * 100).astype(np.int8)
        occ_log[~observed_log] = -1

        static_mask = ((prob > 0.5)) # & (score > 0.0)) # hit - miss
        #static_mask = (prob_log > 0.65) # log
        #static_mask = (self.static_score > 30.0) 
        dynamic_mask = ((prob > 0.05) & (prob < 0.1)) # log

        # OccupancyGrid
        occ = (prob * 100).astype(np.int8)
        occ[~observed] = -1

        #static_occ = np.full_like(occ, -1, dtype=np.int8)
        static_occ = np.zeros_like(occ, dtype=np.int8)
        #static_occ[prob_log < 0.2] = 0
        static_occ[static_mask] = 100

        #dynamic_occ = np.full_like(occ, -1, dtype=np.int8)
        dynamic_occ = np.zeros_like(occ, dtype=np.int8)
        #dynamic_occ[prob_log < 0.3] = 0
        dynamic_occ[dynamic_mask] = 100

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

        # Update previous pose
        self.prev_x = self.position_x
        self.prev_y = self.position_y

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