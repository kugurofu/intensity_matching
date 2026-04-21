import rclpy
from rclpy.node import Node
import numpy as np
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

class BayesPCDFilter(Node):
    def __init__(self):
        super().__init__('bayes_pcd_filter')
        
        # subscription
        self.pcd_obs_sub = self.create_subscription(sensor_msgs.PointCloud2,'/pcd_segment_obs',self.callback,10)

        # publisher
        self.pcd_obs_global_publisher = self.create_publisher(sensor_msgs.PointCloud2,'/pcd_obs_filtered',10)

        # parameter
        #ground 
        self.ground_pixel = 1000/50#障害物のグリッドサイズ設定
        self.MAP_RANGE = 15.0 #[m]
        #self.grid_res = 0.05  # 5cm
        #self.map_size = 20.0  # +-20m

        self.grid_width = int(self.MAP_RANGE * 2 * self.ground_pixel)
        
        self.intensity_grid = np.zeros((self.grid_width, self.grid_width), dtype=np.float32)

        # log-odds
        self.log_odds = np.zeros((self.grid_width, self.grid_width), dtype=np.float32)

        self.l_occ = 0.85
        self.l_free = -0.4
        self.l_min = -2.0
        self.l_max = 3.5

    def pointcloud2_to_array(self, msg):
        # Extract point cloud data
        points = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        
        return point_cloud_matrix
    
    def callback(self, msg):
        #x, y, z = self.pointcloud2_to_xyz(msg)
        points = self.pointcloud2_to_array(msg)
        t_stamp = msg.header.stamp
        
        x = points[0, :]
        y = points[1, :]
        z = points[2, :]
        intensity = points[3, :]

        # ===== グリッド化 =====
        #gx = np.floor((x + self.map_size) / self.grid_res).astype(int)
        #gy = np.floor((y + self.map_size) / self.grid_res).astype(int)
        gx = np.floor((x + self.MAP_RANGE) * self.ground_pixel).astype(int)
        gy = np.floor((y + self.MAP_RANGE) * self.ground_pixel).astype(int)

        valid = (gx >= 0) & (gx < self.grid_width) & (gy >= 0) & (gy < self.grid_width)

        gx = gx[valid]
        gy = gy[valid]
        intensity = intensity[valid]
        #obs_grid = np.zeros_like(self.log_odds, dtype=bool)
        #obs_grid[gx, gy] = True
        
        #intensity_grid = np.zeros_like(self.log_odds)
        
        # 最大値で更新（蓄積）
        for i in range(len(gx)):
            self.intensity_grid[gx[i], gy[i]] = max(self.intensity_grid[gx[i], gy[i]], intensity[i])
            
        # ===== ベイズ更新 =====
        #self.log_odds[obs_grid] += self.l_occ
        #self.log_odds[~obs_grid] += self.l_free
        #self.log_odds = np.clip(self.log_odds, self.l_min, self.l_max)

        #prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        
        unique_idx = np.unique(np.stack((gx, gy), axis=1), axis=0)
        self.log_odds[unique_idx[:,0], unique_idx[:,1]] += self.l_occ

        self.log_odds = np.clip(self.log_odds, self.l_min, self.l_max)

        prob = 1.0 / (1.0 + np.exp(-self.log_odds))

        # ===== 安定セル抽出 =====
        stable = prob > 0.7

        # ===== PointCloudに戻す =====
        idx = np.array(np.where(stable)).T

        if len(idx) == 0:
            return

        px = idx[:, 0] / self.ground_pixel - self.MAP_RANGE
        py = idx[:, 1] / self.ground_pixel - self.MAP_RANGE
        pz = np.zeros_like(px)
        p_intensity = self.intensity_grid[idx[:,0], idx[:,1]]

        filterd_points = np.vstack((px, py, pz, p_intensity)).T.astype(np.float32)

        #msg_out = self.create_pointcloud2(points, msg.header)
        #self.pub.publish(msg_out)
        obs_global_msg = point_cloud_intensity_msg(filterd_points, t_stamp, 'odom')
        self.pcd_obs_global_publisher.publish(obs_global_msg) 

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

def main(args=None):
    rclpy.init(args=args)
    node = BayesPCDFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
