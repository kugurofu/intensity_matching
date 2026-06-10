# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
import std_msgs.msg as std_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import numpy as np
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import time
import geometry_msgs.msg as geometry_msgs
import glob
import cv2
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import transforms3d
from geometry_msgs.msg import Quaternion
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient
from my_msgs.action import StopFlag  # Actionメッセージのインポート
from sensor_msgs.msg import Image

# C++と同じく、Node型を継承します。
class WaypointManagerMaprun(Node):
    # コンストラクタです、PcdRotationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('waypoint_manager_maprun_node')
        
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
        
        qos_profile_sub10 = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )
        # Subscriptionを作成。
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom/combine', self.get_odom, qos_profile_sub)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_fast', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom/combine', self.get_ekf_odom, qos_profile_sub)
        self.subscription = self.create_subscription(Image,'/rgb_reflect_map_local', self.get_local_height_map, qos_profile_sub)
        self.bridge = CvBridge()
        #self.subscription = self.create_subscription(Image,'/local_height_map',self.get_local_height_map,qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # タイマーを0.1秒（100ミリ秒）ごとに呼び出す
        self.timer1 = self.create_timer(0.1, self.waypoint_manager)
        
        # Publisherを作成
        self.current_waypoint_publisher = self.create_publisher(geometry_msgs.PoseArray, 'current_waypoint', qos_profile) #set publish pcd topic name
        self.map_match_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_local', map_qos_profile_sub)
        self.map_match_ref_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_ref', map_qos_profile_sub)
        self.map_match_result_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_result', map_qos_profile_sub)
        #self.map_match_result_publisher = self.create_publisher(sensor_msgs.Image, 'reflect_map_match_result', map_qos_profile_sub)
        
        self.odom_ref_slam_publisher = self.create_publisher(nav_msgs.Odometry, 'odom_ref_slam', qos_profile)
        self.waypoint_path_publisher = self.create_publisher(nav_msgs.Path, 'waypoint_path', qos_profile) 
        self.map_obs_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'map_obs', qos_profile) 
        
        self.fused_pub = self.create_publisher(nav_msgs.Odometry, '/odom_ekf_match', qos_profile)
        self.fused_msg = nav_msgs.Odometry()

        self.timer2 = self.create_timer(0.1, self.publish_fused_value)
        
        #パラメータ
        #waypoint init
        self.current_waypoint = 0
        self.stop_flag = 0
        
        #positon init odom
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        #positon init ekf
        self.ekf_position_x = 0.0 #[m]
        self.ekf_position_y = 0.0 #[m]
        self.ekf_position_z = 0.0 #[m]
        self.ekf_theta_x = 0.0 #[deg]
        self.ekf_theta_y = 0.0 #[deg]
        self.ekf_theta_z = 0.0 #[deg]
        
        #match init
        self.odom_x_buff = 0.0
        self.odom_y_buff = 0.0
        self.ref_slam_x_buff = 0.0
        self.ref_slam_y_buff = 0.0
        self.ref_slam_diff = [0.0,0.0,0.0]
        
        self.match_buff = [0.0, 0.0, 0.0]
        self.set_waypoint = [0.0, 0.0, 0.0]
        
        self.wpxy_match_x = 0.0
        self.wpxy_match_y = 0.0

        self.last_match_x = 0.0
        self.last_match_y = 0.0

        self.last_ekf_match_x = 0.0
        self.last_ekf_match_y = 0.0

        self.match_per_threshold = 0.2 # use fusion match percentage 0.6 ground / rgb 0,35
        
        #image angle
        self.angle_offset = 0
        
        ## ekf
        self.GTheta = None
        self.GTheta0 = None
        #self.GPSthetayaw0 = 0
        #self.DGPStheta = 0
        self.w = None
        self.Q = None  # Process noise covariance
        self.H = None
        self.R = None
        self.R1 = 0.05**2  # High frequency sensor noise covariance
        self.R2 = 0.05**2  # Low frequency sensor noise covariance
        #self.R3 = 0  # High frequency sensor(heading)
        #self.R4 = 0  # Low frequency sensor(heading)
        self.P = None  # Initial covariance
        self.XX = None
        self.prev_time = None
        self.prev_pos = None
        self.Speed = 0
        self.SmpTime = 0.1
        self.GpsXY = None
        #self.GPS_conut = 0
        #self.GOffset = 0
        #self.offsetyaw = 0
        #self.combineyaw = 0
        self.robot_yaw = 0
        self.combyaw = 0
        self.robot_orientationz = 0
        self.robot_orientationw = 0
        #self.Number_of_satellites = 0
        self.kalf_speed_param = 1.0
        
        ### gps position init #########
        self.start_position_init_x = 0.0 #[m]
        self.start_position_init_y = 0.0#4.2 #[m]

        map_base_name = "waypoint_map_rgb"
        folder_path = os.path.expanduser('~/ros2_ws/src/map/nakaniwa_manual')

        # pngファイルを探索
        png_files = glob.glob(os.path.join(folder_path, '*.png'))
        png_file_count = len(png_files)
        map_file_path = os.path.join(folder_path, map_base_name)
        global_height_maps = []
        map_resolution = []
        map_origin = []
        map_occupied_thresh = []
        map_free_thresh = []
        for map_number in range(png_file_count):
            map_number_str = str(map_number).zfill(3)
            png_filename = os.path.join(
                folder_path,
                f'{map_file_path}_{map_number_str}.png'
            )
            print(f"png_filename: {png_filename}")
            global_height_map = cv2.imread(
                png_filename,
                cv2.IMREAD_COLOR
            )
            if global_height_map is None:
                print(f"failed load: {png_filename}")
                continue
            global_height_maps.append(global_height_map)
            yaml_filename = os.path.join(
                folder_path,
                f'{map_file_path}_{map_number_str}.yaml'
            )
            with open(yaml_filename, 'r') as yaml_file:
                map_yaml_data = yaml.safe_load(yaml_file)
            map_resolution.append(map_yaml_data['resolution'])
            map_origin.append(map_yaml_data['origin'])
            map_occupied_thresh.append(map_yaml_data['occupied_thresh'])
            map_free_thresh.append(map_yaml_data['free_thresh'])
        self.global_height_maps = np.stack(global_height_maps)
        self.global_reflect_map_resolution = np.stack(map_resolution)
        self.global_reflect_map_origin = np.stack(map_origin)
        self.global_reflect_map_occupied_thresh = np.stack(
            map_occupied_thresh
        )
        self.global_reflect_map_free_thresh = np.stack(
            map_free_thresh
        )
        wp_x = []
        wp_y = []
        wp_z = []
        for wp_number in range(self.global_height_maps.shape[0]):
            x_offset = (
                len(self.global_height_maps[wp_number][0])
                * self.global_reflect_map_resolution[0]
            ) / 2
            y_offset = (
                len(self.global_height_maps[wp_number][1])
                * self.global_reflect_map_resolution[0]
            ) / 2
            x = (
                self.global_reflect_map_origin[wp_number][0]
                + x_offset
            )
            y = (
                self.global_reflect_map_origin[wp_number][1]
                + y_offset
            )
            z = self.global_reflect_map_origin[wp_number][2]
            wp_x.append(x)
            wp_y.append(y)
            wp_z.append(z)
        self.waypoints = np.array([wp_x, wp_y, wp_z])
        print(f"self.waypoints ={self.waypoints}")
        
        #self.global_reflect_map_pgm = np.stack(global_reflect_map_pgms)
        #self.global_reflect_map_resolution = np.stack(map_resolution)
        #self.global_reflect_map_origin = np.stack(map_origin)
        #self.global_reflect_map_occupied_thresh = np.stack(map_occupied_thresh)
        #self.global_reflect_map_free_thresh = np.stack(map_free_thresh)
        #self.reflect_map_obs_matrices = np.stack(reflect_map_obs_matrices)
        
        self.action_client = ActionClient(self, StopFlag, 'stop_flag')  # ActionClientの設定
        self.action_sent = False  # アクションが送信されたかを追跡
        self.stop = False # stopするかの変数(True=stop, False=go)

        self.is_initialized = False

    def get_local_height_map(self, msg):
        t_stamp = msg.header.stamp
        reflect_map_local_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        now = self.get_clock().now()
        img_time = Time.from_msg(msg.header.stamp)
        delay = (now.nanoseconds - img_time.nanoseconds)/1e9
        print(f"image delay = {delay}")
        print(msg.header.stamp.sec)
        print(msg.header.stamp.nanosec)

        position_z = self.position_z
        theta_x = self.theta_x
        theta_y = self.theta_y
        theta_z = self.theta_z
        position_x=self.ekf_position_x; 
        position_y=self.ekf_position_y; 
        ekf_theta_z=self.ekf_theta_z;
        ekf_match_position_x = self.fused_msg.pose.pose.position.x
        ekf_match_position_y = self.fused_msg.pose.pose.position.y   
        #ekf_match_position_x = position_x
        #ekf_match_position_y = position_y

        # ==========================================================
        # local map crop
        # ==========================================================
        reflect_map_local_raw = crop_center(reflect_map_local_raw, 400, 400)
        reflect_map_local_raw = np.clip(reflect_map_local_raw, 0, 255).astype(np.uint8)

        # ==========================================================
        # global map
        # ==========================================================
        global_map_rgb = self.global_height_maps[self.current_waypoint]
        global_map_rgb = np.clip(global_map_rgb, 0, 255).astype(np.uint8)
        x_offset = (len(global_map_rgb[0]) * self.global_reflect_map_resolution[0]) / 2
        y_offset = (len(global_map_rgb[1]) * self.global_reflect_map_resolution[0]) / 2
        map_x = (self.global_reflect_map_origin[self.current_waypoint][0] + x_offset)
        map_y = (self.global_reflect_map_origin[self.current_waypoint][1] + y_offset)
        position_map = np.array([map_x, map_y, 0.0])
        MAP_RANGE_GL = x_offset
        map_ground_pixel = float(1.0 / self.global_reflect_map_resolution[self.current_waypoint])
        MAP_RANGE = (reflect_map_local_raw.shape[1] / map_ground_pixel) / 2.0

        # ==========================================================
        # multi layer matching
        # ==========================================================
        # global RGB layer
        global_rgb = global_map_rgb
        global_ground, global_mid, global_high = self.extract_layer_maps(global_rgb)

        # local RGB layer
        local_ground_raw, _, _ = self.extract_layer_maps(reflect_map_local_raw)
        
        # preprocess
        global_ground = self.preprocess_layer(global_ground)
        global_mid    = self.preprocess_layer(global_mid)
        global_high   = self.preprocess_layer(global_high)
        local_ground_raw = self.preprocess_layer(local_ground_raw)
        
        # ==========================================================
        # coarse matching (without rotation)
        # ==========================================================
        #coarse_result = cv2.matchTemplate(global_ground.astype(np.uint8), local_ground_raw.astype(np.uint8), cv2.TM_CCOEFF_NORMED) # ground main
        coarse_result = cv2.matchTemplate(global_rgb.astype(np.uint8), reflect_map_local_raw.astype(np.uint8), cv2.TM_CCOEFF_NORMED) # rgb main
        _, coarse_max_val, _, coarse_max_loc = cv2.minMaxLoc(coarse_result)
        #print(f"coarse_max_val = {coarse_max_val}")

        # ==========================================================
        # crop matched region from global map
        # ==========================================================
        h, w = reflect_map_local_raw.shape[:2]
        crop_x = coarse_max_loc[0]
        crop_y = coarse_max_loc[1]
        global_crop = global_map_rgb[crop_y:crop_y+h, crop_x:crop_x+w]
        global_ground_crop = global_ground[crop_y:crop_y+h, crop_x:crop_x+w]

        # サイズ不一致回避
        if (global_crop.shape[0] != h or global_crop.shape[1] != w):
            #print("global_crop size mismatch")
            return

        # ==========================================================
        # angle estimation using cropped map
        # ==========================================================
        #best_angle, best_score = self.find_best_rotation_angle(local_ground_raw, global_ground_crop, angle_range=10, step=0.5)
        #best_angle, dx, dy, best_score = self.estimate_affine_ecc(local_ground_raw, global_ground_crop) # ground main
        best_angle, dx, dy, best_score = self.estimate_affine_ecc(reflect_map_local_raw, global_crop) # rgb main
        #best_angle = 0 # test
        self.angle_offset = best_angle
        #print(f"best_angle = {best_angle}")
        #print(f"best_score = {best_score}")
    
        # ==========================================================
        # rotate local map
        # ==========================================================
        reflect_map_local = self.rotate_image(reflect_map_local_raw, best_angle)
        local_ground, local_mid, local_high = self.extract_layer_maps(reflect_map_local)
        local_ground = self.preprocess_layer(local_ground)
        local_mid = self.preprocess_layer(local_mid)
        local_high = self.preprocess_layer(local_high)

        # ==========================================================
        # coarse matching by ground layer
        # ==========================================================
        #candidates = self.get_topk_template_matches(global_ground, local_ground, map_ground_pixel, MAP_RANGE_GL, MAP_RANGE, position_map, top_k=5) # ground main
        candidates = self.get_topk_template_matches(global_rgb, reflect_map_local, map_ground_pixel, MAP_RANGE_GL, MAP_RANGE, position_map, top_k=5) # rgb main

        # ==========================================================
        # verify using upper layers
        # ==========================================================
        best_candidate = self.verify_candidates_multi_layer(candidates, global_ground, global_mid, global_high, local_mid, local_high, map_ground_pixel, MAP_RANGE_GL, MAP_RANGE, position_map)
        #best_candidate = candidates[0] # no use

        # ==========================================================
        # final result
        # ==========================================================
        if best_candidate is not None:
            ref_slam_x = best_candidate["x"]
            ref_slam_y = best_candidate["y"]
            #ref_slam_x += dx / map_ground_pixel # ECC
            #ref_slam_y -= dy / map_ground_pixel # ECC
            ref_slam_xyz = np.array([ref_slam_x, ref_slam_y, 0.0])
            match_percentage = best_candidate["total_score"]
        else:
            ref_slam_xyz = [None, None, None]
            match_percentage = 0.0
        print(f"match_percentage: {match_percentage}")

        # ==========================================================
        # no match
        # ==========================================================
        if ref_slam_xyz[0] is None:
            ref_slam_x = ekf_match_position_x
            ref_slam_y = ekf_match_position_y
            ref_slam_xyz = np.array([ref_slam_x, ref_slam_y, 0.0])
        else:
            ref_slam_x = ref_slam_xyz[0]
            ref_slam_y = ref_slam_xyz[1]
        self.ref_slam_x_buff = ref_slam_x
        self.ref_slam_y_buff = ref_slam_y
        self.ref_slam_diff = [ref_slam_x - ekf_match_position_x, ref_slam_y - ekf_match_position_y, 0]
        odom_ref_slam_msg = odometry_msg(ref_slam_x, ref_slam_y, position_z, theta_x, theta_y, theta_z + self.angle_offset, t_stamp, 'odom')
        #self.odom_ref_slam_publisher.publish(odom_ref_slam_msg)
        if match_percentage > self.match_per_threshold: # high priority matching 0.35? 0.4?
            self.GpsXY = np.array([ref_slam_x, ref_slam_y ])
            print(f"!!!!!high priority matching: {match_percentage}!!!!!")
            self.last_match_x = ref_slam_x
            self.last_match_y = ref_slam_y
            self.last_ekf_match_x = position_x
            self.last_ekf_match_y= position_y
            self.odom_ref_slam_publisher.publish(odom_ref_slam_msg)
        else:
            delta_x = position_x - self.last_ekf_match_x
            delta_y = position_y - self.last_ekf_match_y
            estimate_x = self.last_match_x + delta_x
            estimate_y = self.last_match_y + delta_y
            self.GpsXY = np.array([estimate_x, estimate_y])
        #self.robot_yaw = (theta_z + self.angle_offset) / 180 * math.pi
        self.robot_yaw = (ekf_theta_z + self.angle_offset) / 180 * math.pi
        #print(f"GpsXY = {self.GpsXY}")

    def make_gradient_feature(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
        return grad.astype(np.uint8)
        
        # サーバーにアクションを送信する関数
    def send_action_request(self):
        goal_msg = StopFlag.Goal()
        
        # stop変数の状態でaの値を決定
        if self.stop: # True
            goal_msg.a = 1  # stop
        else: # False
            goal_msg.a = 0  # go
            
        goal_msg.b = 2  # 任意の値を設定

        # アクションサーバーが利用可能になるまで待機
        self.action_client.wait_for_server()

        # アクションを非同期で送信
        self.future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.response_callback)

    # フィードバックを受け取るコールバック関数
    def feedback_callback(self, feedback):
        self.get_logger().info(f"Received feedback: {feedback.feedback.rate}")

    # 結果を受け取るコールバック関数
    def response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")

        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)

    # 結果のコールバック
    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.sum}")
        
        
    def waypoint_manager(self):
        self.time_stamp = self.get_clock().now().to_msg()
        #self.get_logger().info('waypoint manager cntl')
        ###### odom position ######
        #position_x=self.ekf_position_x; position_y=self.ekf_position_y; 
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z #-self.angle_offset;
        ###### ekf position ######
        position_x=self.ekf_position_x; position_y=self.ekf_position_y; 
        #theta_x=self.ekf_theta_x; theta_y=self.ekf_theta_y; theta_z=self.ekf_theta_z-self.angle_offset;
        ekf_theta_z=self.ekf_theta_z;
        ###### map ekf position ###
        #position_x= self.fused_msg.pose.pose.position.x
        #position_y= self.fused_msg.pose.pose.position.y
        #theta_z = self.robot_yaw
        
        
        #waypoint theta & dist
        #set_waypoint = [self.waypoints[0,self.current_waypoint] - (position_x - odom_position_x), self.waypoints[1,self.current_waypoint] - (position_y - odom_position_y), self.waypoints[2,self.current_waypoint]]
        #set_waypoint = self.waypoints[:,self.current_waypoint] - self.ref_slam_diff
        set_waypoint = self.waypoints[:,self.current_waypoint]
        
        relative_point_x = set_waypoint[0] - position_x
        relative_point_y = set_waypoint[1] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, self.waypoints[2,self.current_waypoint]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        waypoint_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        waypoint_dist = math.sqrt(relative_point_x**2 + relative_point_y**2)
        waypoint_theta = abs(waypoint_rad * (180 / math.pi))
        #self.get_logger().info('####### waypoint_theta : %f #######' % (waypoint_theta))
        
        #set judge dist
        if abs(waypoint_theta) > 90:
            determine_dist = 4.5
        else:
            determine_dist = 4.5
        #check if the waypoint reached
        if waypoint_dist < determine_dist:
            #self.current_waypoint += 1
            if self.current_waypoint < len(self.waypoints[0,:])-1:
                self.current_waypoint += 1
            else:
                # goal:stopをtrueにしてアクションを再送信
                self.stop = True
                self.get_logger().info("Stop flag reset to True")
                self.send_action_request()
        
            #if self.current_waypoint > (len(self.waypoints[0,:]) - 1):
            #    self.stop_flag = 1
            #    self.get_logger().info('GOAL : stop_flag = %f' % (self.stop_flag))
        #self.get_logger().info('current_waypoint:x = %f, y = %f : waypoint_no = %f' % (self.waypoints[0,self.current_waypoint], self.waypoints[1,self.current_waypoint], self.current_waypoint))
        
        #buff
        self.set_waypoint = set_waypoint
        
        #publish
        pose_array = self.current_waypoint_msg(set_waypoint, 'odom')
        self.current_waypoint_publisher.publish(pose_array)
        
    def get_odom(self, msg):
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
        
        
        ########### ekf ###########
        # current_time = self.get_clock().now().to_msg()
        diff_time_stamp = Clock(clock_type=ClockType.ROS_TIME).now()
        current_time = diff_time_stamp.nanoseconds / 1000000000
        if self.prev_time is not None:
            self.SmpTime = current_time - self.prev_time
        else:
            self.SmpTime = 0.1
        self.prev_time = current_time
                
        current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])
        if self.prev_pos is not None:
            distance = np.linalg.norm(current_pos - self.prev_pos)
            self.Speed = distance / self.SmpTime
        else:
            self.Speed = 0
        self.prev_pos = current_pos
        self.is_initialized = True # 初期化フラグを立てる
        
    def get_ekf_odom(self, msg):
        self.ekf_position_x = msg.pose.pose.position.x + self.start_position_init_x
        self.ekf_position_y = msg.pose.pose.position.y + self.start_position_init_y
        self.ekf_position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.ekf_theta_x = 0 #roll /math.pi*180
        self.ekf_theta_y = 0 #pitch /math.pi*180
        self.ekf_theta_z = yaw /math.pi*180
        
        
    def match_images(self, img1, img2, min_matches=5, match_threshold=20):
        
        # ORB特徴点検出器の生成
        orb = cv2.ORB_create()

        # 特徴点とディスクリプタの検出
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # BFMatcherを使用して特徴点をマッチング
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        
         # マッチング結果が少ない場合は処理を中止
        if len(matches) < min_matches:
            self.get_logger().warn(f"Not enough matches ({len(matches)}/{min_matches}). Skipping.")
            return None, None, None, None, None, None

        # 有効なマッチングのみを抽出
        good_matches = [m for m in matches if m.distance < match_threshold]

        # 有効なマッチングが少ない場合は処理を中止
        if len(good_matches) < min_matches:
            self.get_logger().warn(f"Not enough good matches ({len(good_matches)}/{min_matches}). Skipping.")
            return None, None, None, None, None, None
        
        # マッチングされた特徴点を抽出
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # ホモグラフィ行列の計算
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 回転角度の計算
        angle = -np.degrees(np.arctan2(M[1, 0], M[0, 0]))

        # 平行移動量の計算
        tx = M[0, 2]
        ty = M[1, 2]
        
        return angle, tx, ty, good_matches, keypoints1, keypoints2
    def sad_score(self, img1, img2): 
        return np.sum(np.abs(img1.astype(np.int16) - img2.astype(np.int16)))
    
    def find_best_rotation_angle(self, img1, img2, angle_range, step): 
        best_angle = 0 
        min_sad = float('inf') 
        total_pixels = img1.shape[0] * img1.shape[1]
        angle_set = np.arange(-angle_range, angle_range+step, step)
        #print(f"angle_set: {angle_set}")
        for angle in angle_set: 
            M = cv2.getRotationMatrix2D((img1.shape[1] // 2, img1.shape[0] // 2), angle, 1.0) 
            rotated_img = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0])) 
            sad = self.sad_score(rotated_img, img2) 
            match_rate = (total_pixels - sad) / total_pixels * 100 # 絶対差の逆数として計算
            #print(f"angle:match_rate: {angle, match_rate}")
            if sad < min_sad: 
                min_sad = sad 
                best_angle = angle 
        self.angle_offset = best_angle
        #print(f"self.angle_offset: {self.angle_offset}")
        return best_angle, min_sad
    
    def calculate_match_rate(self, sad, total_pixels): 
        # SADスコアを基にマッチ率を計算 
        match_rate = 100 * (1 - sad / (total_pixels * 255)) 
        return match_rate 

    def find_best_match(self, img1, img2): 
        # 大きい方のサイズに合わせて画像をリサイズする 
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2) 
        w = max(w1, w2) 
        # テンプレートマッチングの初期化 
        best_sad = float('inf') 
        best_location = (0, 0) 
        for y in range(h2 - h1 + 1): 
            for x in range(w2 - w1 + 1): 
                # サブ画像を取り出してSADスコアを計算 
                sub_img = img2[y:y+h1, x:x+w1] 
                sad = self.sad_score(img1, sub_img) 
                if sad < best_sad: 
                    best_sad = sad 
                    best_location = (x, y) 
        # マッチ率を計算 
        match_rate = self.calculate_match_rate(best_sad, resized_img1.size) 
        return best_location, best_sad, match_rate
    
    
    def rotate_image(self, image, angle): 
        # 画像の中心を計算 
        (h, w) = image.shape[:2] 
        center = (w // 2, h // 2) 
        # 回転行列を生成 
        M = cv2.getRotationMatrix2D(center, angle, 1.0) 
        # 画像を回転 
        rotated_image = cv2.warpAffine(image, M, (w, h)) 
        return rotated_image
    
    
    def current_waypoint_msg(self, waypoint, set_frame_id):
        pose_array = geometry_msgs.PoseArray()
        pose_array.header.frame_id = set_frame_id
        pose_array.header.stamp = self.time_stamp
        pose = geometry_msgs.Pose()
        pose.position.x = waypoint[0]
        pose.position.y = waypoint[1]
        pose.position.z = waypoint[2]
        
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)
        
        return pose_array
        
    ############ ekf #############
    def orientation_to_yaw(self, z, w):
        yaw = np.arctan2(2.0 * (w * z), 1.0 - 2.0 * (z ** 2))
        return yaw

    def yaw_to_orientation(self, yaw):
        orientation_z = np.sin(yaw / 2.0)
        orientation_w = np.cos(yaw / 2.0)
        return orientation_z, orientation_w
        
    def initializeGPS(self, GpsXY, GTheta, SmpTime):
        self.GTheta0 = GTheta
        self.XX = np.array(
            [GpsXY[0], GpsXY[1], np.cos(GTheta), np.sin(GTheta)])
        self.w = np.array([(1.379e-3)**2, (0.03 * np.pi / 180 * SmpTime)**2])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.array(
            [[(1.379e-3)**2, 0], [0, (0.03 * np.pi / 180 * SmpTime)**2]])
        G0 = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
        self.P = G0 @ self.Q @ G0.T
        
    def KalfGPSXY(self, Speed, SmpTime, GTheta, GpsXY, R1, R2):
        if self.H is None:
            self.initializeGPS(GpsXY, GTheta, SmpTime)

        self.R = np.array([[R1, 0], [0, R2]])

        DTheta = GTheta - self.GTheta0
        self.GTheta0 = GTheta

        # equation of state F G
        F = np.array([
            [1, 0, Speed * SmpTime *
                np.cos(DTheta), -Speed * SmpTime * np.sin(DTheta)],
            [0, 1, Speed * SmpTime *
                np.sin(DTheta), Speed * SmpTime * np.cos(DTheta)],
            [0, 0, np.cos(DTheta), -np.sin(DTheta)],
            [0, 0, np.sin(DTheta), np.cos(DTheta)]
        ])

        G = np.array([
            [np.cos(GTheta), -Speed * SmpTime * np.sin(GTheta)],
            [np.sin(GTheta), Speed * SmpTime * np.cos(GTheta)],
            [0, -np.sin(GTheta)],
            [0, np.cos(GTheta)]
        ])

        Y = np.array([GpsXY[0], GpsXY[1]])

        self.XX = F @ self.XX  # filter equation
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T  # Prior Error Covariance
        # kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @
                                              self.P @ self.H.T + self.R)
        self.XX = self.XX + K @ (Y - self.H @ self.XX)  # estimated value
        self.P = self.P - K @ self.H @ self.P  # Posterior Error Covariance

        return self.XX[:2]
        
    def publish_fused_value(self):
        #if self.Speed is not None and self.SmpTime is not None and self.GTheta is not None:
        if self.Speed is not None and self.SmpTime is not None :
            if self.GpsXY is not None :
                kalf_speed = self.Speed * self.kalf_speed_param
                fused_value = self.KalfGPSXY(
                    kalf_speed, self.SmpTime, self.robot_yaw, self.GpsXY, self.R1, self.R2)    
                
                robot_orientation = self.yaw_to_orientation(self.robot_yaw)
                self.robot_orientationz = robot_orientation[0]
                self.robot_orientationw = robot_orientation[1]

                self.fused_msg.pose.pose.position.x = float(fused_value[0])
                self.fused_msg.pose.pose.position.y = float(fused_value[1])
                self.fused_msg.pose.pose.orientation.z = float(
                    self.robot_orientationz)
                self.fused_msg.pose.pose.orientation.w = float(
                    self.robot_orientationw)
                
                self.fused_msg.header.stamp = self.get_clock().now().to_msg()
                self.fused_msg.header.frame_id = "odom"
                self.fused_pub.publish(self.fused_msg)
                #print(f"fused_value: {fused_value}")
        #print("publish_fused_value called")
        #print(f"GpsXY = {self.GpsXY}")
    
    def extract_layer_maps(self, reflect_map_obs):
        # OpenCVはBGR順
        ground = reflect_map_obs[:, :, 0]
        mid    = reflect_map_obs[:, :, 1]
        high   = reflect_map_obs[:, :, 2]
        return ground, mid, high

    def preprocess_layer(self, img):
        #img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        #img = img.astype(np.uint8)
        # ノイズ除去
        img = img.astype(np.uint8)

        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

        img = cv2.GaussianBlur(img, (15,15), 0)
        #img = cv2.GaussianBlur(img, (5,5), 0)
        return img

    def calc_entropy_score(self, img):
        lap = cv2.Laplacian(img, cv2.CV_64F)
        score = lap.var()
        return score

    def get_topk_template_matches(
            self,
            global_map,
            local_map,
            pixel,
            global_map_range,
            local_map_range,
            position_map,
            top_k=5):
        result = cv2.matchTemplate(
            global_map.astype(np.uint8),
            local_map.astype(np.uint8),
            cv2.TM_CCOEFF_NORMED # TM_CCOEFF_NORMED / TM_SQDIFF_NORMED
        )
        candidates = []
        result_copy = result.copy()
        for _ in range(top_k):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_copy)
            max_loc_x = max_loc[0] / pixel
            max_loc_y = max_loc[1] / pixel
            pos_x = (
                position_map[0]
                - global_map_range
                + max_loc_x
                + local_map_range
            )
            pos_y = (
                position_map[1]
                + global_map_range
                - max_loc_y
                - local_map_range
            )
            candidates.append({
                "score": float(max_val), # max_val / 1.0 - min_val
                "x": float(pos_x),
                "y": float(pos_y),
                "loc": max_loc
            })

            # 近傍抑制
            cv2.circle(
                result_copy,
                max_loc,
                30,
                -1,
                thickness=-1
            )

        return candidates

    def verify_candidates_multi_layer(
            self,
            candidates,
            global_ground,
            global_mid,
            global_high,
            local_mid,
            local_high,
            pixel,
            global_map_range,
            local_map_range,
            position_map):

        best_candidate = None
        best_total_score = -9999

        for cand in candidates:

            score_ground = cand["score"]

            x_pix = int(
                (
                    cand["x"]
                    - position_map[0]
                    + global_map_range
                    - local_map_range
                ) * pixel
            )

            y_pix = int(
                (
                    position_map[1]
                    + global_map_range
                    - cand["y"]
                    - local_map_range
                ) * pixel
            )

            h = local_mid.shape[0]
            w = local_mid.shape[1]

            # 範囲外チェック
            if (
                x_pix < 0 or
                y_pix < 0 or
                x_pix + w >= global_mid.shape[1] or
                y_pix + h >= global_mid.shape[0]
            ):
                continue

            crop_mid = global_mid[
                y_pix:y_pix+h,
                x_pix:x_pix+w
            ]

            crop_high = global_high[
                y_pix:y_pix+h,
                x_pix:x_pix+w
            ]

            # mid verification
            res_mid = cv2.matchTemplate(
                crop_mid,
                local_mid,
                cv2.TM_CCOEFF_NORMED
            )

            _, mid_score, _, _ = cv2.minMaxLoc(res_mid)

            # high verification
            res_high = cv2.matchTemplate(
                crop_high,
                local_high,
                cv2.TM_CCOEFF_NORMED
            )

            _, high_score, _, _ = cv2.minMaxLoc(res_high)

            # 情報量評価
            entropy_mid = self.calc_entropy_score(local_mid)
            entropy_high = self.calc_entropy_score(local_high)

            # sparse layerは重みを下げる
            if entropy_mid < 30:
                mid_weight = 0.0
            else:
                mid_weight = 0.5

            if entropy_high < 30:
                high_weight = 0.0
            else:
                high_weight = 0.7

            total_score = (
                score_ground
                + mid_weight * mid_score
                + high_weight * high_score
            )

            # temporal consistency
            dist = np.sqrt(
                (cand["x"] - self.ref_slam_x_buff)**2 +
                (cand["y"] - self.ref_slam_y_buff)**2
            )

            dynamic_limit = max(1.0, self.Speed * 3.0)

            if dist > 1.5: # 1.0 / dynamic_limit
                continue

            cand["total_score"] = total_score

            if total_score > best_total_score:
                best_total_score = total_score
                best_candidate = cand

        return best_candidate
    
    def estimate_affine_ecc(self, local_img, global_img):

        local_gray = local_img.astype(np.float32) / 255.0
        global_gray = global_img.astype(np.float32) / 255.0

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            50,
            1e-5
        )

        try:
            cc, warp_matrix = cv2.findTransformECC(
                global_gray,
                local_gray,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria
            )

            dx = warp_matrix[0, 2]
            dy = warp_matrix[1, 2]

            angle = np.degrees(
                np.arctan2(
                    warp_matrix[1,0],
                    warp_matrix[0,0]
                )
            )

            return angle, dx, dy, cc

        except cv2.error:
            return 0.0, 0.0, 0.0, 0.0

def crop_center(image, crop_width, crop_height): 
    # 画像の高さと幅を取得 
    height, width = image.shape[:2]
    # 中央の座標を計算 
    center_x, center_y = width // 2, height // 2 
    # 切り抜きの左上と右下の座標を計算 
    x1 = center_x - (crop_width // 2) 
    y1 = center_y - (crop_height // 2) 
    x2 = center_x + (crop_width // 2) 
    y2 = center_y + (crop_height // 2) 
    # 画像を切り抜く 
    cropped_image = image[y1:y2, x1:x2] 
    return cropped_image

def path_msg(waypoints, stamp, parent_frame):
    wp_msg = nav_msgs.Path()
    wp_msg.header.frame_id = parent_frame
    wp_msg.header.stamp = stamp
        
    # ウェイポイントを追加
    for i in range(waypoints.shape[1]):
        waypoint = geometry_msgs.PoseStamped()
        waypoint.header.frame_id = parent_frame
        waypoint.header.stamp = stamp
        waypoint.pose.position.x = waypoints[0, i]
        waypoint.pose.position.y = waypoints[1, i]
        waypoint.pose.position.z = 0.0
        waypoint.pose.orientation.w = 1.0
        wp_msg.poses.append(waypoint)
    return wp_msg

def odometry_msg(pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, stamp, frame_id):
    odom_msg = nav_msgs.Odometry()
    odom_msg.header.stamp = stamp
    odom_msg.header.frame_id = frame_id
    
    # 位置情報を設定
    odom_msg.pose.pose.position.x = pos_x 
    odom_msg.pose.pose.position.y = pos_y
    odom_msg.pose.pose.position.z = pos_z
    
    # YawをQuaternionに変換
    roll = theta_x /180*math.pi
    pitch = theta_y /180*math.pi
    yaw = theta_z /180*math.pi
    quat = transforms3d.euler.euler2quat(roll, pitch, yaw)
    odom_msg.pose.pose.orientation = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
    
    return odom_msg


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
        
# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    waypoint_manager_maprun = WaypointManagerMaprun()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(waypoint_manager_maprun)
    # 明示的にノードの終了処理を行います。
    waypoint_manager_maprun.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()