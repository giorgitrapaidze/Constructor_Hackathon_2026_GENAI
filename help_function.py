from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
import numpy as np
from scipy.spatial import KDTree
import os
import matplotlib.pyplot as plt
import json
import cv2
from scipy.spatial.transform import Rotation as R



def rip_trajectory(mcap_file_path, target_topic):
    print(f"parsing pairs from {mcap_file_path} ...")
    trajectory = []
    
    with open(mcap_file_path, "rb") as f:
        for msg in read_ros2_messages(f):
            if msg.channel.topic == target_topic:
                ros_data = msg.ros_msg
                try:
                    x = ros_data.x_m
                    y = ros_data.y_m
                    z = ros_data.z_m
                    v = ros_data.v_mps
                    gas = ros_data.gas
                    brake = ros_data.brake
                    roll = ros_data.roll_rad
                    pitch = ros_data.pitch_rad
                    yaw = ros_data.yaw_rad
                    trajectory.append([x, y, z, v, gas, brake, roll, pitch, yaw])
                except AttributeError:
                    msg_dict = vars(ros_data)
                    if 'x_m' in msg_dict:
                        trajectory.append([msg_dict['x_m'], msg_dict['y_m'], msg_dict['z_m']])

    print(f"Parsing complete! Have {len(trajectory)} pairs\n")
    return np.array(trajectory)


def load_track_boundaries(json_path):
    print(f"Executing deep parse on {json_path}...")
    try:
        with open(json_path, 'r') as f:
            bnd_data = json.load(f)
        boundaries = bnd_data.get('boundaries', {})
        left_bound = np.array(boundaries.get('left_border', []))
        right_bound = np.array(boundaries.get('right_border', []))
        
        print(f"Boundaries loaded. Left points: {len(left_bound)}, Right points: {len(right_bound)}")
        return left_bound, right_bound
    except Exception as e:
        print(f"CRITICAL: Boundary JSON parse failed: {e}")
        return np.array([]), np.array([])
    

def extract_from_mcap(file_name, target_topic):
    name = file_name[:-5] + ".npy"
    if os.path.exists(name):
        print(f"Have {name}, load data from {name}")
        data = np.load(name)
        return data
    else:
        print(f"No {name}")
        data = rip_trajectory(file_name, target_topic)
        np.save(name, data)
        print(f"Data are saved into {name}!\n")
        return data


def rip_and_render_video(mcap_file_path, points_3d, camera_matrix, dist_coeffs, student_data, early_brake_points):
    print(f"[Agent Vision] Online. Syncing telemetry & video from {mcap_file_path}...")
    cam_topic = "/constructor0/sensor/camera_fl/compressed_image"
    
    frame_idx = 0
    max_state_idx = len(student_data) - 1
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_video = cv2.VideoWriter("Agent_Final_Lap.mp4", fourcc, 10.0, (1506, 728))

    with open(mcap_file_path, "rb") as f:
        for msg in read_ros2_messages(f):
            if msg.channel.topic == cam_topic:
                raw_data = msg.ros_msg.data
                np_arr = np.frombuffer(raw_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    state_idx = min(frame_idx * 10, max_state_idx)
                    current_state = student_data[state_idx]
                    
                    stu_pos = current_state[0:3]   
                    stu_rpy = current_state[6:9]   
                    stu_v_kmh = current_state[3] * 3.6
                    stu_gas = current_state[4] * 100.0
                    stu_brake = current_state[5] * 100.0
                    stu_brake_bar = current_state[5] / 100000.0
                    
                    rot_car = R.from_euler('xyz', stu_rpy, degrees=False) 
                    cam_offset = np.array([1.5, 0.3, 1.0])
                    cam_pos_world = stu_pos + rot_car.apply(cam_offset)

                    dcm_optical_to_car = np.array([
                        [ 0.0,  0.0,  1.0], 
                        [-1.0,  0.0,  0.0], 
                        [ 0.0, -1.0,  0.0]  
                    ], dtype=np.float64)
                    
                    rot_optical = R.from_matrix(dcm_optical_to_car)
                    rot_cam_world = rot_car * rot_optical
                    
                    rot_view = rot_cam_world.inv()
                    tvec = rot_view.apply(-cam_pos_world).reshape(3, 1)
                    rvec = rot_view.as_rotvec().reshape(3, 1)

                    points_cam = rot_view.apply(points_3d - cam_pos_world)
                    front_mask = (points_cam[:, 2] > 2.0) & (points_cam[:, 2] < 80.0) 
                    visible_points_3d = points_3d[front_mask]
                    
                    if len(visible_points_3d) > 0:
                        img_pts, _ = cv2.projectPoints(visible_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
                        img_pts = img_pts.squeeze().reshape(-1, 2)
                        
                        height, width = 728, 1506
                        v_mask = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < width) & \
                                 (img_pts[:, 1] >= 0) & (img_pts[:, 1] < height)
                        valid_pts = img_pts[v_mask].astype(int)
                        
                        for pt in valid_pts:
                            cv2.circle(frame, tuple(pt), radius=3, color=(0, 255, 0), thickness=-1)

                    
                    warning_triggered = False

                    if len(early_brake_points) > 0:
                        eb_cam = rot_view.apply(early_brake_points - cam_pos_world)
                        eb_mask = (eb_cam[:, 2] > 2.0) & (eb_cam[:, 2] < 80.0)
                        vis_eb_3d = early_brake_points[eb_mask]
                        z_mask = (eb_cam[:, 2] > 15.0) & (eb_cam[:, 2] < 35.0) 
                        x_mask = np.abs(eb_cam[:, 0]) < 2.0
                        close_danger_mask = z_mask & x_mask
                        if np.any(close_danger_mask):
                            warning_triggered = True

                        if len(vis_eb_3d) > 0:
                            eb_img_pts, _ = cv2.projectPoints(vis_eb_3d, rvec, tvec, camera_matrix, dist_coeffs)
                            eb_img_pts = eb_img_pts.squeeze().reshape(-1, 2)
                            
                            eb_v_mask = (eb_img_pts[:, 0] >= 0) & (eb_img_pts[:, 0] < width) & \
                                        (eb_img_pts[:, 1] >= 0) & (eb_img_pts[:, 1] < height)
                            valid_eb_pts = eb_img_pts[eb_v_mask].astype(int)
                            
                            for pt in valid_eb_pts:
                                cv2.circle(frame, tuple(pt), radius=10, color=(0, 0, 255), thickness=-1)
                                cv2.circle(frame, tuple(pt), radius=4, color=(255, 255, 255), thickness=-1) 
                        if warning_triggered:
                            cv2.putText(frame, "!!! EARLY BRAKING DETECTED !!!", (450, 350), 
                                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                    
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (20, 20), (450, 180), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    cv2.putText(frame, "CONSTRUCTOR GENAI AGENT - ALPHA", (40, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    cv2.putText(frame, f"SPEED : {stu_v_kmh:>5.1f} KM/H", (40, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"THROTTLE : {stu_gas:>5.1f} %", (40, 135), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if stu_gas>10 else (255,255,255), 2)
                    cv2.putText(frame, f"BRAKE    : {stu_brake_bar:>5.1f} BAR", (40, 170), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if stu_brake_bar>2.0 else (255,255,255), 2)
                    

                    out_video.write(frame)
                    cv2.imshow("Agent AR HUD - Hackathon Alpha", frame)
                    frame_idx += 1
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
    cv2.destroyAllWindows()
    out_video.release()
    print("Video stream ended and successfully saved to Agent_Final_Lap.mp4")