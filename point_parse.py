from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
import numpy as np
from scipy.spatial import KDTree
import os
import matplotlib.pyplot as plt
import json
import cv2
from help_function import rip_trajectory
from help_function import load_track_boundaries
from help_function import extract_from_mcap
from help_function import rip_and_render_video
from scipy.spatial.transform import Rotation as R

inner_bnd, outer_bnd = load_track_boundaries("yas_marina_bnd.json")
coach_file = "hachathon_fast_laps.npy"
student_file = "hachathon_good_lap.npy"
wheel_to_wheel = "hachathon_wheel_to_wheel.npy"

coach_data = extract_from_mcap("hackathon_fast_laps.mcap", "/constructor0/state_estimation")
print("The shape of coach data: ", coach_data.shape) 

student_data = extract_from_mcap("hackathon_good_lap.mcap", "/constructor0/state_estimation")
print("The shape of student data: ", student_data.shape) 

spatial_coach = coach_data[:, :2]
spatial_student = student_data[:, :2]

print("Creating KD-Tree index...")
coach_tree = KDTree(coach_data[:, :2]) 
print("KD-Tree finished.")

distances, indices = coach_tree.query(spatial_student)
optimal_velocity = coach_data[indices, 3]
optimal_gas = coach_data[indices, 4]
optimal_brake = coach_data[indices, 5]

speed_delta_kmh = (student_data[:, 3] - optimal_velocity) * 3.6 
early_brake_mask = (optimal_brake < 0.1) & (student_data[:, 5] > 0.4)
early_brake_points = spatial_student[early_brake_mask]

plt.figure(figsize=(14, 11), facecolor='#111111') 
ax = plt.gca()
ax.set_facecolor('#111111')

plt.figure(figsize=(12, 10))
plt.scatter(coach_data[:, 0], coach_data[:, 1], c='lime', s=1, label='Optimal Racing Line (Coach)')
plt.scatter(student_data[:, 0], student_data[:, 1], c='red', s=1, label='Actual Trajectory (Student)')
if inner_bnd.size > 0:
    plt.plot(inner_bnd[:, 0], inner_bnd[:, 1], color='#555555', linewidth=1.2, alpha=0.7, label='Inner Bound')
if outer_bnd.size > 0:
    plt.plot(outer_bnd[:, 0], outer_bnd[:, 1], color='#555555', linewidth=1.2, alpha=0.7, label='Outer Bound')

scatter = plt.scatter(spatial_student[:, 0], spatial_student[:, 1], 
                      c=speed_delta_kmh, cmap='RdYlGn', s=5, 
                      vmin=-20, vmax=0, zorder=3)

if len(early_brake_points) > 0:
    sparse_brakes = early_brake_points[::15] 
    plt.scatter(sparse_brakes[:, 0], sparse_brakes[:, 1], 
                c='crimson', marker='X', s=150, edgecolors='white', linewidths=1.5,
                zorder=4, label='WARNING: Early Braking')

cbar = plt.colorbar(scatter, fraction=0.03, pad=0.04)
cbar.set_label('Speed Delta (km/h) [Red=Slower, Green=Optimal]', color='white', fontsize=12)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')


plt.axis('equal')
plt.title("Yas Marina Circuit - Agent Spatial Anchoring", fontsize=16, fontweight='bold')
plt.xlabel("Global X Coordinate (meters)")
plt.ylabel("Global Y Coordinate (meters)")
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)



camera_matrix = np.array([
    [2555.26339, 0.0,        751.52582],
    [0.0,        2538.42728, 469.37862],
    [0.0,        0.0,        1.0      ]
], dtype=np.float64)
dist_coeffs = np.array([-0.38385, 0.1615, -0.00085, 0.00053, 0.0], dtype=np.float64)


points_3d = np.ascontiguousarray(coach_data[:, :3], dtype=np.float64)

early_brake_points_3d = student_data[early_brake_mask, :3]


rip_and_render_video(
    "hackathon_good_lap.mcap", 
    points_3d, 
    camera_matrix, 
    dist_coeffs, 
    student_data, 
    early_brake_points_3d
)

plt.show()