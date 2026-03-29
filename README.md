# Constructor_Hackathon_2026_GENAI
This is a repo for submission of Constructor University Hackathon 2026 GENAI Track 2 Autonomous driving

The help_function.py contains the function that used in the main script point_parse.py
Ensure hackathon_good_lap.mcap and hackathon_fast_laps.mcap are placed in the root directory. Then simply execute: python3 point_parse.py

The green string in the video is the trajectory of a professional driver (coach) projected onto absolute spatial coordinate system.

The red dots with white centers represent the places where the student made a conservative mistake in the previous lap. 

## Hightlight 1: Spatial Anchoring over Time-series
We used KD-Tree to achieve 3D absolute space mapping instead of traditional timestamp alignment, because the coach and the student are not in the same place at the same timestamp.

## Hightlight 2: 3D AR prrojection Engine
We did not rely on the 3D engines. We deconstructed the camera's intrinsic parameters $K$ and extrinsic parameters $[R|T]$, and use frustum Z-Culling to post the points in the video.

## Hightlight 3: Ready for High-Frequency Sensor Fusion
The current architecture is already running successfully based on 9-dimensional state tensor. Its fully ready to connect to 20Hz Badenia brake disc temperature and 250Hz Kistler slip angle for predicting tire grip limits.

## TODO
Refactor file ingestion pipeline to support dynamic CLI arguments via argparse for batch processing.
