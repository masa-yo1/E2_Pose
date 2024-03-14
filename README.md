E2Pose for Tennis: Fully Convolutional Networks for End-to-End Tennis Player Pose Estimation
===================================================================================================

Introduction
------------

This project adapts the E2Pose model, originally proposed for multi-person pose estimation in autonomous driving, to the domain of tennis player pose estimation. The adapted model leverages the feature pyramid and original head architecture of E2Pose, which allows for a lightweight model that can be trained end-to-end and performed in real-time on a resource-limited platform during inference.

The primary goal of this project is to accurately estimate the position of tennis players on the court using a single camera placed behind the court. The model processes the video feed frame-by-frame, detects the players' positions, and outputs the coordinates of both feet for each player in each frame.

Usage
-----

1. Set up the environment and install the necessary dependencies as described in the original E2Pose repository.

2. Run the `inference_footprint.py` script with the following command:

    ```
    python inference_footprint.py --input_video /path/to/your/video.mp4 --output_csv /path/to/output/footprints.csv
    ```

    Replace `/path/to/your/video.mp4` with the path to your input video file and `/path/to/output/footprints.csv` with the desired path for the output CSV file.

3. When the script starts, a window will display the first frame of the video. Manually select the outer boundaries of the tennis court by clicking on the four corners of the court in a clockwise or counterclockwise order. Press 'Enter' to confirm the selection.

4. The script will then process the video frame-by-frame, detecting the positions of the players' feet. The output will be saved in the specified CSV file, with each row containing the following information:
   - Frame number
   - Timestamp
   - Player 1 left foot X coordinate
   - Player 1 left foot Y coordinate
   - Player 1 right foot X coordinate
   - Player 1 right foot Y coordinate
   - Player 2 left foot X coordinate
   - Player 2 left foot Y coordinate
   - Player 2 right foot X coordinate
   - Player 2 right foot Y coordinate
  


![Videotogif](https://github.com/masa-yo1/E2_Pose/assets/102569005/98282069-fce2-4019-8d69-16af17392759)
![movie12_footprint_AdobeExpress](https://github.com/masa-yo1/E2_Pose/assets/102569005/eaf40bb3-248f-4a76-97c9-4f02202c9709)


Citation
========

This code is based on the work of Masakazu Tobeta, Yoshihide Sawada, Ze Zheng, Sawa Takamuku, Naotake Natori. "E2Pose: Fully Convolutional Networks for End-to-End Multi-Person Pose Estimation". 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).


#Acknowledgements
================

We would like to express our gratitude to the authors of E2Pose for making their code available and for their groundbreaking work in the field of multi-person pose estimation. Their work has served as a valuable foundation for our adaptation of the model to the domain of tennis player pose estimation.
