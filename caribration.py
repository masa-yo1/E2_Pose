import csv
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
import cv2
import sys
import os


#python .\caribration.py ./Outputs/csv/data_foot_movie12_opposite.csv 


def calculate_homography(court_coords, target_coords):
    # Convert lists of tuples to numpy arrays
    court_coords_array = np.float32(court_coords)
    target_coords_array = np.float32(target_coords)

    # Calculate homography matrix
    h, status = cv2.findHomography(court_coords_array, target_coords_array)

    return h

def convert_coordinates(coords, h):
    # Convert the list of tuples to a homogeneous numpy array
    coords_array = np.float32(coords).reshape(-1, 1, 2)

    # Apply the homography to the coordinates
    converted_coords_array = cv2.perspectiveTransform(coords_array, h)

    # Convert the homogeneous numpy array back to a list of tuples
    converted_coords = [tuple(coord[0]) for coord in converted_coords_array]

    return converted_coords

def convert_dataset(input_csv_path, output_csv_path, target_coords):
    with open(input_csv_path, 'r') as input_file, open(output_csv_path, 'w', newline='') as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = ['Frame', 'Time','Position_X','Position_Y' , 'Court_Coords', 'Converted_Court_Coords', 'Converted_LeftFoot_X', 'Converted_LeftFoot_Y', 'Converted_RightFoot_X', 'Converted_RightFoot_Y']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            frame = int(row['Frame'])
            time = float(row['Time'])
            left_foot_x = float(row['LeftFoot_X'])
            left_foot_y = float(row['LeftFoot_Y'])
            right_foot_x = float(row['RightFoot_X'])
            right_foot_y = float(row['RightFoot_Y'])
            court_coords = ast.literal_eval(row['Court_Coords'])  # string型の座標をtupleに変換

            # Calculate homography matrix for this frame
            h = calculate_homography(court_coords, target_coords)

            converted_court_coords = convert_coordinates(court_coords, h)
            converted_left_foot_x, converted_left_foot_y = convert_coordinates([(left_foot_x, left_foot_y)], h)[0]
            converted_right_foot_x, converted_right_foot_y = convert_coordinates([(right_foot_x, right_foot_y)], h)[0]
            
            new_row = {
                'Frame': frame,
                'Time': time,
                'Position_X': (converted_left_foot_x + converted_right_foot_x) / 2,
                'Position_Y': (converted_left_foot_y + converted_right_foot_y) / 2,
                'Converted_Court_Coords': converted_court_coords,
                'Converted_LeftFoot_X': converted_left_foot_x,
                'Converted_LeftFoot_Y': converted_left_foot_y,
                'Converted_RightFoot_X': converted_right_foot_x,
                'Converted_RightFoot_Y': converted_right_foot_y
            }
            writer.writerow(new_row)

def get_player_position(row):
    left_foot_x = float(row['Converted_LeftFoot_X'])
    left_foot_y = float(row['Converted_LeftFoot_Y'])
    right_foot_x = float(row['Converted_RightFoot_X'])
    right_foot_y = float(row['Converted_RightFoot_Y'])

    # Calculate the midpoint of the left foot and the right foot
    player_x = (left_foot_x + right_foot_x) / 2
    player_y = (left_foot_y + right_foot_y) / 2

    return player_x, player_y

# Specify the coordinates after conversion
target_coords = [(-5.485, 0), (5.485, 0), (5.485, 23.77), (-5.485, 23.77)]

# コマンドライン引数からの入力ファイル名と出力ファイル名を取得
# sys.argvはリストで、[0]はスクリプトの名前、[1]は最初の引数、[2]は2番目の引数
input_csv_path = sys.argv[1]  # './Outputs/csv/data_foot_movie12_mask.csv'

# ファイルパスからファイル名を取得し、拡張子を削除
base_filename = os.path.splitext(os.path.basename(input_csv_path))[0]

# ファイル名に接尾語を追加して新しい出力ファイル名を生成
output_csv_path = f'./Outputs/calibration_csv/caribrate_{base_filename}.csv'  # './Outputs/calibration_csv/data_foot_movie12_mask_converted.csv'
output_video_path = f'./Outputs/calibration_videos/{base_filename}.gif'  # '.\Outputs\calibration_videos\output.gif'

# Convert the values in the dataset
convert_dataset(input_csv_path, output_csv_path, target_coords)


# Load data from the converted CSV file
with open(output_csv_path, 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Remove frames where the player's y-coordinate is more than 11
data = [row for row in data if float(row['Position_Y']) < 11]

# Prepare for the plot
fig, ax = plt.subplots()

# Get the coordinates of the court from the data of the first frame
court_coords = ast.literal_eval(data[0]['Converted_Court_Coords'])

# Draw the court
court = patches.Rectangle((court_coords[0][0], court_coords[0][1]), court_coords[1][0] - court_coords[0][0], court_coords[2][1] - court_coords[0][1], fill=False)
ax.add_patch(court)

# Get the initial position of the player
player_x, player_y = get_player_position(data[0])

# Plot the position of the player
player, = plt.plot(player_x, player_y, 'ro')

# Define the function to update the animation
def update(frame_number):
    player_x, player_y = get_player_position(data[frame_number])
    player.set_data(player_x, player_y)
    
    # Update frame number
    plt.title(f'Frame: {frame_number}')
    
    return player,

# Create the animation
ani = FuncAnimation(fig, update, frames=range(len(data)), blit=True)

# Set plot limits larger than the court
court_width = court_coords[1][0] - court_coords[0][0]
court_height = court_coords[2][1] - court_coords[0][1]
ax.set_xlim(court_coords[0][0] - 0.3 * court_width, court_coords[1][0] + 0.3 * court_width)
ax.set_ylim(court_coords[0][1] - 0.3 * court_height, court_coords[2][1] + 0.3 * court_height)

# Save the animation as a video
Writer = ani.save(output_video_path, writer='pillow', fps=30)

# Print the number of frames
print(f'Number of frames: {len(data)}')
