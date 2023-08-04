import pandas as pd
import numpy as np
import argparse

def closest_time_index(time_array, video_time):
    idx = np.searchsorted(time_array, video_time, side='left')
    if idx > 0 and (idx == len(time_array) or np.fabs(video_time - time_array[idx-1]) <= np.fabs(video_time - time_array[idx])):
        return idx-1
    else:
        return idx

def merge_csv_efficiently(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Position_Yが11以上のデータをdf1から消去
    df1 = df1[df1['Position_Y'] < 11]

    new_position_x = []
    new_position_y = []

    time_array = df1['Time'].values

    for video_time in df2['Video Time']:
        idx = closest_time_index(time_array, video_time)
        
        # timeに一番近い値と、その前後1つずつの値を取得し、その3つの平均値を計算
        position_x_values = df1.loc[idx-1:idx+1, 'Position_X']
        position_y_values = df1.loc[idx-1:idx+1, 'Position_Y']

        new_position_x.append(position_x_values.mean())
        new_position_y.append(position_y_values.mean())

    df2['Near_Position_X'] = new_position_x
    df2['Near_Position_Y'] = new_position_y

    df2.to_csv(output_file, index=False)

def main(args):
    merge_csv_efficiently(args.file1, args.file2, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge two csv files.')
    parser.add_argument('--file1', type=str, help='Path to the first csv file.')
    parser.add_argument('--file2', type=str, help='Path to the second csv file.')
    parser.add_argument('--output', type=str, help='Path to the output csv file.')
    
    args = parser.parse_args()
    main(args)
