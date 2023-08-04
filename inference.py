            ## coding: UTF-8
import os
import tqdm
import argparse
import pathlib
import json
import glob
import cv2
import time
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

#from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants

from models import layers
from models.model import E2PoseModel
from utils import tf_util, draw
from utils.define import POSE_DATASETS

#--------------
# Parse args
#--------------
def parse_args():
    from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('--model'          , type=pathlib.Path)
    parser.add_argument('--dst'            , type=pathlib.Path, default=pathlib.Path('./outsample'))
    parser.add_argument('--src'            , type=str         , default='none')
    parser.add_argument('--backbone'       , type=str         , default='ResNet50')
    parser.add_argument('--dataset'        , type=str         , default='COCO')
    parser.add_argument('--dev'            , type=strtobool   , default=False)
    parser.add_argument('--add_fps'        , type=strtobool   , default=True)
    parser.add_argument('--add_blur'       , type=strtobool   , default=False)
    parser.add_argument('--tftrt'          , type=strtobool   , default=False)
    args = parser.parse_args()
    if not args.model.exists():
        print(args.model)
        raise ValueError('inference model is not found')
    args.src      = glob.glob(args.src)
    args.dev      = bool(args.dev)
    args.add_fps  = bool(args.add_fps)
    args.add_blur = bool(args.add_blur)
    args.tftrt    = bool(args.tftrt)
    return args


class E2PoseInference():
    def __init__(self, graph_path):
        print(graph_path)
    
    def decode(self, pred, src_hw, th=0.5):
        pv, kpt      = pred
        pv           = np.reshape(pv[0], [-1])
        kpt          = kpt[0][pv>=th]
        kpt[:,:,-1] *= src_hw[0]
        kpt[:,:,-2] *= src_hw[1]
        kpt[:,:,-3] *= 2
        ret = []
        for human in kpt:
            mask   = np.stack([(human[:,0] >= th).astype(np.float32)], axis=-1)
            human *= mask
            human  = np.stack([human[:,_ii] for _ii in [1,2,0]], axis=-1)
            ret.append({'keypoints': np.reshape(human, [-1]).tolist(), 'category_id':1})
        return ret

#TensorRTに最適化されたモデルの推論を行うための機能を提供
class E2PoseInference_by_trt(E2PoseInference):
    def __init__(self, model_path):
        self.graph_model = tf.saved_model.load(str(model_path), tags=[tag_constants.SERVING]).signatures['serving_default']
        self.model       = convert_to_constants.convert_variables_to_constants_v2(self.graph_model)
        self.check_output_idx()
    
    def predict(self, x):            
        y = self.model(tf.constant(x, dtype=np.float32))
        return y
        
    def check_output_idx(self):
        output       = self.predict(np.zeros([1,] + [_v for _v in self.inputs[0].shape[1:]], dtype=np.float32))
        self.ret_idx = [[ii for ii, _o in enumerate(output) if _o.shape[-1] == 1][0], [ii for ii, _o in enumerate(output) if _o.shape[-1] == 3][0]]

    @property
    def inputs(self):
        return self.graph_model.inputs
    
    def __call__(self, x, training=False):
        output = self.predict(x)
        return [output[idx].numpy() for idx in self.ret_idx]
    
    def __call__(self, x, training=False):
        output = self.predict(x)
        humans = self.decode(output, src_hw=[x.shape[2], x.shape[3]], th=0.5)

        feet_data = []
        for human in humans:
            keypoints = human['keypoints']
            # 足のデータのインデックス: 15, 16
            feet = keypoints[15:17]
            feet_data.append(feet)

        return feet_data


class E2PoseInference_by_pb(E2PoseInference):
    def __init__(self, graph_path):
        with tf.io.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        self.graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name='e2pose')
        self.persistent_sess = tf.compat.v1.Session(graph=self.graph, config=None)

        input_op           = [op for op in self.graph.get_operations() if 'inputimg' in op.name][0]
        out_pv_op          = [op for op in self.graph.get_operations() if 'pv/concat' in op.name][-1]
        out_kpt_op         = [op for op in self.graph.get_operations() if 'kvxy/concat' in op.name][-1]
        self.tensor_image  = input_op.outputs[0]
        self.tensor_output = [out_pv_op.outputs[0], out_kpt_op.outputs[0]]
    
    @property
    def inputs(self):
        return [self.tensor_image]

    def __call__(self, x, training=False):
        return self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: x})
    

def load_model(args):
    frozen_path  = args.model.parent / 'frozen_model.pb'
    trt_path     = args.model.parent / 'trt_model'
    builded_trt  = args.model.parent / 'trt_build'
    h5_path      = args.model.parent / 'keras_model.hdf5'
    tf_path      = args.model.parent / 'saved_model'
    if args.tftrt:
        inference_model_path = trt_path
        #inference_model_path = builded_trt
    else:
        inference_model_path = frozen_path
    if not inference_model_path.exists():
        if not frozen_path.is_file():
            model = tf.keras.models.load_model(str(args.model), compile=False, custom_objects={'E2PoseModel':E2PoseModel, 'VariableDropout':layers.VariableDropout, 'PixelShuffler':layers.PixelShuffler, 'CropOrPad':layers.CropOrPad})
            if str(args.model) not in [str(_path) for _path in [frozen_path,trt_path,h5_path,tf_path,builded_trt]]:
                # add preprocess layer
                fn_preprosess = E2PoseModel.get_preprosess(args.backbone)
                tx_in  = tf.keras.Input(shape=model.inputs[0].shape[1:], batch_size=1, name='u8img')
                tx     = tf.keras.layers.Lambda(lambda x: fn_preprosess(x), name='preprocess')(tx_in)
                tx_out = model(tx)
                model  = tf.keras.models.Model(tx_in, tx_out, name='e2pose')
            model.save(str(h5_path))
            model.save(str(tf_path))
            tf_util.convert_kerasmodel_to_frozen_pb(model, frozen_path)
            del model
            tf.keras.backend.clear_session()
        if args.tftrt:
            if not tf_path.exists():
                tf_util.convert_frozen_to_savedmodel(frozen_path, tf_path)
            tf_util.convert_savedmodel_to_trtmodel(tf_path, trt_path)
            tf_util.convert_savedmodel_to_trtmodel(tf_path, builded_trt, build=True)
        tf.keras.backend.clear_session()
    model = E2PoseInference_by_trt(inference_model_path) if args.tftrt else E2PoseInference_by_pb(inference_model_path)
    return model


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"({x1}, {y1}) --> ({x2}, {y2})")
    print(f"({x2-x1}, {y2-y1})")


def select_tennis_court(img):
    """
    Select the tennis court on the first frame and return the coordinates
    """
    coords = []

    def onselect(verts):
        # vertsには選択した頂点のリストが格納されます。リストの各要素は(x, y)の形式です。
        coords.extend([(int(vert[0]), int(vert[1])) for vert in verts])

    fig, ax = plt.subplots()
    ax.imshow(img)

    polygon_selector = PolygonSelector(ax, onselect)
    plt.show()

    # 選択した4つの頂点を返します。頂点が4つ以外の場合、適切にエラーハンドリングを行ってください。
    return coords

def draw_court_on_frame(image, court_coords):
    # OpenCVのpolylines関数を使用して四角形を描画します。
    # court_coordsは(x, y)の形式の頂点のリストであると仮定します。
    pts = np.array(court_coords, np.int32)
    pts = pts.reshape((-1, 1, 2))  # OpenCVのpolylines関数はこの形式の入力を要求します。
    image = cv2.polylines(image, [pts], True, (0, 255, 0), 2)  # 緑色で2pxの太さで描画
    return image


def draw_footprints(image, foot_data_list, current_frame, frame_rate=30):
    """
    This function draws the footprints on the image.

    Args:
    - image: The image on which to draw the footprints.
    - foot_data_list: The list containing the foot data.
    - current_frame: The current frame number.
    - frame_rate: The frame rate of the video.
    """
    # For each set of foot data in the foot data list...
    for foot_data in foot_data_list:
    # If the frame number of the foot data is less than or equal to the current frame number and greater than the current frame number minus frame_rate...
        if foot_data['Frame'] <= current_frame and foot_data['Frame'] > current_frame - frame_rate/10:
            # Calculate the midpoint of the positions of the left and right foot.
            mid_point_x = int((foot_data['LeftFoot_X'] + foot_data['RightFoot_X']) / 2)
            mid_point_y = int((foot_data['LeftFoot_Y'] + foot_data['RightFoot_Y']) / 2)
            
            # Draw a circle at the midpoint of the positions of the left and right foot.
            cv2.circle(image, (mid_point_x, mid_point_y), radius=5, color=(255, 0, 0), thickness=-1)

    return image





   



#--------------
# Main
#--------------
if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    model   = load_model(args)
    painter = draw.Painter(args.dataset)
    
    with draw.seaquence_writer(args.dst, dev=args.dev) as writer:
        # Create a list to store the foot data
        foot_data_list = []
        
        first_frame = True
        for ii, (src_path, raw, frame) in tqdm.tqdm(enumerate(draw.read_src(args.src, resize=model.inputs[0].shape[1:3]))):
            if first_frame:
                # Select the tennis court coordinates from the original image
                court_coords = select_tennis_court(raw)
                first_frame = False

            stt       = time.perf_counter()
            pred      = model(np.stack([frame], axis=0))
            humans    = model.decode(pred, raw.shape[:2])
            pred_time = time.perf_counter() - stt

            # Draw the tennis court based on the selected coordinates
            raw = draw_court_on_frame(raw, court_coords)
            footprint = draw_footprints(raw, foot_data_list, ii, frame_rate=30)

            if args.add_blur:
                _size = [int(_v/70) for _v in raw.shape[:2][::-1]]
                raw   = cv2.blur(raw, tuple(_size))
            image     = painter(raw, humans)
            if args.add_fps:
                image = painter.add_fps_text(image, 1/pred_time)
            writer(src_path, footprint, {'time':pred_time, 'humans':humans})

            # Update foot_data_list with the positions of the feet
            for human in humans:
                keypoints = np.array(human['keypoints']).reshape(-1, 3)
                # 足のデータのインデックス: 15, 16
                left_foot = keypoints[15]
                right_foot = keypoints[16]
                foot_data_list.append({
                    'Frame': ii,
                    'LeftFoot_X': left_foot[0],
                    'LeftFoot_Y': left_foot[1],
                    'RightFoot_X': right_foot[0],
                    'RightFoot_Y': right_foot[1]
                })


            

        # 動画ファイル名を取得
        for src in args.src:
            video_filename = os.path.basename(src)
            base_name = os.path.splitext(video_filename)[0]
            csv_path = pathlib.Path(f'{args.dst}/csv/data_foot_{base_name}.csv')

            # foot_data_listの内容をCSVファイルに書き込む
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['Frame', 'LeftFoot_X', 'LeftFoot_Y', 'RightFoot_X', 'RightFoot_Y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(foot_data_list)
