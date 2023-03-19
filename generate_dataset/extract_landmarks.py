import mediapipe as mp
import numpy as np
from mediapy import read_video
from pathlib import Path
from pqdm.processes import pqdm
import itertools
import argparse

mp_holistic = mp.solutions.holistic

def get_unique_ids(connections):
    ids = []
    for connection in connections:
        ids.append(connection[0])
        ids.append(connection[1])
    return list(np.unique(ids))

POSE_IDS = get_unique_ids(mp_holistic.POSE_CONNECTIONS)
FACE_IDS = get_unique_ids(mp_holistic.FACEMESH_TESSELATION)
HAND_IDS = get_unique_ids(mp_holistic.HAND_CONNECTIONS)

def component_points(component, replace_with_landmark=None, length=None):
    ''' component.landmark returns the landmarks sorted by id '''
    if component is not None:
        lm = component.landmark
        return np.array([[p.x, p.y, p.z] for p in lm])
    else:
        if replace_with_landmark:
            lm = replace_with_landmark
            return np.repeat([[lm.x, lm.y, lm.z]], length, axis=0)
        else:
            if length is None:
                raise Exception("length must be provided")
            return np.zeros((length, 3))

def process_holistic(color_video_path, depth_video_path):
    video = read_video(color_video_path)
    w, h = video.shape[1:3]
    
    if depth_video_path:
        depth = read_video(depth_video_path)
    else:
        depth = None
    
    datas = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=False) as holistic:
        
        for i, frame in enumerate(video):
            # Extract skeleton landmarks
            results = holistic.process(frame)
            
            # Skip if pose was not found
            if results.pose_landmarks is None:
                continue
            
            # Extract landmarks
            pose_component = results.pose_landmarks
            pose_data = component_points(pose_component)
            face_data = component_points(results.face_landmarks, pose_component.landmark[mp_holistic.PoseLandmark.NOSE], len(FACE_IDS))
            lh_data = component_points(results.left_hand_landmarks, pose_component.landmark[mp_holistic.PoseLandmark.LEFT_WRIST], len(HAND_IDS))
            rh_data = component_points(results.right_hand_landmarks, pose_component.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST], len(HAND_IDS))
            
            # Concatenate into [joints, 3] shape
            data = np.concatenate([pose_data, face_data, lh_data, rh_data])
            
            # Append depth into [joints, 4] shape
            if depth is not None:
                kinect_depth = []
                scaled_data = data[:, :2] * [w, h]
                for x, y in np.array(scaled_data, dtype="int32"):
                    if 0 < x < w and 0 < y < h:
                        kinect_depth.append(depth[i, y, x, 0] / 255)
                    else:
                        kinect_depth.append(0)
                kinect_vec = np.expand_dims(np.array(kinect_depth), axis=-1)
                data = np.concatenate([data, kinect_vec], axis=-1)
            
            # Append to list
            datas.append(data)
    
    # Stack into [frames, joints, 3] shape
    stacked = np.stack(datas)
    
    return stacked

def parallel_process_handler(args):
    # extract arguments
    color_video_path, depth_video_path, destination = args
    
    # call main function
    output = process_holistic(color_video_path, depth_video_path)
    
    # save in file
    output_filename = str(destination / color_video_path.name)
    output_filename = output_filename.replace("_color", "")
    output_filename = output_filename.replace(".mp4", ".npy")
    np.save(output_filename, output)
    
    # return output filename
    return output_filename
    

def main(args):
    source = args.source
    destination = args.destination
    
    if not destination.exists():
        destination.mkdir()
    
    depth_videos = list(source.glob('*_depth.mp4'))
    depth_videos.sort()
    
    color_videos = list(source.glob('*_color.mp4'))
    color_videos.sort()
    
    handler_args = zip(color_videos, depth_videos, itertools.repeat(destination))
    presults = pqdm(handler_args, parallel_process_handler, n_jobs=args.n_jobs)
    
    print(len(presults))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract skeleton landmarks of videos.')
    parser.add_argument('--source', type=Path, help='Path to source folder')
    parser.add_argument('--destination', type=Path, help='Path to destination folder')
    parser.add_argument('--n_jobs', type=int, help='Parallel jobs')
    args = parser.parse_args()
    main(args)
    
