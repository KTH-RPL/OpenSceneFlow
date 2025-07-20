"""
# Created: 2024-02-26 12:42
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), Tianshuai Hu (thuaj@connect.ust.hk)
# 
# This file is part of SeFlow (https://github.com/KTH-RPL/SeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Preprocess Data, save as h5df format for faster loading
# This one is for Waymo dataset, refer a lot to 
# Kyle's ZeroFlow code: https://github.com/kylevedder/zeroflow/tree/master/data_prep_scripts/waymo

# env need: pip install waymo-open-dataset-tf-2.11.0==1.5.0
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# NOTE(2023/02/29): it's really important to set this! otherwise, the point cloud will be wrong. really wried.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import multiprocessing
from multiprocessing import Pool, current_process
from pathlib import Path
from tqdm import tqdm
import numpy as np
import fire, time, h5py

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils

import tensorflow as tf

import os, sys, json
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index, SE2
GROUND_HEIGHT_THRESHOLD = 0.4  # 40 centimeters
RANGE_MAX_VALID = 50

def is_ground_points(
    raster_heightmap,
    global_to_raster_se2,
    global_to_raster_scale,
    global_point_cloud,
) -> np.ndarray:
    """Remove ground points from a point cloud.
    Args:
        point_cloud: Numpy array of shape (k,3) in global coordinates.
    Returns:
        ground_removed_point_cloud: Numpy array of shape (k,3) in global coordinates.
    """
    def get_ground_heights(
        raster_heightmap,
        global_to_raster_se2,
        global_to_raster_scale,
        global_point_cloud,
    ) -> np.ndarray:
        """Get ground height for each of the xy locations in a point cloud.
        Args:
            point_cloud: Numpy array of shape (k,2) or (k,3) in global coordinates.
        Returns:
            ground_height_values: Numpy array of shape (k,)
        """

        global_points_xy = global_point_cloud[:, :2]

        raster_points_xy = (
            global_to_raster_se2.transform_point_cloud(global_points_xy) * global_to_raster_scale
        )

        raster_points_xy = np.round(raster_points_xy).astype(np.int64)

        ground_height_values = np.full((raster_points_xy.shape[0]), np.nan)
        # outside max X
        outside_max_x = (raster_points_xy[:, 0] >= raster_heightmap.shape[1]).astype(bool)
        # outside max Y
        outside_max_y = (raster_points_xy[:, 1] >= raster_heightmap.shape[0]).astype(bool)
        # outside min X
        outside_min_x = (raster_points_xy[:, 0] < 0).astype(bool)
        # outside min Y
        outside_min_y = (raster_points_xy[:, 1] < 0).astype(bool)
        ind_valid_pts = ~np.logical_or(
            np.logical_or(outside_max_x, outside_max_y),
            np.logical_or(outside_min_x, outside_min_y),
        )

        ground_height_values[ind_valid_pts] = raster_heightmap[
            raster_points_xy[ind_valid_pts, 1], raster_points_xy[ind_valid_pts, 0]
        ]

        return ground_height_values
    ground_height_values = get_ground_heights(
        raster_heightmap,
        global_to_raster_se2,
        global_to_raster_scale,
        global_point_cloud,
    )
    is_ground_boolean_arr = (
        np.absolute(global_point_cloud[:, 2] - ground_height_values) <= GROUND_HEIGHT_THRESHOLD
    ) | (np.array(global_point_cloud[:, 2] - ground_height_values) < 0)
    return is_ground_boolean_arr

def convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    point_flows,
    range_image_top_pose,
    ri_index=0,
    keep_polar_features=False,
):
    """Convert range images to point cloud.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
      points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        (NOTE: Will be {[N, 6]} if keep_polar_features is true.
      cp_points: {[N, 6]} list of camera projections of length 5
        (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    flows = []

    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features
    )

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))

        flow = point_flows[c.name][ri_index]
        flow_tensor = tf.reshape(tf.convert_to_tensor(value=flow.data), flow.shape.dims)
        flow_points_tensor = tf.gather_nd(flow_tensor, tf.compat.v1.where(range_image_mask))

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.compat.v1.where(range_image_mask))

        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        flows.append(flow_points_tensor.numpy())

    return points, cp_points, flows

def parse_range_image_and_camera_projection(frame):
    """
      Parse range images and camera projections given a frame.

    Args:
       frame: open dataset frame proto

    Returns:
       range_images: A dict of {laser_name,
         [range_image_first_return, range_image_second_return]}.
       camera_projections: A dict of {laser_name,
         [camera_projection_from_first_return,
          camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
    """
    range_images = {}
    camera_projections = {}
    point_flows = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if (
            len(laser.ri_return1.range_image_compressed) > 0
        ):  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if len(laser.ri_return1.range_image_flow_compressed) > 0:
                range_image_flow_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_flow_compressed, "ZLIB"
                )
                ri = dataset_pb2.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_flow_str_tensor.numpy()))
                point_flows[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, "ZLIB"
                )
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy())
                )

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
        if (
            len(laser.ri_return2.range_image_compressed) > 0
        ):  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            if len(laser.ri_return2.range_image_flow_compressed) > 0:
                range_image_flow_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.range_image_flow_compressed, "ZLIB"
                )
                ri = dataset_pb2.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_flow_str_tensor.numpy()))
                point_flows[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, point_flows, range_image_top_pose

def get_car_pc_global_pc_flow_transform(frame: dataset_pb2.Frame):

    # Parse the frame lidar data into range images.
    range_images, camera_projections, point_flows, range_image_top_poses = parse_range_image_and_camera_projection(frame)

    # Project the range images into points.
    points_lst, cp_points, flows_lst = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        point_flows,
        range_image_top_poses,
        keep_polar_features=True)

    car_frame_pc = points_lst[0][:, 3:]
    car_frame_flows = flows_lst[0][:, :3]
    car_frame_labels = flows_lst[0][:, 3]
    num_points = car_frame_pc.shape[0]

    # # Transform the points from the vehicle frame to the world frame.
    world_frame_pc = np.concatenate([car_frame_pc, np.ones([num_points, 1])], axis=-1)
    car_to_global_transform = np.reshape(np.array(frame.pose.transform), [4, 4])
    world_frame_pc = np.transpose(np.matmul(car_to_global_transform, np.transpose(world_frame_pc)))[:, :3]

    # # Transform the points from the world frame to the map frame.
    offset = frame.map_pose_offset
    points_offset = np.array([offset.x, offset.y, offset.z])
    world_frame_pc += points_offset
    
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    for c in calibrations:
        if calibrations[0].name == dataset_pb2.LaserName.TOP:
            break
    ego2sensor = np.reshape(np.array(c.extrinsic.transform), [4, 4])
    car_frame_pc[:, 2] = car_frame_pc[:, 2] - ego2sensor[2, 3] # move center to sensor height
    return (
        car_frame_pc,
        world_frame_pc,
        car_frame_flows,
        car_frame_labels,
        car_to_global_transform,
    )

def load_ground_height_raster(map_path: Path):

    raster_heightmap = np.load(map_path / "ground_height.npy")
    transform = json.load(open(map_path / "se2.json"))

    transform_rotation = np.array(transform["R"]).reshape(2, 2)
    transform_translation = np.array(transform["t"])
    transform_scale = np.array(transform["s"])

    transform_se2 = SE2(rotation=transform_rotation, translation=transform_translation)

    return raster_heightmap, transform_se2, transform_scale

def process_log(data_dir: Path, log, log_map_folder, output_dir: Path, n = None) :

    def create_group_data(group, pc, pose, gm = None, flow_0to1=None, flow_valid=None, flow_category=None, ego_motion=None):
        group.create_dataset('lidar', data=pc.astype(np.float32))
        group.create_dataset('pose', data=pose.astype(np.float64))
        if ego_motion is not None:
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))
        if gm is not None:
            group.create_dataset('ground_mask', data=gm.astype(bool))
        if flow_0to1 is not None:
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            flow_valid = np.ones_like(gm)
            flow_valid[gm] = 0
            INSIDE_RANGE = np.logical_and(pc[:, 0] < RANGE_MAX_VALID, pc[:, 0] > -RANGE_MAX_VALID) & np.logical_and(pc[:, 1] < RANGE_MAX_VALID, pc[:, 1] > -RANGE_MAX_VALID)
            flow_valid[~INSIDE_RANGE] = 0
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            # From the Waymo Open dataset.proto:
            # // If the point is not annotated with scene flow information, class is set
            # // to -1. A point is not annotated if it is in a no-label zone or if its label
            # // bounding box does not have a corresponding match in the previous frame,
            # // making it infeasible to estimate the motion of the point.
            # // Otherwise, (vx, vy, vz) are velocity along (x, y, z)-axis for this point
            # // and class is set to one of the following values:
            # //  -1: no-flow-label, the point has no flow information.
            # //   0:  unlabeled or "background,", i.e., the point is not contained in a
            # //       bounding box.
            # //   1: vehicle, i.e., the point corresponds to a vehicle label box.
            # //   2: pedestrian, i.e., the point corresponds to a pedestrian label box.
            # //   3: sign, i.e., the point corresponds to a sign label box.
            # //   4: cyclist, i.e., the point corresponds to a cyclist label box.
            # replace all -1 and 0 to 0
            flow_category[flow_category < 0] = 0 # NONE, Background
            # replace 1 to 19, since av2 index 19 is for REGULAR_VEHICLE
            flow_category[flow_category == 1] = 19
            # replace 2 to 17, since av2 index 16 is for PEDESTRIAN
            flow_category[flow_category == 2] = 17
            # replace 3 to 21, since av2 index 21 is for SIGN
            flow_category[flow_category == 3] = 21
            # no replace 4 to 4, since av2 index 4 is for BICYCLIST
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.int8))
    
    raster_heightmap, transform_se2, transform_scale = load_ground_height_raster(log_map_folder.parent / log_map_folder.stem)
    all_data = list(tf.data.TFRecordDataset(data_dir / log, compression_type='').as_numpy_iterator())
    first_frame = dataset_pb2.Frame.FromString(bytearray(all_data[0]))
    scene_id = first_frame.context.name
    total_lens = len(all_data)
    for data_idx in range(1, total_lens):
        if data_idx >= total_lens - 2:
            # 0: no correct flow label, end(total_lens - 1) - 1: no correct pose flow
            continue
        frame = dataset_pb2.Frame.FromString(bytearray(all_data[data_idx]))
        if scene_id != frame.context.name:
            print(f"Scene ID mismatch: {scene_id} vs {frame.context.name}")
            break
        car_frame_pc, global_frame_pc, flow, label, pose = get_car_pc_global_pc_flow_transform(frame)
        _, _, _, _, pose1 = get_car_pc_global_pc_flow_transform(dataset_pb2.Frame.FromString(bytearray(all_data[data_idx+1])))
        ego_motion = np.linalg.inv(pose1) @ pose
        pose_flow = car_frame_pc[:, :3] @ ego_motion[:3, :3].T + ego_motion[:3, 3] - car_frame_pc[:, :3]
        ground_mask = is_ground_points(raster_heightmap, transform_se2, transform_scale, global_frame_pc)
        timestamp = frame.timestamp_micros
        if car_frame_pc.shape[0] < 256:
            print(f'{scene_id}/{timestamp} has less than 256 points, skip this scenarios. Please check the data if needed.')
            break
        with h5py.File(output_dir/f'{scene_id}.h5', 'a') as f:
            group = f.create_group(str(timestamp))
            create_group_data(group, car_frame_pc, pose, ego_motion=ego_motion, gm=np.array(ground_mask), flow_0to1=(flow/10.0+pose_flow), flow_category=label)
        # if data_idx > 10:
        #     break

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)

def process_logs(data_dir: Path, map_dir: Path, output_dir: Path, nproc: int):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
       Args:
         data_dir: Argoverse 2.0 directory
         output_dir: Output directory.
    """
    if not (data_dir).exists():
        print(f'{data_dir} not found')
        return
    
    logs = sorted(os.listdir(data_dir))
    args = sorted([(data_dir, log, map_dir/log, output_dir) for log in logs])
    print(f'Using {nproc} processes to process {len(args)} logs.')
    
    # # for debug
    # for x in tqdm(args):
    #     proc(x, ignore_current_process=True)
    #     break

    if nproc <= 1:
        for x in tqdm(args):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(args), ncols=100))

def main(
    flow_data_dir: str = "/home/kin/data/waymo/flowlabel",
    mode: str = "valid",
    map_dir: str = "/home/kin/data/waymo/flowlabel/map",
    output_dir: str ="/home/kin/data/waymo/flowlabel/preprocess",
    nproc: int = (multiprocessing.cpu_count() - 1),
    create_index_only: bool = False,
):
    output_dir_ = Path(output_dir) / mode
    if create_index_only:
        create_reading_index(Path(output_dir_))
        return
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(Path(flow_data_dir) / mode, Path(map_dir), output_dir_, nproc)
    create_reading_index(output_dir_)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")