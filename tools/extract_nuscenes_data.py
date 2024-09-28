import os
import sys
import shelve
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.dirname(__file__))) #Add parent folder to path

from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from src.utils import TimestampData

parser = ArgumentParser()
parser.add_argument("--dataset_root", default="/media/storage/datasets/nuscenes")
parser.add_argument("--dataset_version", default="v1.0-trainval")

def box_to_dict(box):
    return dict(
        center = list(box.center),
        size = list(box.wlh),
        rotation_matrix = box.rotation_matrix,
        label = box.label,
        score = box.score,
        name = box.name,
        token = box.token,
    )

def main(args):
    nusc = NuScenes(version=args.dataset_version, dataroot=args.dataset_root, verbose=False)
    nusc_can = NuScenesCanBus(dataroot=args.dataset_root)

    class_to_colour = {}
    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        class_to_colour[index] = colour

    nuscenes_simplified = shelve.open("nuscenes_simplified.dataset", "n")
    nuscenes_simplified["class_to_colour"] = class_to_colour

    """
        To extract
            - pcd_path: List
            - pcd_labels_path: List
            - ego_pose: List
            - cam_front: List
            - cam_front_extrinsict: Single dict
            - lidar_extriniscs: Single dict
            - Box detections: List boxes
            - image_path: str
    """

    for i in tqdm(range(len(nusc.scene))):
        scene = nusc.scene[i]
        sample = None

        try:
            pose = nusc_can.get_messages(scene["name"], 'pose')
            pose_dataset = TimestampData(pose, [p["utime"] for p in pose])
        except Exception:
            pose_dataset = None

        scene_data = []

        for _ in range(scene["nbr_samples"]):
            data = {}
            sample_token = scene["first_sample_token"] if sample is None else sample["next"]
            sample = nusc.get("sample", sample_token)

            lidar_token = sample['data']['LIDAR_TOP']
            lidar = nusc.get('sample_data', lidar_token)
            lidar_extrinsics = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            cam_front = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            ego_pose = nusc.get("ego_pose", lidar_token)

            data["lidar_extrinsics"] = lidar_extrinsics
            data["lidar_origin"] = lidar_extrinsics['translation']
            data["lidar_pcd_path"] = nusc.get_sample_data_path(lidar_token)
            data["lidar_pcd_labels_path"] = os.path.join(nusc.dataroot, f"lidarseg/{nusc.version}", lidar_token + '_lidarseg.bin')
            data["cam_front_extrinsics"] = nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token']);
            data["image_path"] = os.path.join(nusc.dataroot, cam_front["filename"])
            data["car_world_position"] = ego_pose["translation"]
            data["car_rotation"] = ego_pose["rotation"]
            data["can_pose_at_timestamp"] = pose_dataset.get_data_at_timestamp(lidar["timestamp"]) if pose_dataset is not None else None
            data["box_detections"] = [box_to_dict(box) for box in nusc.get_boxes(lidar_token)]
            scene_data.append(data)

        nuscenes_simplified[str(i)] = scene_data

    nuscenes_simplified.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
