from nuscenes import NuScenes

def main():
    #Load nuscenes data
    nusc = NuScenes(version='v1.0-mini', dataroot='/media/storage/datasets/nuscenes-v1.0-mini', verbose=False)
    colourmap = {}

    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        colourmap[index] = colour

    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])
    try:
        while(True):
            if(sample["next"] == str()):
                break

            sample = nusc.get("sample", sample["next"])
        
            lidar_top_token = sample['data']['LIDAR_TOP']
            nusc.render_sample_data(
                lidar_top_token,
                with_anns=True,
                show_lidarseg=True,
                use_flat_vehicle_coordinates=True,
                underlay_map = False
            )
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
