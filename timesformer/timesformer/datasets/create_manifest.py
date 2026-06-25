import pandas as pd
import os

video_metadata_file = '/datasets/kinetics/kinetics400/validate.csv'

label_map_file = 'kinetics_400_labels.csv'

output_manifest_file = 'validate_manifest.txt'

split_dir = 'val'

def generate_manifest():
    label_df = pd.read_csv(label_map_file)
    class_to_id = pd.Series(label_df.id.values, index=label_df.name).to_dict()
    print(f"-> Found {len(class_to_id)} class labels.")


    video_df = pd.read_csv(video_metadata_file)

    print(f"-> Found metadata for {len(video_df)} video clips.")


    manifest_lines = []
    for _, row in video_df.iterrows():
        class_name = row['label']
        youtube_id = row['youtube_id']
        start_time = row['time_start']
        end_time = row['time_end']

        label_id = class_to_id.get(class_name)
        if label_id is None:
            continue

        video_filename = f"{youtube_id}_{int(start_time):06d}_{int(end_time):06d}.mp4"

        relative_path = os.path.join(split_dir, video_filename)

        manifest_lines.append(f"{relative_path} {label_id}")

    with open(output_manifest_file, 'w') as f:
        f.write("\n".join(manifest_lines))



if __name__ == '__main__':
    generate_manifest()