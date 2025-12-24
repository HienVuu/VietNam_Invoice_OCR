import os
import json
import subprocess
import argparse
import shutil

def create_ground_truth(image_dir, json_dir, temp_output='temp_output'):
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            base_name = os.path.splitext(filename)[0]

            # Run pipeline to generate JSON
            cmd = f"python run.py --image_paths \"{img_path}\" --output_path \"{temp_output}\""
            subprocess.run(cmd, shell=True)

            # Find generated JSON
            json_file = os.path.join(temp_output, base_name + '.json')
            if os.path.exists(json_file):
                # Copy to ground truth dir (you can edit manually later)
                shutil.copy(json_file, os.path.join(json_dir, base_name + '.json'))
                print(f"Generated ground truth for {filename}")
            else:
                print(f"Failed to generate for {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, help='Directory with images')
    parser.add_argument('--json_dir', required=True, help='Directory to save ground truth JSONs')
    args = parser.parse_args()

    create_ground_truth(args.image_dir, args.json_dir)
    print("Ground truth generation done. Manually edit JSONs for accuracy.")
