import subprocess
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 使用 ffmpeg 提取帧
    command = [
        'ffmpeg',
        '-i', video_path,          # 输入视频文件路径
        os.path.join(output_folder, '%05d.jpg')  # 输出帧的文件路径格式
    ]

    subprocess.run(command, check=True)
    print(f'Frames have been extracted to {output_folder}')

# 遍历指定目录下的所有 .avi 文件并提取帧
parent_directory = r'E:\wlh\MPN-main\data\shanghaichuli\video\frames'  # 父目录路径

for filename in os.listdir(parent_directory):
    if filename.endswith('.avi'):
        video_path = os.path.join(parent_directory, filename)
        # 为每个视频创建一个对应的输出文件夹
        output_folder = os.path.join(parent_directory, os.path.splitext(filename)[0])
        extract_frames(video_path, output_folder)
