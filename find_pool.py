import cv2
import os
import multiprocessing

# 视频文件路径
video_path = '/home/haoge/ymq1/20160501.mp4'
# 设置保存每一帧的文件夹
output_folder = 'pool_frames_output'

# 创建保存文件夹，如果不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def save_frame(frame, frame_number, output_folder):
    frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

def process_video(video_path, output_folder):
    # 创建文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取逻辑核心数
    logical_cores = os.cpu_count()

    # 设置进程池大小
    process_count = logical_cores if logical_cores <= 16 else 16  # 限制最大16个进程

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    pool = multiprocessing.Pool(process_count)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 异步保存帧
        pool.apply_async(save_frame, args=(frame, frame_number, output_folder))
        frame_number += 1

    pool.close()
    pool.join()
    cap.release()

if __name__ == "__main__":
    process_video(video_path,output_folder)
