import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 1. 计算 SSIM
def compute_frame_ssim(frame1, frame2):
    # 将图像转换为灰度图像，SSIM 计算是基于灰度图的
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 计算两个图像的SSIM
    ssim_index, _ = ssim(gray1, gray2, full=True)
    return ssim_index

def compute_video_ssim(test_video_path, template_frame, frame_rate, frame_width, frame_height):
    # 读取待计算视频
    cap_test = cv2.VideoCapture(test_video_path)  # 待计算视频

    # 获取待计算视频的帧数
    frame_count_test = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建模板视频流：使用模板帧替代模板视频
    template_frame_resized = cv2.resize(template_frame, (frame_width, frame_height))

    # 计算每一帧与模板帧的 SSIM
    ssim_values = []
    for _ in range(frame_count_test):
        ret_test, frame_test = cap_test.read()

        if not ret_test:
            break
        
        # 计算每一帧的 SSIM
        ssim_value = compute_frame_ssim(frame_test, template_frame_resized)
        ssim_values.append(ssim_value)

    # 计算整个视频的平均 SSIM
    average_ssim = np.mean(ssim_values)
    cap_test.release()

    return average_ssim

# 2. 遍历文件夹处理所有视频
def process_videos_in_folder(input_folder, template_frame_path, output_folder):
    # 获取文件夹中的所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

    # 读取模板帧
    template_frame = cv2.imread(template_frame_path)

    for video_file in video_files:
        # 获取视频的完整路径
        video_path = os.path.join(input_folder, video_file)

        # 读取待计算视频，获取帧率、帧宽和帧高
        cap_test = cv2.VideoCapture(video_path)

        # 获取待计算视频的参数
        frame_rate = int(cap_test.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frame_count_test = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT()))

        cap_test.release()

        # 计算视频的相似度
        video_ssim = compute_video_ssim(video_path, template_frame, frame_rate, frame_width, frame_height)

        # 输出结果
        print(f"Video: {video_file}, SSIM: {video_ssim}")

# 3. 主程序
if __name__ == "__main__":
    input_folder = '/home/haoge/solopose/SoloShuttlePose/res 2016-3/videos/20160501-3'  # 存放待计算视频的文件夹
    template_frame_path = '/home/haoge/ymq1/tiqupho_2016_3/keyframe_5663.jpg'  # 模板帧图像路径
    output_folder = 'path/to/output'  # 生成模板视频的输出文件夹

    # # 确保输出文件夹存在
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # 处理文件夹中的所有视频
    process_videos_in_folder(input_folder, template_frame_path, output_folder)
