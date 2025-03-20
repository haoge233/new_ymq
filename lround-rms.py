import os
import pandas as pd
import librosa
from moviepy.editor import VideoFileClip

# 视频文件夹路径和xlsx文件路径
video_folder = "/home/haoge/ymq1/2016-3-lround_30_D40"
xlsx_file = "/home/haoge/ymq1/2016-3-xlsx/2016-3-round-score_D40.xlsx"
output_xlsx_file = "lround-Rms.xlsx"

# 读取xlsx文件
df = pd.read_excel(xlsx_file)

# 定义一个函数来提取视频的音频并计算RMS值
def calculate_rms_from_video(video_path):
    # 提取视频的音频
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_file = "temp_audio.wav"
    audio.write_audiofile(audio_file)

    # 加载音频文件
    audio_data, sr = librosa.load(audio_file, sr=None)

    # 计算RMS值（音频能量）
    rms = librosa.feature.rms(y=audio_data)
    rms_value = rms.mean()  # 计算RMS的平均值，作为代表整体能量的指标

    # 删除临时音频文件
    os.remove(audio_file)

    return rms_value

# 为每个视频文件计算RMS并添加到DataFrame
rms_values = []
for index, row in df.iterrows():
    # 获取视频文件路径
    video_name = row['视频名称']
    video_path = os.path.join(video_folder, video_name)

    if os.path.exists(video_path):
        # 计算RMS值
        rms_value = calculate_rms_from_video(video_path)
        rms_values.append(rms_value)
    else:
        # 如果视频文件不存在，则填充NaN
        rms_values.append(None)

# 将RMS值添加到DataFrame
df['音频RMS'] = rms_values

# 保存为新的xlsx文件
df.to_excel(output_xlsx_file, index=False)

print(f"已将RMS值写入新的文件: {output_xlsx_file}")
