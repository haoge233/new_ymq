# import librosa
# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt


# save_path="RMS_2016-1.png"


# save_path1="Spectrogram of Audio.png"
# # 载入音频文件
# audio_path = '/home/haoge/ymq1/20160501.mp3'  # 提取后的音频文件路径
# # 加载音频文件，保留原采样率
# y, sr = librosa.load(audio_path, sr=None)

# # 定义时间区间（单位：秒）
# start_time = 0  # 从6.47秒开始
# end_time = 33.25 * 60  # 到10分钟处（600秒）

# # 计算音频片段的样本范围
# start_sample = int(start_time * sr)
# end_sample = int(end_time * sr)

# # 裁剪音频片段
# y_segment = y[start_sample:end_sample]

# # 计算RMS能量
# rms = librosa.feature.rms(y=y_segment)

# # 获取RMS特征的帧索引
# rms_frames = np.arange(rms.shape[-1])


# start_frame = int(start_sample / 512)  # 512是hop_length的默认值

# # 将帧索引转换为时间（单位：秒）
# rms_times = librosa.frames_to_time(rms_frames, sr=sr, hop_length=512)  # hop_length是步长

# # 将横坐标调整为真实时间，从start_time开始
# rms_times += start_time

# # 画图：显示RMS能量
# plt.figure(figsize=(10, 6))
# plt.plot(rms_times, rms[0], label='RMS Energy')  # 横坐标为真实时间
# # plt.plot(rms_frames + start_frame, rms[0], label='RMS Energy')  # 横坐标为帧索引，从start_frame开始
# plt.xlabel('Time (seconds)')
# plt.ylabel('RMS Energy')
# plt.title(f'RMS Energy from {start_time}s to {end_time}s')
# plt.grid(True)
# plt.legend()
# # plt.show()
# plt.savefig(save_path)



# import librosa
# import librosa.display
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # 设置为无图形界面的后端
# import matplotlib.pyplot as plt

# save_path = "RMS_2016-1-top15.png"
# save_path1 = "Spectrogram of Audio.png"

# # 载入音频文件
# audio_path = '/home/haoge/ymq1/20160501.mp3'  # 提取后的音频文件路径
# # 加载音频文件，保留原采样率
# y, sr = librosa.load(audio_path, sr=None)
# print(233)
# # 定义时间区间（单位：秒）
# start_time = 0  # 从6.47秒开始
# end_time = 33.25 * 60  # 到10分钟处（600秒）

# # 计算音频片段的样本范围
# start_sample = int(start_time * sr)
# end_sample = int(end_time * sr)

# # 裁剪音频片段
# y_segment = y[start_sample:end_sample]

# # 计算RMS能量
# rms = librosa.feature.rms(y=y_segment)

# # 获取RMS特征的帧索引
# rms_frames = np.arange(rms.shape[-1])

# # 将帧索引转换为时间（单位：秒）
# rms_times = librosa.frames_to_time(rms_frames, sr=sr, hop_length=512)  # hop_length是步长

# # 将横坐标调整为真实时间，从start_time开始
# rms_times += start_time

# # 获取RMS值及其对应的时间
# rms_values = rms[0]

# # 排序：获取从大到小的前15个RMS值及其对应时间
# top_indices = np.argsort(rms_values)[::-1][:500]
# top_rms_values = rms_values[top_indices]
# top_rms_times = rms_times[top_indices]

# # 打印前15个RMS值及其时间
# print("Top 15 RMS values and corresponding times:")
# for i in range(500):
#     print(f"Time: {top_rms_times[i]:.2f}s, RMS: {top_rms_values[i]:.6f}")

# # 画图：显示RMS能量
# plt.figure(figsize=(10, 6))
# plt.plot(rms_times, rms_values, label='RMS Energy')  # 横坐标为真实时间
# plt.scatter(top_rms_times, top_rms_values, color='red', label='Top 15 RMS', zorder=5)
# plt.xlabel('Time (seconds)')
# plt.ylabel('RMS Energy')
# plt.title(f'RMS Energy from {start_time}s to {end_time}s')
# plt.grid(True)
# plt.legend()
# # plt.show()
# plt.savefig(save_path)


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

save_path = "RMS_2016-1-topkey15.png"
save_path1 = "Spectrogram of Audio.png"

# 载入音频文件
audio_path = '/home/haoge/ymq1/20160501.mp3'  # 提取后的音频文件路径
# 加载音频文件，保留原采样率
print(233)
y, sr = librosa.load(audio_path, sr=None)
print(333)
# 定义时间区间（单位：秒）
start_time = 0  # 从6.47秒开始
end_time = 33.25 * 60  # 到10分钟处（600秒）

# 计算音频片段的样本范围
start_sample = int(start_time * sr)
end_sample = int(end_time * sr)

# 裁剪音频片段
y_segment = y[start_sample:end_sample]

# 计算RMS能量
rms = librosa.feature.rms(y=y_segment)

# 获取RMS特征的帧索引
rms_frames = np.arange(rms.shape[-1])

# 将帧索引转换为时间（单位：秒）
rms_times = librosa.frames_to_time(rms_frames, sr=sr, hop_length=512)  # hop_length是步长

# 将横坐标调整为真实时间，从start_time开始
rms_times += start_time

# 获取RMS值及其对应的时间
rms_values = rms[0]

# 设置时间窗口大小（单位：秒），例如10秒内视为一个点
window_size = 5  # 10秒的窗口

# 计算每个窗口的最大RMS值及其对应的时间
window_start_times = []
window_max_rms = []
window_max_time = []

# 遍历RMS时间和值，按时间窗口分组
i = 0
last_window_max_time = None
last_window_max_rms = None

while i < len(rms_times):
    # 窗口内的时间范围
    window_start = rms_times[i]
    window_end = window_start + window_size

    # 找出窗口内的所有RMS值及其时间
    window_indices = np.where((rms_times >= window_start) & (rms_times < window_end))[0]

    if len(window_indices) > 0:
        # 找到窗口内最大RMS值及对应的时间
        max_rms_idx = window_indices[np.argmax(rms_values[window_indices])]
        current_max_time = rms_times[max_rms_idx]
        current_max_rms = rms_values[max_rms_idx]

        # 如果当前窗口的最大值与上一个窗口的最大值的时间差小于窗口大小，则保留较大的RMS值
        if last_window_max_time is not None and abs(current_max_time - last_window_max_time) < window_size:
            if current_max_rms > last_window_max_rms:
                # 保留当前最大值，删除上一个
                window_max_time[-1] = current_max_time
                window_max_rms[-1] = current_max_rms
            # 如果当前最大值小于上一个，则跳过当前窗口
        else:
            # 如果时间差大于窗口大小，则保留当前窗口的最大值
            window_start_times.append(window_start)
            window_max_rms.append(current_max_rms)
            window_max_time.append(current_max_time)

        # 更新上一个窗口的最大值
        last_window_max_time = current_max_time
        last_window_max_rms = current_max_rms

    # 跳到下一个窗口
    i = window_indices[-1] + 1 if len(window_indices) > 0 else i + 1

# 将结果转换为numpy数组方便排序
window_max_rms = np.array(window_max_rms)
window_max_time = np.array(window_max_time)

# 排序：获取从大到小的前15个RMS值及其对应时间
top_indices = np.argsort(window_max_rms)[::-1][:15]
top_rms_values = window_max_rms[top_indices]
top_rms_times = window_max_time[top_indices]

# 打印前15个RMS值及其时间
print("Top 15 RMS values and corresponding times:")
for i in range(len(top_rms_times)):
    print(f"Time: {top_rms_times[i]:.2f}s, RMS: {top_rms_values[i]:.6f}")

# 画图：显示RMS能量
plt.figure(figsize=(10, 6))
plt.plot(rms_times, rms_values, label='RMS Energy')  # 横坐标为真实时间
plt.scatter(top_rms_times, top_rms_values, color='red', label='Top 15 RMS', zorder=5)
plt.xlabel('Time (seconds)')
plt.ylabel('RMS Energy')
plt.title(f'RMS Energy from {start_time}s to {end_time}s')
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig(save_path)
