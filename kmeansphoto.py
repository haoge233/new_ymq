import cv2

def extract_frames(video_path, frame_numbers):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    
    # 获取视频的总帧数
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    
    frames = []
    for frame_num in frame_numbers:
        if frame_num >= total_frames:
            print(f"Frame number {frame_num} exceeds total frames in video.")
            continue
        
        # 设置视频读取位置
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # 读取帧
        ret, frame = video_capture.read()
        
        if ret:
            frames.append(frame)
        else:
            print(f"Failed to read frame {frame_num}")
    
    video_capture.release()
    return frames

# 示例：从视频中提取帧号为 [10, 50, 100, 150] 的帧
video_path = "/home/haoge/solopose/SoloShuttlePose/videos 20160430/20160430/output.mp4"
frame_numbers = [38966, 116634, 115691, 102919, 13170]
frames = extract_frames(video_path, frame_numbers)

# 你可以进一步处理这些帧，比如显示或保存它们
for idx, frame in enumerate(frames):
    # cv2.imshow(f"Frame {frame_numbers[idx]}", frame)
    print(idx)
    cv2.imwrite(f"/home/haoge/ymq1/kmeanspho/frame_{frame_numbers[idx]}.jpg", frame)  # 保存帧为图片

cv2.waitKey(0)
cv2.destroyAllWindows()
