import os
import shutil

# 源目录和目标目录
src_dir = '/home/haoge/ymq1'  # 源目录路径
dst_dir = '/home/haoge/new_ymq'  # 目标目录路径

# 确保目标目录存在
os.makedirs(dst_dir, exist_ok=True)

# 遍历源目录下的文件
for file in os.listdir(src_dir):
    if file.endswith('.py'):  # 只复制 .py 文件
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        shutil.copy2(src_file, dst_file)  # 保留文件的元信息（如修改时间）
        print(f'已复制: {src_file} -> {dst_file}')

print('复制完成！')
