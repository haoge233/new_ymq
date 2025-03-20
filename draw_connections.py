from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 图片宽高（以像素为单位）
width = 1101
height = 2848

# 关节点坐标
keypoints = [
        [
            1017.9193725585938,
            581.5896606445312
        ],
        [
            984.816650390625,
            580.1497192382812
        ],
        [
            1012.1624145507812,
            575.8297729492188
        ],
        [
            986.255859375,
            587.349609375
        ],
        [
            1015.0408325195312,
            587.349609375
        ],
        [
            984.816650390625,
            617.589111328125
        ],
        [
            1009.2838745117188,
            629.1090087890625
        ],
        [
            954.5924072265625,
            611.8292236328125
        ],
        [
            1023.6763916015625,
            653.588623046875
        ],
        [
            966.1063842773438,
            575.8297729492188
        ],
        [
            966.1063842773438,
            578.709716796875
        ],
        [
            938.7606811523438,
            702.5479125976562
        ],
        [
            958.91015625,
            716.9476928710938
        ],
        [
            917.171875,
            750.0671997070312
        ],
        [
            1000.6483764648438,
            793.2666015625
        ],
        [
            871.115966796875,
            804.786376953125
        ],
        [
            957.4708862304688,
            860.9456176757812
        ]
    ]

# 骨架连接关系
connections = [
    (11, 5),
    (12, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (4, 11),
    (4, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16)
]

# # 读取图片
# image_path = "/home/haoge/ymq1/score_output_folder_2016-1/10338_annotated.jpg"  # 替换为你要读取的图片路径
# image = Image.open(image_path)

# # 创建 ImageDraw 对象来绘制图像
# draw = ImageDraw.Draw(image)

# # 绘制骨架
# for (start, end) in connections:
#     start_point = keypoints[start ]  # 调整为0-indexed
#     end_point = keypoints[end ]  # 调整为0-indexed
#     draw.line([start_point[0], start_point[1], end_point[0], end_point[1]], fill='red', width=2)

# # 绘制关节点
# for point in keypoints:
#     draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill='green')

# # 保存新图像
# output_file = 'skeleton_pose_on_image3.png'  # 替换为你希望保存的文件路径
# image.save(output_file)

# # 显示保存后的图像
# image.show()

def draw_diff(keypoints,keypoints1):
    image_path = "white1.png"  # 替换为你要读取的图片路径
    image = Image.open(image_path)

    # 创建 ImageDraw 对象来绘制图像
    draw = ImageDraw.Draw(image)

    # 绘制骨架
    for (start, end) in connections:
        start_point = keypoints[start ]  # 调整为0-indexed
        end_point = keypoints[end ]  # 调整为0-indexed
        draw.line([start_point[0], start_point[1], end_point[0], end_point[1]], fill='red', width=2)

    # 绘制关节点
    for point in keypoints:
        draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill='blue')

    # 绘制骨架
    for (start, end) in connections:
        start_point = keypoints1[start ]  # 调整为0-indexed
        end_point = keypoints1[end ]  # 调整为0-indexed
        draw.line([start_point[0], start_point[1], end_point[0], end_point[1]], fill='red', width=2)

    # 绘制关节点
    for point in keypoints1:
        draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill='green')

        # 保存新图像
    output_file = 'skeleton_pose_on_image2.png'  # 替换为你希望保存的文件路径
    image.save(output_file)

