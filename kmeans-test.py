import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
image_path = "/home/haoge/ymq1/keyframes_output/keyframe_10201.jpg"  # Replace with your image path
def visualize_core_poses_on_image(core_poses, image_path, save_dir=""):
    """
    在指定背景图片上可视化每个核心姿态并标注每个关键点的序号，保存为新图
    """
    # 读取背景图片
    image = cv2.imread(image_path)

    for i, pose in enumerate(core_poses):
        # 复制原始图片，避免覆盖同一张图片
        img_copy = image.copy()

        pose = np.array(pose)  # 转换为 numpy 数组以便绘制
        x = pose[:, 0]  # x 坐标
        y = pose[:, 1]  # y 坐标

        # 在图片上绘制每个关键点
        for j, (x_i, y_i) in enumerate(zip(x, y)):
            # 用小圆点标注关键点
            cv2.circle(img_copy, (int(x_i), int(y_i)), 5, (0, 0, 255), -1)
            # 标注每个关键点的序号
            cv2.putText(img_copy, str(j + 1), (int(x_i) + 5, int(y_i) + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 保存新图像
        save_path = f"{save_dir}core_pose_{i + 1}.png"
        cv2.imwrite(save_path, img_copy)  # 保存为图片



class PoseClustering:
    def __init__(self, json_data, n_clusters=5,clustering_method="kmeans"):
        self.json_data = json_data  # 提供的JSON数据
        self.n_clusters = n_clusters  # 设置要找的核心姿态数
        self.scaler = StandardScaler()
        self.clustering_method = clustering_method.lower()  # 转换为小写方便比较

    def pnt(self):
        i=0
        for frame_id, data in self.json_data.items():
            # print(frame_id)
            bottom_keypoints = np.array(data['bottom'])
            # print(bottom_keypoints.size)
            # print(bottom_keypoints)
            if bottom_keypoints.size>1 :
                i=i+1
        print(i)

    
    def preprocess_data(self):
        """
        预处理数据，提取关节间的角度和距离等特征
        """
        X = []
        poses = []
        frame_ids = []  # 初始化 frame_id 列表
        for frame_id, data in self.json_data.items():
            # bottom_keypoints = np.array(data['bottom'])
            bottom_keypoints = np.array(data['bottom'])
            if bottom_keypoints.size<=1 :
                continue
            
            # 提取每一帧的姿态特征
            features = self.extract_features(bottom_keypoints)
            if np.any(np.isnan(features)):
                print(f"Skipping frame {features}: Features contain NaN")
                continue
            X.append(features)
            poses.append(bottom_keypoints)
            frame_ids.append(frame_id)  # 将 frame_id 添加到列表中


        X = np.array(X)
        
        # 标准化数据
        X = self.scaler.fit_transform(X)
        
        return X,poses,frame_ids
    
    def extract_features(self, bottom_keypoints):
        """
        提取每帧的12个角度特征
        """
        angle_definitions = [
            (5, 6, 11), (5, 11, 7), (6, 5, 12), (6, 12, 8),
            (7, 5, 9), (8, 6, 10), (11, 5, 12), (11, 12, 13),
            (12, 6, 11), (12, 11, 14), (13, 11, 15), (14, 12, 16)
        ]
        angles = []
        for a, b, c in angle_definitions:
            angle = self.calculate_angle(
                bottom_keypoints[a ],
                bottom_keypoints[b ],
                bottom_keypoints[c ]
            )
            angles.append(angle)
        return angles
    
    def calculate_angle(self, p1, p2, p3):
        """
        计算三点之间的夹角
        """
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def cluster_poses(self):
        """
        使用K-means聚类，找出5个核心姿态
        """
        X, poses,frame_ids = self.preprocess_data()

        if self.clustering_method == "kmeans":
            model = KMeans(n_clusters=self.n_clusters, random_state=42)
            labels = model.fit_predict(X)
            centers = model.cluster_centers_
        elif self.clustering_method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)  # 超参数可以调整
            labels = model.fit_predict(X)
            centers = None  # DBSCAN 无明确中心点
        elif self.clustering_method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=self.n_clusters)
            labels = model.fit_predict(X)
            centers = None  # 需要自己计算簇中心
        elif self.clustering_method == "gmm":
            model = GaussianMixture(n_components=self.n_clusters, random_state=42)
            labels = model.fit_predict(X)
            centers = model.means_
        else:
            raise ValueError("Unsupported clustering method. Choose from 'kmeans', 'dbscan', 'hierarchical', 'gmm'.")
        

        # 计算核心姿态和对应的 frame_id
        core_poses, core_frame_ids = [], []
        if centers is not None:  # 有明确中心点
            for center in centers:
                closest_idx = np.argmin(np.linalg.norm(X - center, axis=1))
                core_poses.append(poses[closest_idx])
                core_frame_ids.append(frame_ids[closest_idx])
        else:  # 无明确中心点的算法
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # 噪声点
                    continue
                cluster_points = X[labels == label]
                cluster_center = cluster_points.mean(axis=0)
                closest_idx = np.argmin(np.linalg.norm(X - cluster_center, axis=1))
                core_poses.append(poses[closest_idx])
                core_frame_ids.append(frame_ids[closest_idx])
        print(core_frame_ids)

        # 统计每个簇的数量
        cluster_counts = np.bincount(labels[labels >= 0])  # 排除噪声点
        for cluster_id, count in enumerate(cluster_counts):
            print(f"Cluster {cluster_id}: {count} points")

        return labels,frame_ids

        # # 使用K-means聚类
        # kmeans = KMeans(n_clusters=self.n_clusters)
        # kmeans.fit(X)
        
        # # 获取每个聚类的中心点作为核心姿态
        # core_poses_idx = kmeans.cluster_centers_
        # core_poses = []
        # core_frame_ids = []  # 新增一个列表用于存储对应的 frame_id
        # for idx in core_poses_idx:
        #     closest_pose_idx = np.argmin(np.linalg.norm(X - idx, axis=1))  # 找到最近的姿态
        #     core_poses.append(poses[closest_pose_idx])
        #     core_frame_ids.append(frame_ids[closest_pose_idx])  # 对应的 frame_id
        # print(core_frame_ids)


        # labels = kmeans.labels_

        # # 统计每个聚类中心对应的数据点数量
        # cluster_counts = np.bincount(labels)

        # for cluster_id, count in enumerate(cluster_counts):
        #     print(f"Cluster {cluster_id}: {count} points")
        # # return np.array(core_poses)
        # return np.array(core_poses),X, kmeans.labels_, kmeans.cluster_centers_, poses, frame_ids
        
        # return core_poses
    
    def predict_core_poses(self):
        core_poses,X, labels, centers, poses, frame_ids= self.cluster_poses()
        print(f"Core Poses (Cluster Centers): {core_poses}")
        visualize_core_poses_on_image(core_poses,image_path)
        
        return core_poses
    

    def visualize_clusters(self):
        """可视化数据分布和聚类中心"""
        _,X, labels, centers, poses, frame_ids = self.cluster_poses()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centers_pca = pca.transform(centers)

        plt.figure(figsize=(10, 7))
        X_pca = pca.fit_transform(X)
        centers_pca = pca.transform(centers)

        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
        plt.title("Pose Clusters Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.colorbar(scatter, label='Cluster Label')
        plt.grid(True)
        plt.savefig('cluster_visualization.png')


def load_min_similarity_indices(file_path):
    """
    从文件中加载相似度计算时的最大相似度索引
    """
    with open(file_path, 'r') as f:
        min_similarity_indices = json.load(f)
    
    # 转换为 frame_id -> min_similarity_idx 映射
    min_similarity_mapping = {str(frame_id): idx for frame_id, idx in min_similarity_indices}
    return min_similarity_mapping

def compare_similarity_and_cluster(cluster_indices, min_similarity_mapping, frame_ids, output_file="comparison_results.txt"):
    """
    比较聚类结果和相似度最大索引是否一致
    """
    match_count = 0
    with open(output_file, 'w') as f:
        for idx, frame_id in enumerate(frame_ids):
            cluster_idx = cluster_indices[idx]  # 聚类得到的索引
            min_similarity_idx = min_similarity_mapping.get(str(frame_id))  # 相似度计算时得到的最大相似度索引
            
            if cluster_idx == min_similarity_idx:  # 比较索引是否一致
                match_count += 1
            f.write(f"Frame {frame_id}: Cluster Index = {cluster_idx}, Max Similarity Index = {min_similarity_idx}\n")
    
    match_ratio = match_count / len(frame_ids)
    print(f"Match Ratio: {match_ratio:.2f}")

    # 输出到文件
    result = {
        "match_ratio": match_ratio,
        "match_count": match_count,
        "total_frames": len(frame_ids)
    }

    with open('match_comparison_results.json', 'w') as f:
        json.dump(result, f, indent=4)

# 假设我们有一个JSON文件
json_file_path = '/home/haoge/solopose/SoloShuttlePose/res 2016-1/players/player_kp/20160501-1.json'
min_similarity_file = 'min_similarity_indices.json'
# 读取JSON数据
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

pose_clustering = PoseClustering(json_data, clustering_method="hierarchical")

# pose_clustering.pnt()

# 获取5个核心姿态
# core_poses = pose_clustering.predict_core_poses()
# pose_clustering.visualize_clusters()

# 获取聚类后的索引
cluster_indices, frame_ids = pose_clustering.cluster_poses()

# 从文件中加载最大相似度索引
min_similarity_mapping = load_min_similarity_indices(min_similarity_file)

# 比较聚类结果和最大相似度结果
compare_similarity_and_cluster(cluster_indices, min_similarity_mapping, frame_ids)
