from sklearn.datasets import make_blobs
from matplotlib import pyplot
import numpy as np
import random
import pickle
import copy
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import time

def euclid_distance(x, y):
    # 欧式距离
    return np.sqrt(sum(np.square(x - y)))


def assign_points(data, centroids):
    """
    将所有数据点划分到距离最近的中心
    :param data:数据集
    :param centroids:中心点集合
    :return:
    """
    """
     cluster_points = [[centroid] for centroid in centroids]
    labels = []
    for point in data:
        # 计算节点point到每个中心的距离，并将其划分到最近的中心点
        distances = [euclid_distance(point, centroid) for centroid in centroids]
        label = np.argmin(distances)  # 选择距离最近的簇中心
        labels.append(label)
        cluster_points[label].append(point)  # 将point加入距离最近的簇中

    """
    distances = cdist(np.vstack(data), centroids, euclid_distance)
    labels = np.argmin(distances, axis=1).tolist()
    cluster_points = [[centroid] for centroid in centroids]

    for idx, point in enumerate(data):
        cluster_label = labels[idx]
        cluster_points[cluster_label].append(point)

    # print('np pair-wise distance time: {0}'.format(time.time() - start))

    return labels, cluster_points


def pam(data, centroid_num):
    """
    kmedoids
    :param data: data points over which to compute centroid_num of centroids
    :param centroid_num: number of centroids to compute
    :return:
    """
    # Randomly shuffle the data and pick the top centroid_num of points as starting values of the centroids
    indexs = list(range(len(data)))
    random.shuffle(indexs)
    init_centroids_index = indexs[:centroid_num]
    centroids = data[init_centroids_index, :]  # 中心点的数组
    labels = []  # The cluster to which each point belongs
    stop_flag = False
    # stopping criteria: stop if no centroid assignments change.
    while not stop_flag:
        stop_flag = True
        cluster_points = [[centroid] for centroid in centroids]  # make a list of centroids
        labels = []
        # For each point in the data (subset), find the closest centroid and record the index
        for point in data:
            distances = [euclid_distance(point, centroid) for centroid in centroids]
            label = np.argmin(distances)
            labels.append(label)
            cluster_points[label].append(point)

        # For each point in each cluster, find the distance with the cluster center. Make a list of these distances
        # and find the sum of the distances.
        distances = []
        for i in range(centroid_num):
            distances.extend([euclid_distance(point_1, centroids[i]) for point_1 in cluster_points[i]])
        old_distances_sum = sum(distances)

        for i in range(centroid_num):
            # For each centroid, swap centroid with another point in the subset. Then,
            # recompute cluster assignments. Note that even though only a single centroid was modified,
            # this can effect other cluster assignments. This is why all cluster assignments need to be
            # recomputed. Then compute minimum distance as before. If this distance is lower than before,
            # Make the swap permanent.
            #
            for point in data:
                new_centroids = copy.deepcopy(centroids)  # 假设的中心集合
                new_centroids[i] = point
                labels, cluster_points = assign_points(data, new_centroids)
                # 计算新的聚类误差
                distances = []
                for j in range(centroid_num):
                    distances.extend([euclid_distance(point_1, new_centroids[j]) for point_1 in cluster_points[j]])
                new_distances_sum = sum(distances)

                # If new distance is less than old, update the centroid
                if new_distances_sum < old_distances_sum:
                    old_distances_sum = new_distances_sum
                    centroids[i] = point
                    stop_flag = False # centroid assignment changed, so toggle stop_flag
    return centroids, labels, old_distances_sum
