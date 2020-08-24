import time
import multiprocessing
from multiprocessing import cpu_count
import random
from kmedoids import pam
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import pickle
import argparse
import os
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
def setup_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', default=100000, type=int, required=False, help='Number of data points generated')
    parser.add_argument('--subset_size', default=1000, type=int, required=False, help='phase 1: The size of the sampled subset')
    parser.add_argument('--subset_num', default=2, type=int, required=False, help='phase 1: Number of sampled subset')
    parser.add_argument('--centroid_num', default=10, type=int, required=False, help='Number of cluster centers')
    return parser.parse_args()


args = setup_args()


def euclid_distance(x, y):
    # 欧式距离
    return np.sqrt(sum(np.square(x - y)))


def generate_data(n_points, centroid_num, n_features):
    """
    生成数据
    :param n_points: 生成数据的数量
    :param centroid_num: 生成数据的中心点数量
    :param n_features: 数据维度
    :return:
    """
    data, target = make_blobs(n_samples=n_points, n_features=n_features, centers=centroid_num)
    # 添加噪声
    np.put(data, [n_points, 0], 10, mode='clip')
    np.put(data, [n_points, 1], 10, mode='clip')
    # 画图
    # pyplot.scatter(data[:, 0], data[:, 1], c=target)
    # pyplot.title("generate data")
    # pyplot.show()
    with open("data", "wb") as f:
        pickle.dump(data.tolist(), f)
    return data.tolist()


def assign_points(data, centroids, dist_func):
    """
    assigns points to closest matching centroid
    :param data:数据集
    :param centroids:中心点集合
    :return:
    """

    distances_sum = 0
    distances = cdist(np.vstack(data), centroids, euclid_distance)
    labels = np.argmin(distances, axis=1).tolist()
    cluster_points = [[centroid] for centroid in centroids]

    for idx, point in enumerate(data):
        cluster_label = labels[idx]
        distances_sum += distances[idx][cluster_label]
        cluster_points[cluster_label].append(point)

    return labels, cluster_points, distances_sum


def sampling_and_clustering(data, n_samples, centroid_num, dist_func):
    """
    对data进行随机采样，并且进行聚类
    :param data:数据集
    :param n_samples:每个子集的数据点的数量
    :param centroid_num:中心点的数量
    :return:
    """
    if n_samples > len(data):
        return data
    subset = random.sample(data, n_samples)
    subset = np.array(subset)
    # 对随机采样的子集进行聚类，获得子集的中心集合centroids
    centroids, _, _, assign_points_exec_time = pam(subset, centroid_num)
    # 将Entire data的点划分到最近的中心，计算聚类误差
    labels, cluster_points, distances_sum = assign_points(data, centroids, dist_func)
    return centroids, labels, distances_sum, assign_points_exec_time


def search_centroid(data, dist_func):
    """
    # Consider each point in the data. For each point, compute sum of distances with all other points
    # Then pick the point that minimizes that distance. This is robust to outliers (because outliers will
    # be far away from all other points and will not influence the choice of the point picked as the centroid)
    :param data:数据集
    :return:
    """

    """
     min_distances_sum = float("Inf")
    centroid = None
    start = time.time()
    for point in data:
        # 计算每个节点到假定中心的距离
        distances = [dist_func(point_1, point) for point_1 in data]
        distances_sum = sum(distances)
        # 计算出该聚簇中各个点与其他所有点的总和，若是有小于当前中心点的距离总和的，中心点去掉
        if distances_sum < min_distances_sum:
            min_distances_sum = distances_sum
            centroid = point

    print('brute force pair-wise distance time: {0}'.format(time.time() - start))
    """

    # more efficient implementation
    start = time.time()
    dist = squareform(pdist(np.vstack(data), dist_func))
    sum_dist = np.sum(dist, axis=1)
    idx = np.argmin(sum_dist)
    min_dist = sum_dist[idx]
    centroid = data[idx]
    # print('np pair-wise distance time: {0}'.format(time.time() - start))
    return centroid


def phase1(data, subset_size, subset_num, centroid_num, dist_func, pool):
    """
    In phase 1, we randomly select subset_size samples from the dataset and run k-medoids on each subset. We
    then select the centroids for the subset with the lowest sum of points-to-cluster-center distance. The point
    of doing this is to lower computation requirements. k-medoid is a N^2 calculation and performing the calculation
    over a subset of points (lower N) lowers the computation requirements. By sampling a large number of subsets,
    we increase the odds that one of the subsets includes representative points from each cluster
    Once we identify the centroid of the clusters, we recompute the centroids using the entire dataset
    :param data: The full dataset over which we want to perform clustering
    :param subset_size: Size of randomly sampled subsets
    :param subset_num: Number of randomly sampled subsets
    :param centroid_num: Number of centroids to compute (for each subset, centroid_num centroids will be computed)
    :param pool: Multiprocessing pool
    :return: best matching centroids, labels for points in best subset, minimum distance, total execution time
    """
    start = time.perf_counter()  # 开始计时
    results = []
    for i in range(subset_num):
        result = pool.apply_async(sampling_and_clustering, (data, subset_size, centroid_num, dist_func))  # 异步并行计算
        results.append(result)

    min_distancec_sum = float('inf')
    best_labels = None
    best_centroids = None

    assign_points_exec_time = 0
    for i in range(0, subset_num):
        centroids, labels, distances_sum, assign_points_exec_time_ = results[i].get()
        assign_points_exec_time += assign_points_exec_time_
        if distances_sum < min_distancec_sum:
            min_distancec_sum = distances_sum
            best_centroids = centroids
            best_labels = labels
    end = time.perf_counter()  # 计时结束
    phase1_time = end - start  # 耗费时间phase1 消耗的时间
    print("PHASE 1 Processing time：{}".format(phase1_time))
    print("PHASE 1 Processing time (assign_points)：{}".format(assign_points_exec_time/subset_num))
    print("PHASE 1 Finished：")
    for centroid in best_centroids:
        print(centroid)
    print("minimum distance:{}".format(min_distancec_sum))
    return best_centroids, best_labels, min_distancec_sum, phase1_time


def phase2(data, centroids, centroid_num, dist_func, pool):
    start = time.perf_counter()  # 开始计时

    # In phase2, we assign labels to all the points in the dataset. Recall that in phase 1, we only consider points in
    # that subset. Now we are considering all points in the dataset, and assigning each point to its closest matching
    # cluster.
    labels, cluster_points, _ = assign_points(data, centroids, dist_func)

    results = []
    # Now consider each cluster one by one and search for the centroid again. This needs to be done because
    # Each cluster now consists of all matching points in the dataset
    for i in range(centroid_num):
        result = pool.apply_async(search_centroid, (cluster_points[i], dist_func,))  # 异步并行计算
        results.append(result)
    new_centroids = [result.get() for result in results]

    # Now re-assign data points, because the cluster centroids may have changed as a result of the previous centroid
    # search
    labels, cluster_points, distances_sum = assign_points(data, new_centroids, dist_func)
    end = time.perf_counter()
    phase2_time = end - start
    print("PHASE 2 execution time：{}".format(phase2_time))
    print("PHASE 2 Finished：")
    for centroid in new_centroids:
        print(centroid)
    print("Sum of inter point-centroid distances:{}".format(distances_sum))
    return new_centroids, labels, distances_sum, phase2_time


def draw_scatter(title, x, y, centroids, labels, n_points, subset_size, subset_num, centroid_num):
    pyplot.scatter(x, y, c=labels)
    centroids_x = []
    centroids_y = []
    for centroid in centroids:
        centroids_x.append(centroid[0])
        centroids_y.append(centroid[1])
    pyplot.scatter(centroids_x, centroids_y, c="r", marker="p")
    pyplot.title(title)
    path = "results"
    if not os.path.exists(path):
        os.mkdir(path)
    pyplot.savefig(
        "{}/{}_{}_{}_{}_{}.png".format(path, title, n_points, subset_size, subset_num, centroid_num))
    pyplot.show()


def main():
    n_points = args.n_points  # Total number of data points
    subset_size = args.subset_size  # Size of a subset to sample
    subset_num = args.subset_num  # Number of subsets to samples
    centroid_num = args.centroid_num  # Number of centroids to identify
    n_features = 2  # Dimension of each data point
    # use data from file for consistent perf comparison
    with open("data", "rb") as f:
        data = pickle.load(f)
    # data = generate_data(n_points, centroid_num, n_features)
    # pool = multiprocessing.Pool(processes=cpu_count())
    pool = multiprocessing.Pool(processes=11)
    ##########################
    #   PHASE 1
    ##########################
    print("PHASE 1...")
    # In phase 1, we randomly select subset_size samples from the dataset and run k-medoids on each subset. We
    # then select the centroids for the subset with the lowest sum of points-to-cluster-center distance. The point
    # of doing this is to lower computation requirements. k-medoid is a N^2 calculation and performing the calculation
    # over a subset of points (lower N) lowers the computation requirements. By sampling a large number of subsets,
    # we increase the odds that one of the subsets includes representative points from each cluster
    # Once we identify the centroid of the clusters, we recompute the centroids using the entire dataset
    centroids, labels, distances_sum, phase1_time = phase1(data, subset_size, subset_num,
                                                           centroid_num, euclid_distance, pool)
    data = np.array(data)
    draw_scatter("PHASE1", data[:, 0], data[:, 1], centroids, labels, n_points, subset_size, subset_num, centroid_num)

    ##########################
    #   PHASE 2
    ##########################
    print("PHASE 2...")
    centroids, labels, distances_sum, phase2_time = phase2(data, centroids, centroid_num, euclid_distance, pool)
    draw_scatter("PHASE2", data[:, 0], data[:, 1], centroids, labels, n_points, subset_size, subset_num, centroid_num)
    print("Total execution time: {}".format(phase1_time + phase2_time))

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()


# cupy install instructions:
# Install cuda:
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal
# make sure to add cuda binaries (eg nvcc) to the PATH:
# export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
# install cudnn
# install nccl
    ## after installing the local/network deb, do sudo apt update
    ## then sudo apt install libnccl2=2.7.8-1+cuda11.0 libnccl-dev=2.7.8-1+cuda11.0
# pip install cupy