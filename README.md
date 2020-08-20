# PAMAE: Parallel k-Medoids Clustering with High Accuracy and Efficiency论文复现

Derived from: https://github.com/yangjianxin1/PAMAE

I have optimized the assign_points and search_centroid functions by using the native scipy.spatial.distance implementations to compute distances between list of points rather than nested for loops. These optimizations result in ~30% reduction in computation time