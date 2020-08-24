import torch
from scipy.spatial.distance import cdist
import numpy as np
import time

print(torch.__version__)
dev = torch.cuda.current_device()
N = 5
M1 = 180000 # 100000 results in OOM on a 1080 Ti GPU in full precision
M2 = int(M1/10)
pts1 = np.random.random_sample((M1, N))
pts2 = np.random.random_sample((M2, N))

def py_cdist(pts1, pts2):
    distances = cdist(pts1, pts2, metric='euclidean')
    labels = np.argmin(distances, axis=1).tolist()
    cluster_points = [[pt2] for pt2 in pts2]

    for idx, point in enumerate(pts1):
        cluster_label = labels[idx]
        cluster_points[cluster_label].append(point)
    return labels


def torch_cdist(pts1, pts2, use_half=False):
    pts1_t = torch.from_numpy(pts1)
    pts2_t = torch.from_numpy(pts2)

    if use_half:
        # convert to half precision
        pts1_t = pts1_t.to(torch.half).cuda(0)
        pts2_t = pts2_t.to(torch.half).cuda(0)
    else:
        pts1_t = pts1_t.cuda(0)
        pts2_t = pts2_t.cuda(0)

    distances = torch.cdist(pts1_t, pts2_t, p=2)

    labels = torch.argmin(distances, axis=1).cpu().numpy().tolist()
    cluster_points = [[pt2] for pt2 in pts2]

    for idx, point in enumerate(pts1):
        cluster_label = labels[idx]
        cluster_points[cluster_label].append(point)
    return labels


start = time.time()
for i in range(0, 4):
    labels_torch = torch_cdist(pts1, pts2, use_half=True)
end = time.time()
print('cdist-torch-GPU: {0}'.format(end-start))

start = time.time()
for i in range(0, 4):
    labels_py = py_cdist(pts1, pts2)
end = time.time()
print('cdist-CPU: {0}'.format(end-start))

are_identical = labels_py == labels_torch
print("labels are identical? :{0}".format(are_identical))