import numpy as np
from tqdm import tqdm

import torch


def torch_pairwise_distances(U, V):
    tU = torch.tensor(U).float().cuda()
    tV = torch.tensor(V).float().cuda()

    normsU = (tU**2).sum(-1)
    normsV = (tV**2).sum(-1)

    dists = torch.clamp_min(
        normsU[:, None] - 2 * tU @ tV.t() + normsV[None, :], 0.0
    )
    return dists.cpu().numpy()


def compute_pairwise_distance(
    data_x, data_y=None, batch_row=10_000, batch_col=10_000
):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x

    dists = np.zeros((data_x.shape[0], data_y.shape[0]), dtype=np.float32)

    for row_begin in range(0, data_x.shape[0], batch_row):
        row_end = min(data_x.shape[0], row_begin + batch_row)
        for col_begin in range(0, data_y.shape[0], batch_col):
            col_end = min(data_y.shape[0], col_begin + batch_col)
            dists[row_begin:row_end, col_begin:col_end] = torch_pairwise_distances(
                data_x[row_begin:row_end], data_y[col_begin:col_end]
            )
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k, batch_size=10_000):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    radiis = []
    for i in range(0, input_features.shape[0], batch_size):
        from_i = i
        to_i = min(input_features.shape[0], from_i + batch_size)
        distances = compute_pairwise_distance(input_features[from_i:to_i], input_features)
        radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
        radiis.append(radii)
    return np.hstack(radiis)


@torch.no_grad()
def precision_metric(
    real_features, fake_features,
    nearest_k=5, batch_size=10_000, compute_realisms=False
):
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)

    precisions = []
    realism_scores = []
    for i in range(0, fake_features.shape[0], batch_size):
        from_i = i
        to_i = min(fake_features.shape[0], from_i + batch_size)

        distance_real_fake = compute_pairwise_distance(
            real_features, fake_features[from_i:to_i])

        precision = (
                distance_real_fake <
                np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).any(axis=0)
        precisions.append(precision)
        if compute_realisms:
            realism = (
                np.expand_dims(real_nearest_neighbour_distances, axis=1) / distance_real_fake
            ).max(axis=0)
            realism_scores.append(realism)

    if compute_realisms:
        return np.hstack(precisions).mean(), np.hstack(realism_scores)
    else:
        return np.hstack(precisions).mean(), None


@torch.no_grad()
def recall_metric(real_features, fake_features, nearest_k=5, batch_size=10_000):
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)

    recalls = []
    for i in range(0, real_features.shape[0], batch_size):
        from_i = i
        to_i = min(real_features.shape[0], from_i + batch_size)

        distance_real_fake = compute_pairwise_distance(
            real_features[from_i:to_i], fake_features)

        recall = (
                distance_real_fake <
                np.expand_dims(fake_nearest_neighbour_distances, axis=0)
        ).any(axis=1)
        recalls.append(recall)

    return np.hstack(recalls).mean()
