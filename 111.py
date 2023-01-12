import numpy as np
ratios = np.array([0.5, 1, 2])
scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

num_anchors = len(ratios) * len(scales)

# initialize output anchors
anchors = np.zeros((num_anchors, 4))

    # scale base_size
anchors[:, 2:] =  np.tile(scales, (2, len(ratios))).T
areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

