import numpy as np


def distanceTransform(bimage):
    height = bimage.shape[0]
    width = bimage.shape[1]
    offset_h = 1
    offset_w = 1

    distanceTransform = np.zeros((height + 2, width + 2), dtype=np.uint8)
    distanceTransform_final = np.zeros((height, width), dtype=np.uint8)

    distanceTransform[0, :] = 254
    distanceTransform[height + offset_h, :] = 254
    distanceTransform[:, 0] = 254
    distanceTransform[:, width + offset_w] = 254

    for i in range(height):
        for j in range(width):
            if (bimage[i, j] != 0):
                new_i = i + offset_h
                new_j = j + offset_w
                value = min(distanceTransform[i, new_j], distanceTransform[new_i, j]) + 1
                if (value > 255):
                    value = 255
                distanceTransform[new_i, new_j] = value

    for i in range(height - 1, -1, -1):
        for j in range(width - 1, -1, -1):
            if (bimage[i, j] != 0):
                new_i = i + offset_h
                new_j = j + offset_w
                value = min(distanceTransform[new_i + 1, new_j], distanceTransform[new_i, new_j + 1]) + 1
                value = min(value, distanceTransform[new_i, new_j])
                if (value > 255):
                    value = 255
                distanceTransform[new_i, new_j] = value
                distanceTransform_final[i, j] = value

    return distanceTransform_final
