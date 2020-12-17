import numpy as np


def distance_transform(bimage):
    height = bimage.shape[0]
    width = bimage.shape[1]
    offset_h = 1
    offset_w = 1

    distance_transform = np.zeros((height + 2, width + 2), dtype=np.uint8)
    distance_transform_final = np.zeros((height, width), dtype=np.uint8)

    distance_transform[0, :] = 254
    distance_transform[height + offset_h, :] = 254
    distance_transform[:, 0] = 254
    distance_transform[:, width + offset_w] = 254

    for i in range(height):
        for j in range(width):
            if (bimage[i, j] != 0):
                new_i = i + offset_h
                new_j = j + offset_w
                value = min(distance_transform[i, new_j], distance_transform[new_i, j]) + 1
                if (value > 255):
                    value = 255
                distance_transform[new_i, new_j] = value

    for i in range(height - 1, -1, -1):
        for j in range(width - 1, -1, -1):
            if (bimage[i, j] != 0):
                new_i = i + offset_h
                new_j = j + offset_w
                value = min(distance_transform[new_i + 1, new_j], distance_transform[new_i, new_j + 1]) + 1
                value = min(value, distance_transform[new_i, new_j])
                if (value > 255):
                    value = 255
                distance_transform[new_i, new_j] = value
                distance_transform_final[i, j] = value

    return distance_transform_final
