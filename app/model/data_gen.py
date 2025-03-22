"""This module generates the data visualization for the model adjust process."""
import numpy as np
import cv2

from app.model.enumerators import Fields


def gen_visualization(_map: np.ndarray, vis_shape: list) -> np.ndarray:
    snake_head = np.argwhere(_map == Fields.SNAKE_HEAD.value)[0]

    types = [
        Fields.EMPTY,
        Fields.WALL,
        Fields.FOOD,
        Fields.SNAKE_BODY,
        Fields.SNAKE_HEAD
    ]

    zoom_out = [cv2.resize((_map == t.value).astype(float), vis_shape, interpolation=cv2.INTER_AREA) for t in types]
    head = zoom_out.pop(-1)
    zoom_out[-1] += head
    zoom_out = np.stack(zoom_out, axis=-1)
    zoom_in = np.zeros(vis_shape, dtype=float)

    center = list()
    for i, d in enumerate(zoom_in.shape):
        c = d / 2
        if c % 1 == 0.5:
            center.append(int(c))
        else:
            if snake_head[i] < c:
                center.append(int(c - 1))
            else:
                center.append(int(c))
    center = np.array(center)

    start = snake_head - center
    end = start + vis_shape
    zoom_start = np.zeros([2], dtype=int)
    zoom_start[start < 0] = -start[start < 0]
    start[start < 0] = 0
    zoom_end = np.array(vis_shape, dtype=int)
    for i in range(2):
        if end[i] > _map.shape[i]:
            zoom_end[i] -= end[i] - _map.shape[i]
            end[i] = _map.shape[i]
    
    zoom_in[zoom_start[0]: zoom_end[0], zoom_start[1]: zoom_end[1]] = _map[start[0]: end[0], start[1]: end[1]]
    
    zoom_in = [(zoom_in == i).astype(float) for i in range(-1, 4)]
    head = zoom_in.pop(-1)
    zoom_in[-1] += head
    zoom_in = np.stack(zoom_in, axis=-1)

    return np.concat([zoom_in, zoom_out], axis=-1)
