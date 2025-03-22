import numpy as np

from app.model.enumerators import Fields, Colors


def gen_frame(_map) -> np.ndarray:
    shape = list(_map.shape)
    shape.append(3)

    frame = np.zeros(shape, dtype=np.uint8)
    frame[tuple(np.argwhere(_map == Fields.EMPTY.value).T)] = Colors.EMPTY.value
    frame[tuple(np.argwhere(_map == Fields.WALL.value).T)] = Colors.WALL.value
    frame[tuple(np.argwhere(_map == Fields.SNAKE_BODY.value).T)] = Colors.SNAKE_BODY.value
    frame[tuple(np.argwhere(_map == Fields.SNAKE_HEAD.value).T)] = Colors.SNAKE_HEAD.value
    frame[tuple(np.argwhere(_map == Fields.FOOD.value).T)] = Colors.FOOD.value

    return frame
