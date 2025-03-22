
from enum import Enum


class Fields(Enum):
    WALL = -1
    EMPTY = 0
    SNAKE_BODY = 1
    SNAKE_HEAD = 2
    FOOD = 3


class Directions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Colors(Enum):
    WALL = (156, 85, 47)
    EMPTY = (109, 255, 79)
    SNAKE_BODY = (13, 56, 3)
    SNAKE_HEAD = (8, 36, 2)
    FOOD = (207, 0, 0)