"""Field class module"""
import numpy as np

from app.model.enumerators import Fields, Directions


class Snake:
    def __init__(self, shape: tuple, n_foods: int):
        self.__map = np.zeros(shape, dtype=int)
        self.__map[0, :] = Fields.WALL.value
        self.__map[-1, :] = Fields.WALL.value
        self.__map[:, 0] = Fields.WALL.value
        self.__map[:, -1] = Fields.WALL.value

        possible_fields = np.argwhere(self.__map == Fields.EMPTY.value)
        self.__snake = np.array([possible_fields[np.random.choice(np.arange(len(possible_fields)))]], dtype=int)
        # self.__snake = np.array([int(i / 2) for i in shape], dtype=int)

        for i in range(n_foods):
            self.__create_food()

        self.__moves = 0
        self.__no_eating_moves = 0
        self.__score = 0
        self.__snake_size = 1
        self.__game_over = False

    def __create_food(self) -> None:
        possible_fields = np.argwhere(self.get_map() == Fields.EMPTY.value)

        if len(possible_fields) == 0:
            return
        
        field = possible_fields[np.random.choice(np.arange(len(possible_fields)))]
        self.__map[tuple(field)] = Fields.FOOD.value
        
    def get_map(self) -> np.ndarray:
        _map = self.__map.copy()
        _map[tuple(self.__snake.T)] = Fields.SNAKE_BODY.value
        _map[tuple(self.__snake[0].T)] = Fields.SNAKE_HEAD.value
        return _map

    def game_over(self):
        return self.__game_over
    
    def snake_pos(self):
        return self.__snake[0]
    
    def get_score(self) -> float:
        return self.__score
    
    def move(self, direction: Directions) -> None:
        if self.__game_over:
            return

        if direction == Directions.UP or direction == Directions.UP.value:
            movement = np.array((-1, 0), dtype=int)
        elif direction == Directions.DOWN or direction == Directions.DOWN.value:
            movement = np.array((1, 0), dtype=int)
        elif direction == Directions.LEFT or direction == Directions.LEFT.value:
            movement = np.array((0, -1), dtype=int)
        elif direction == Directions.RIGHT or direction == Directions.RIGHT.value:
            movement = np.array((0, 1), dtype=int)
        
        new_pos = self.snake_pos() + movement
        field = self.get_map()[tuple(new_pos.T)]

        self.__moves += 1
        self.__no_eating_moves += 1

        ret = None
        if field == Fields.EMPTY.value:
            self.__snake = np.roll(self.__snake, 1, axis=0)
            self.__snake[0] = new_pos

        elif field == Fields.FOOD.value:
            self.__snake = np.concat([[new_pos], self.__snake], axis=0)
            self.__snake_size += 1

            self.__score += (self.__snake_size ** 2) / self.__moves
            self.__create_food()
            self.__map[tuple(new_pos.T)] = Fields.EMPTY.value

            self.__no_eating_moves = 0
            ret = 1

        elif field in [Fields.SNAKE_BODY.value, Fields.WALL.value]:
            self.__game_over = True
            ret = -1

        if self.__no_eating_moves >= 100:
            self.__game_over = True
            ret = -1

        return ret
