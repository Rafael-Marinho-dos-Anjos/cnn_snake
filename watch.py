from json import loads

import numpy as np
import cv2
import torch
import imageio

from app.model.snake import Snake
from app.model.data_gen import gen_visualization
from app.view.gen_frames import gen_frame

import onnxruntime

def inference(*inputs):
    onnx_inputs = [tensor.numpy(force=True) for tensor in inputs]
    ort_session = onnxruntime.InferenceSession(
        "./cnn.onnx", providers=["CPUExecutionProvider"]
    )
    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

    return onnxruntime_outputs

MAP_SHAPE = (25, 25)
VIS_SHAPE = (11, 11)
N_FOODS = 15

game = Snake(MAP_SHAPE, N_FOODS)
frames = list()

while not game.game_over():
    direction = inference(torch.FloatTensor(gen_visualization(game.get_map(), VIS_SHAPE)))
    direction = np.argmax(direction)
    
    game.move(direction)

    frame = gen_frame(game.get_map())
    frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_NEAREST)
    frames.append(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("snake", frame)
    cv2.waitKey(2)

imageio.mimsave("snake.gif", frames[:-5], duration=0.2)
