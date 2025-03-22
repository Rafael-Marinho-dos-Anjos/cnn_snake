from copy import deepcopy

import torch
from torch import nn
import matplotlib.pyplot as plt

from learning_models.cnn_torch_2 import CNN
from app.model.snake import Snake
from app.model.data_gen import gen_visualization


MAP_SHAPE = (50, 50)
VIS_SHAPE = (11, 11)
N_FOODS = 25
EPOCHS = 10000

last_10_scores = list()
best_last_10 = 0
scores = list()

model = CNN(8, 11)
model.load_state_dict(torch.load("learning_models/trained_weights/cnn_21.pth"))
optim = torch.optim.Adam(model.parameters(), lr=5e-6, amsgrad=True)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    game = Snake(MAP_SHAPE, N_FOODS)
    no_eating = 0

    while not game.game_over():
        vis = gen_visualization(game.get_map(), VIS_SHAPE)
        m = torch.FloatTensor(vis)
        pred = model(m)

        response = game.move(torch.argmax(pred).item())

        if response is not None:
            if response == 1:
                y = torch.zeros([4])
                y[torch.argmax(pred).item()] = 1

            elif response == -1:
                y = torch.ones([4])
                y[torch.argmax(pred).item()] = 0

                y = y * 5
        
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
        else:
            no_eating += 1
            if no_eating >= 50:
                model.zero_grad()
                no_eating = 0

            if game.game_over(): # Excedeu o número de jogadas sem comer
                y = torch.ones([4])
                y[torch.argmax(pred).item()] = 0

                y = y * 10
        
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()

    print(f"epoch: [{epoch+1}/{EPOCHS}]  \tscore: {game.get_score():.2f} {'<- No food' if game.get_score() == 0 else ''}", end="")

    last_10_scores.append(game.get_score())
    if len(last_10_scores) > 10:
        last_10_scores.pop(0)
        mean_score = sum(last_10_scores) / 10
        scores.append(mean_score)
        if mean_score > best_last_10:
            best_model = deepcopy(model)
            best_last_10 = mean_score
            print("       <- New best!!!", end="")
    else:
        scores.append(sum(last_10_scores) / len(last_10_scores))

    print()

plt.plot(scores)
plt.title("Score médio")
plt.savefig("scores.png")
# plt.show()

onnx = torch.onnx.export(best_model, (torch.zeros((11, 11, 8), dtype=torch.float32),), dynamo=True)
onnx.save("cnn.onnx")

PATH = "learning_models/trained_weights/cnn.pth"
torch.save(best_model.state_dict(), PATH)
