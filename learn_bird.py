from multi_layer import MultiLayerNetwork
from world import world_bird_chirp
import KTimage

# import numpy as np
# import matplotlib.pyplot as plt


if __name__ == '__main__':
    bird = world_bird_chirp()

    in_out_size, data_sets = bird.dim()

    print(bird.dim())

    network = MultiLayerNetwork(
        layout=(in_out_size, 200, in_out_size),
        transfer_function = MultiLayerNetwork.sigmoid_function,
        last_transfer_function = MultiLayerNetwork.direct_function)

    training_data = []
    for i in range(data_sets - 1):
        inputs = bird.sensor()
        bird.act()
        data_set = [inputs, bird.sensor()]
        training_data.append(data_set)

    errors = network.train_until_fit(
        training_data=training_data,
        train_steps=500,
        learn_rate=0.2,
        max_trains=1000)

    # plt.plot(errors)
    # plt.show()

    KTimage.exporttiles(
        X=network.weigths,
        h=network.layout[1],
        w=network.layout[0]+1,
        filename="test",
        x=network.layout[2],
        y=network.layout[1]+1)
