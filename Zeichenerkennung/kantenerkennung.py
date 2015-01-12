from multi_layer import MultiLayerNetwork
from presizer import PreSizer
import os


class EdgeDetector(object):

    """docstring for EdgeDetector"""

    FOLDER = "data"

    def __init__(self):
        super(EdgeDetector, self).__init__()

        self.outputFun = print

        size = PreSizer.IMAGE_SIZE

        self.edgeNetwork = MultiLayerNetwork(
            layout=(size, size / 2, size),
            transfer_function=MultiLayerNetwork.sigmoid_function,
            last_transfer_function=MultiLayerNetwork.direct_function)

    def train(self):
        files = [x for x in os.listdir(EdgeDetector.FOLDER)
                 if x.endswith(".jpg")]
        dataSet = []

        for image in files:
            img = PreSizer.getOptimizedImage(EdgeDetector.FOLDER + "/" + image)
            data = PreSizer.getDataFromImage(img)

            expected = data

            dataSet.append([data, expected])

        self.outputFun("starting training")

        print(files[0])
        print(len(dataSet[0][0]), PreSizer.IMAGE_SIZE)
        print(dataSet[0][0])

        self.edgeNetwork.train_until_fit(
            dataSet, 1000, 0.001, 800000)

        self.isTrained = True


def main():
    edge = EdgeDetector()
    edge.train()

if __name__ == '__main__':
    main()
