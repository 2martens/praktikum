from multi_layer import MultiLayerNetwork
from presizer import PreSizer
import numpy as np
import os
import operator


class Recognizer(object):

    """Klasse zum erkennen von Zeichen"""

    # Das Netzwerk soll die Zahlen von 0 bis 9 und die Buchstaben von
    # A bis Z erkennen
    DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
    # hier wird das Netz auf der Festplatte gespeichert
    NETWORK_PATH = "/tmp/Recognizer.npy"

    def __init__(self, outputFunction=print):
        super(Recognizer, self).__init__()

        self.outputFun = outputFunction

        # Das netzwerk zum erkennen von Zeichen
        self.recognizeNetwork = MultiLayerNetwork(
            layout=(PreSizer.IMAGE_SIZE,
                    PreSizer.IMAGE_HEIGHT * 2,
                    len(Recognizer.DIGITS)),
            transfer_function=MultiLayerNetwork.sigmoid_function,
            last_transfer_function=MultiLayerNetwork.theWinnerTakesItAll,
            output_function=outputFunction)

        self.isTrained = False

    def train(self, folderpaths, learnrate=0.1, maxtrains=800000):
        """
        Trainiert das Netzwerk mit allen Bildern in den Ordnern von folderpaths
        Alle Bilder müssen auf .jpg enden und mit dem soll Zeichen Beginnen
        Bsp: "A_irgendwas.jpg" um den Buchstaben A zu lernen.

        folderpaths - Eine liste von Ordnern
        """

        dataSet = []
        for folderpath in folderpaths:
            try:
                files = [x for x in os.listdir(folderpath)
                         if x.endswith(".jpg")]

                for image in files:
                    img = PreSizer.getOptimizedImage(folderpath + "/" + image)
                    data = PreSizer.getDataFromImage(img)

                    expected = np.zeros(len(Recognizer.DIGITS))
                    expected[Recognizer.DIGITS.find(image[0])] = 1

                    dataSet.append([data, expected])
            except FileNotFoundError:
                print("no such folder \"{}\" -> ignoring".format(folderpath))

        self.outputFun("starting training")

        self.recognizeNetwork.train_until_fit(
            dataSet, 1000, learnrate, maxtrains)
        self.isTrained = True

    def decodedAnswer(self, result):
        data = list(result)
        if 1 in data:
            return Recognizer.DIGITS[data.index(1)]
        else:
            return "not classified"

    def getResult(self, imgagePath):
        img = PreSizer.getOptimizedImage(imgagePath)
        # der Presizer gibt wenn das bild leer ist kein image Type zurück
        try:
            data = PreSizer.getDataFromImage(img)
        except AttributeError:
            print("empty image")
            return "empty"
        # img.show()

        result = self.recognizeNetwork.calc(data)
        return self.decodedAnswer(result)

    def loadNetwork(self):
        self.isTrained = self.recognizeNetwork.loadWeights(
            Recognizer.NETWORK_PATH)

    def saveNetwork(self):
        self.recognizeNetwork.saveWeights(Recognizer.NETWORK_PATH)

    def getResults(self, imgagePath):
        img = PreSizer.getOptimizedImage(imgagePath)
        # der Presizer gibt wenn das bild leer ist kein image Type zurück
        try:
            data = PreSizer.getDataFromImage(img)
        except AttributeError:
            print("empty image")
            return []
        # img.show()

        prefun = self.recognizeNetwork.last_layer_transfer
        self.recognizeNetwork.last_layer_transfer = Recognizer.toPercentage

        result = self.recognizeNetwork.calc(data)
        self.recognizeNetwork.last_layer_transfer = prefun

        probabilities = {}
        for index, prob in enumerate(result):
            if prob > 0:
                probabilities[Recognizer.DIGITS[index]] = prob

        probSorted = sorted(
            probabilities.items(), key=operator.itemgetter(1), reverse=True)

        return probSorted

    @staticmethod
    def toPercentage(value):
        # minValue = np.amin(value)
        # if minValue < 0:
            # value = np.array(list(map(lambda x: x - minValue, value)))
        # s = np.sum(value)
        s = 0
        for x in value:
            if x >= 0:
                s += x

        return np.array(list(map(lambda x: x / s, value)))


def main():
    net = Recognizer(print)
    net.train(["data", "gen_data"], 0.1)

    testDataDir = "testData"

    correctCount = 0
    wrongCount = 0

    print("testing with test data")

    files = [x for x in os.listdir(testDataDir)
             if x.endswith(".jpg")]

    for index, image in enumerate(files):
        expected = files[index][0]
        result = net.getResult(testDataDir + "/" + image)

        print("{} - {}".format(expected, result))

        if result == expected:
            correctCount += 1
        else:
            wrongCount += 1

    print("correct: {}   wrong:{}  --> {}%".
          format(correctCount, wrongCount,
                 (correctCount / (correctCount + wrongCount)) * 100))


if __name__ == '__main__':
    main()
