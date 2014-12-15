from mlp.multi_layer import MultiLayerNetwork
from PIL import Image, ImageChops
import numpy as np
import math
import os


class PreSizer(object):

    """
    Schneidet ein Bild auf die richtige größe,
    bevor es einem Netzwerk übergeben wird
    """

    # Bildgröße mit der gearbeitet wird
    IMAGE_WIDTH = 75
    IMAGE_HEIGHT = 75
    IMGE_FORM = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH

    @staticmethod
    def trim(image):
        bg_color = image.getpixel((0, 0))
        bg = Image.new(image.mode, image.size, bg_color)
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            content = image.crop(bbox)
            content.thumbnail(PreSizer.IMGE_FORM)

            width, heigth = content.size
            # immer auf die selbe größe packen mit text in der Mitte
            empty = Image.new(image.mode,
                              (PreSizer.IMAGE_WIDTH, PreSizer.IMAGE_HEIGHT),
                              bg_color)
            empty.paste(content,
                        (math.floor((PreSizer.IMAGE_WIDTH - width) / 2),
                         math.floor((PreSizer.IMAGE_HEIGHT - heigth) / 2)))
            return empty

    @staticmethod
    def getOptimizedImage(imagePath):
        """
        lädt das Bild von der Festplatte und Schneidet es in die Richtige größe
        zoomt auf den Content
        """
        image = Image.open(imagePath)
        return PreSizer.trim(image)


class Recognizer(object):

    """Klasse zum erkennen von Zeichen"""

    # Das Netzwerk soll die Zahlen von 0 bis 9 und die Buchstaben von
    # A bis Z erkennen
    DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # hier wird das Netz auf der Festplatte gespeichert
    NETWORK_PATH = "/tmp/Recognizer"

    def __init__(self):
        super(Recognizer, self).__init__()

        # Das netzwerk zum erkennen von Zeichen
        self.recognizeNetwork = MultiLayerNetwork(
            layout=(PreSizer.IMAGE_SIZE,
                    PreSizer.IMAGE_HEIGHT,
                    len(Recognizer.DIGITS)),
            transfer_function=MultiLayerNetwork.sigmoid_function,
            last_transfer_function=MultiLayerNetwork.step_function)
        # if possible restore network status
        self.recognizeNetwork.loadWeights(Recognizer.NETWORK_PATH)

    def getDataFromImage(self, image):
        return list(map(lambda x: 0 | x[2] << 16 | x[1] << 8 | x[0],
                        image.getdata()))

    def train(self, folderpath):
        """
        Trainiert das Netzwerk mit allen Bildern in folderpath
        Alle Bilder müssen auf .jpg enden und mit dem soll Zeichen Beginnen
        Bsp: "A_irgendwas.jpg" um den Buchstaben A zu lernen.
        """

        files = [x for x in os.listdir(folderpath) if x.endswith(".jpg")]
        dataSet = []

        for image in files:
            img = PreSizer.getOptimizedImage(folderpath + "/" + image)
            data = self.getDataFromImage(img)

            expected = np.zeros(len(Recognizer.DIGITS))
            expected[Recognizer.DIGITS.find(image[0])] = 1

            dataSet.append([data, expected])

        print(files)
        print("starting training")

        self.recognizeNetwork.train_until_fit(dataSet, 200, 0.2, 50000)

    def getResult(self, imgagePath):
        img = PreSizer.getOptimizedImage(imgagePath)
        data = self.getDataFromImage(img)

        result = self.recognizeNetwork.calc(data)
        return result
        # result = self.recognizeNetwork.calc()

if __name__ == '__main__':
    recognizer = Recognizer()
    recognizer.train(
        "/home/dennis/workspace/python/Praktikum Neurale Netze/data")


    print(recognizer.getResult("/home/dennis/workspace/python/Praktikum Neurale Netze/test.jpg"))
