# -*- coding: utf8 -*-

from mlp.multi_layer import MultiLayerNetwork
from PIL import Image, ImageChops
import numpy as np
import math
import os
import sys

import pygame
import time
import shutil


class PreSizer(object):

    """
    Schneidet ein Bild auf die richtige größe,
    bevor es einem Netzwerk übergeben wird
    """

    # Bildgröße mit der gearbeitet wird
    IMAGE_WIDTH = 14
    IMAGE_HEIGHT = 20
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
                    PreSizer.IMAGE_SIZE / 2,
                    len(Recognizer.DIGITS)),
            transfer_function=MultiLayerNetwork.sigmoid_function,
            last_transfer_function=MultiLayerNetwork.step_function)
        # if possible restore network status
        self.isTrained = self.recognizeNetwork.loadWeights(
            Recognizer.NETWORK_PATH)

    def getDataFromImage(self, image):
        return list(map(lambda x: 0 | x[2] << 16 | x[1] << 8 | x[0],
                        image.getdata()))

    def train(self, folderpath, outputF=print):
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
        outputF("starting training")

        self.recognizeNetwork.train_until_fit(dataSet, 1000, 0.1, 800000, outputF)
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
            data = self.getDataFromImage(img)
        except AttributeError:
            return "empty"
        # img.show()

        result = self.recognizeNetwork.calc(data)
        return self.decodedAnswer(result)
        # result = self.recognizeNetwork.calc()


class Gui(object):

    """Gui class"""

    TRAIN_DATA = "data"

    DRAW_COLOR = (0, 0, 0)
    DRAW_WIDTH = 10
    BACKGROUND_COLOR = 0xffffff

    MSG_COLOR = (255, 255, 255)
    MSG_BACKGROUND_COLOR = 0x000000
    IMAGE_NAME = "current.jpg"

    def __init__(self):
        super(Gui, self).__init__()

        self.recognizer = Recognizer()

        pygame.init()
        width = 600
        self.height = 400

        self.msgAreaStart = width * 3 / 4
        self.screen = pygame.display.set_mode((width, self.height))

        self.drawArea = pygame.Surface((self.msgAreaStart, self.height))
        self.drawArea.fill(Gui.BACKGROUND_COLOR)
        self.msgArea = pygame.Surface((width * 1 / 4, self.height))
        self.msgArea.fill(Gui.MSG_BACKGROUND_COLOR)

        self.font = pygame.font.SysFont("monospace", 22)

        self.drawStart = pygame.mouse.get_pos()
        self.currentResult = ""
        self.running = True

        self.updateStatus()

    def updateStatus(self):
        self.msgArea.fill(Gui.MSG_BACKGROUND_COLOR)

        trained = self.font.render(
            "Trainiert: {}".format(self.recognizer.isTrained), True, Gui.MSG_COLOR)

        result = self.font.render(
            "Ergebnis: {}".format(self.currentResult), True, Gui.MSG_COLOR)
        self.msgArea.blit(trained, (10, 170))
        self.msgArea.blit(result, (10, 200))

    def showMsg(self, msg):
        self.handleEvents()

        self.drawArea.fill(Gui.BACKGROUND_COLOR)
        message = self.font.render(msg, True, Gui.DRAW_COLOR)
        self.drawArea.blit(message, (15, 190))
        self.screen.blit(self.drawArea, (0, 0))
        pygame.display.flip()

    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                sys.exit()

            elif event.type == pygame.MOUSEMOTION:
                drawEnd = pygame.mouse.get_pos()
                # (button1, button2, button3)
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    pygame.draw.line(
                        self.drawArea, Gui.DRAW_COLOR, self.drawStart, drawEnd, Gui.DRAW_WIDTH)
                self.drawStart = drawEnd

            elif event.type == pygame.MOUSEBUTTONUP:
                pygame.image.save(self.drawArea, Gui.IMAGE_NAME)
                self.currentResult = self.recognizer.getResult(Gui.IMAGE_NAME)
                self.updateStatus()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    self.drawArea.fill(Gui.BACKGROUND_COLOR)

                elif event.key == pygame.K_t:
                    self.recognizer.train(Gui.TRAIN_DATA, self.showMsg)
                    self.updateStatus()

                elif event.key == pygame.K_n:
                    self.saveCurrentImage()

    def saveCurrentImage(self):
        """Speichert das aktuelle Bild in den Lerndaten"""
        # Eingabebox erstellen und anzeigen
        inputString = ""
        while True:
            # get input
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                self.running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    break
                elif event.key == pygame.K_BACKSPACE:
                    inputString = inputString[:-1]
                elif event.key < 128:
                    inputString = inputString + chr(event.key)

            self.drawArea.fill(Gui.BACKGROUND_COLOR)
            pygame.draw.rect(self.drawArea, Gui.DRAW_COLOR, ((
                self.msgAreaStart - 200) / 2,
                (self.height - 30) / 2, 200, 30), 5)

            box_text = self.font.render(
                "Soll Ergebnis: " + inputString, True, Gui.DRAW_COLOR)
            self.drawArea.blit(box_text,
                               ((self.msgAreaStart - 190) / 2,
                                (self.height - 15) / 2))

            self.screen.blit(self.drawArea, (0, 0))
            pygame.display.flip()

        self.drawArea.fill(Gui.BACKGROUND_COLOR)

        if (len(inputString) > 0):
            timestr = time.strftime("%d%m%Y-%I%M%S")
            dest = Gui.TRAIN_DATA + "/" + inputString + "_" + timestr + ".jpg"
            shutil.copyfile(Gui.IMAGE_NAME, dest)

    def run(self):
        while self.running:
            self.handleEvents()
            self.screen.blit(self.drawArea, (0, 0))
            self.screen.blit(self.msgArea, (self.msgAreaStart, 0))
            pygame.display.flip()


def main():
    gui = Gui()
    gui.run()


if __name__ == '__main__':
    main()
