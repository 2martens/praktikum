#!/usr/bin/python

from PIL import Image, ImageChops
import math
import os
import shutil


class PreSizer(object):

    """
    Schneidet ein Bild auf die richtige größe,
    bevor es einem Netzwerk übergeben wird
    """

    # Bildgröße mit der gearbeitet wird
    IMAGE_WIDTH = 20
    IMAGE_HEIGHT = 30
    IMGE_FORM = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH

    @staticmethod
    def trim(image):
        bg_color = image.getpixel((0, 0))
        if (bg_color != 255):
            print(bg_color)
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
        image = image.convert("L")
        return PreSizer.trim(image)

    @staticmethod
    def getDataFromImage(image):
        # print("pixel: ", image.im.getpixel((0, 0)))
        return list(map(lambda x: int(x / 255),
                        image.getdata()))
        # return image.histogram()


def main():
    print("create Folder with optimized images..")

    dataDir = "data"
    optDataDir = "opt_data"

    try:
        shutil.rmtree(optDataDir)
    except FileNotFoundError:
        pass

    os.makedirs(optDataDir)

    files = [x for x in os.listdir(dataDir)
             if x.endswith(".jpg")]

    for index, image in enumerate(files):
        img = PreSizer.getOptimizedImage(dataDir + "/" + image)
        img.save("{}/{}".format(optDataDir, files[index]))

if __name__ == '__main__':
    main()
