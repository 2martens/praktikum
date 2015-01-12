from recognizer import Recognizer
import pygame
import sys
import time
import shutil


class Gui(object):

    """Gui class"""

    TRAIN_DATA = "data"

    DRAW_COLOR = (0, 0, 0)
    DRAW_WIDTH = 10
    BACKGROUND_COLOR = 0xffffff

    MSG_COLOR = (255, 255, 255)
    MSG_BACKGROUND_COLOR = 0x000000
    IMAGE_NAME = "current.jpg"

    MSG_FONT_SIZE = 21

    def __init__(self):
        super(Gui, self).__init__()

        self.recognizer = Recognizer(self.showMsg)

        pygame.init()
        self.width = 600
        self.height = 400

        self.font = pygame.font.SysFont("", Gui.MSG_FONT_SIZE)

        self.msgAreaStart = self.width * 3 / 4
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.drawArea = pygame.Surface((self.msgAreaStart, self.height))
        self.drawArea.fill(Gui.BACKGROUND_COLOR)

        self.msgArea = pygame.Surface((self.width * 1 / 4, self.height / 2))
        self.msgArea.fill(Gui.MSG_BACKGROUND_COLOR)

        self.keyMapArea = self.getKeyMapSurface()

        self.drawStart = pygame.mouse.get_pos()
        self.currentResult = ""
        self.running = True

        self.updateStatus()

    def getKeyMapSurface(self):
        clearString = "c - Zurücksetzen"
        newImgString = "n - Bild speichern"
        trainString = "t - Trainieren"
        saveString = "s - N. speicher"
        loadString = "l - N. laden"

        clearImg = self.font.render(clearString, True, Gui.MSG_COLOR)
        saveImg = self.font.render(newImgString, True, Gui.MSG_COLOR)
        train = self.font.render(trainString, True, Gui.MSG_COLOR)
        savetxt = self.font.render(saveString, True, Gui.MSG_COLOR)
        loadtxt = self.font.render(loadString, True, Gui.MSG_COLOR)

        surface = pygame.Surface((self.width * 1 / 4, self.height / 2))
        surface.fill(Gui.MSG_BACKGROUND_COLOR)

        surface.blit(clearImg, (10, 10))
        surface.blit(saveImg, (10, 10 + Gui.MSG_FONT_SIZE))
        surface.blit(train, (10, 10 + (2 * Gui.MSG_FONT_SIZE)))
        surface.blit(savetxt, (10, 10 + (3 * Gui.MSG_FONT_SIZE)))
        surface.blit(loadtxt, (10, 10 + (4 * Gui.MSG_FONT_SIZE)))

        return surface

    def updateStatus(self):
        self.msgArea.fill(Gui.MSG_BACKGROUND_COLOR)

        trained = self.font.render(
            "Trainiert: {}".format(self.recognizer.isTrained), True, Gui.MSG_COLOR)

        result = self.font.render(
            "Ergebnis: {}".format(self.currentResult), True, Gui.MSG_COLOR)
        self.msgArea.blit(trained, (10, 148))
        self.msgArea.blit(result, (10, 170))

    def showMsg(self, msg):
        self.handleEvents()

        self.drawArea.fill(Gui.BACKGROUND_COLOR)
        message = self.font.render(msg, True, Gui.DRAW_COLOR)
        self.drawArea.blit(message, (15, self.height - Gui.MSG_FONT_SIZE))
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
                        self.drawArea, Gui.DRAW_COLOR,
                        self.drawStart, drawEnd, Gui.DRAW_WIDTH)
                self.drawStart = drawEnd

            elif event.type == pygame.MOUSEBUTTONUP:
                pygame.image.save(self.drawArea, Gui.IMAGE_NAME)
                self.currentResult = self.recognizer.getResult(Gui.IMAGE_NAME)
                self.updateStatus()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    self.drawArea.fill(Gui.BACKGROUND_COLOR)

                elif event.key == pygame.K_t:
                    self.recognizer.train(Gui.TRAIN_DATA)
                    self.updateStatus()

                elif event.key == pygame.K_n:
                    self.saveCurrentImage()

                elif event.key == pygame.K_s:
                    self.recognizer.saveNetwork()

                elif event.key == pygame.K_l:
                    self.recognizer.loadNetwork()
                    self.updateStatus()

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
            self.screen.blit(
                self.msgArea, (self.msgAreaStart, self.height / 2))
            self.screen.blit(self.keyMapArea, (self.msgAreaStart, 0))
            pygame.display.flip()


def main():
    gui = Gui()
    gui.run()


if __name__ == '__main__':
    main()
