#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import glob
import numpy as N
import mlp.KTimage as KT


class world_retina(object):
    # parameters:
    # path        = directory from where to read the images
    # max_num_img = number of images to read
    # max_width   = widest image    (needed for memory allocation)
    # max_height  = highest image                "
    # sizeRetina  = edge length of square-shaped image patch that will be returned by sensor()
    #               must be an even number!

    def __init__(self, path, max_num_img, max_width, max_height, sizeRetina):
        self.__width = N.zeros(max_num_img)
        self.__height = N.zeros(max_num_img)
        self.__edge = sizeRetina
        margin = self.__edge  # N.ceil(self.__edge/2.)
        self.__range_width = N.zeros((max_num_img, 2))
        self.__range_height = N.zeros((max_num_img, 2))
        print("shape self.__range_width = {}".format(self.__range_width))
        tmp = str(path) + '/*.pgm'
        self.__data = N.zeros((max_num_img, max_height, max_width))
        counter = 0
        for infile in glob.glob(tmp):
            print("Processing {} as image {}".format(infile, counter))
            # read image
            img, self.__height[counter], self.__width[
                counter] = KT.importimage(str(infile))
            img = N.reshape(
                img, (self.__height[counter], self.__width[counter]))
            # enlarge img with zeros to size of __data[.]
            diff_height = max_height - self.__height[counter]
            if diff_height:
                A = N.zeros((diff_height, self.__width[counter]))
                img = N.concatenate((img, A), axis=0)
            diff_width = max_width - self.__width[counter]
            if diff_width:
                B = N.zeros((self.__height[counter], diff_width))
                img = N.concatenate((img, B), axis=1)
            # copy img to __data
            self.__data[counter] = N.copy(img)  # -N.mean(img))
            # scale values to [0..1]
            self.__data[counter] /= N.max(self.__data[counter])
            # compute safety margin
            self.__range_width[counter] = [
                margin, self.__width[counter] - margin]
            self.__range_height[counter] = [
                int(margin), int(self.__height[counter] - margin)]
            counter += 1
            if counter >= max_num_img:
                break
        self.__numOfData = counter
        self.newinit()

    def cut_patch(self, n, w, h):
        offset = self.__edge / 2
        # subtract mean
        lo = w - offset
        hi = w + offset
        bo = h - offset
        top = h + offset
        # subtract mean
        tmp = self.__data[n, bo:top, lo:hi]
        # subtraction of the mean value of each patch!
        return N.copy((tmp - N.mean(tmp)))

    def dim(self):
        # square input - note that this is one patch - there is no maximum
        # number of patches nor any seqlen!
        return (self.__edge, self.__edge)

    def newinit(self):
        # choose random picture
        self.rnd_pic = N.random.randint(0, self.__numOfData)
        # pic random retina location within picture
        self.rnd_w = N.random.randint(
            self.__range_width[self.rnd_pic][0],
            self.__range_width[self.rnd_pic][1])
        self.rnd_h = N.random.randint(
            self.__range_height[self.rnd_pic][0],
            self.__range_height[self.rnd_pic][1])

    def act(self):
        # world reaction - always get a new random image patch
        self.newinit()

    def sensor(self):
        # return an image patch, flattened to a vector of length __edge*__edge
        retval = self.cut_patch(self.rnd_pic, self.rnd_w, self.rnd_h)
        return N.copy(N.reshape(retval, N.shape(retval)[0] * N.shape(retval)[1]))

if __name__ == "__main__":
    print("testing world_retina")
    myData = world_retina('BSDS30_filt', 1, 481, 481, 2)
    tinypatch = myData.sensor()
    print("2x2 pixel sample:", tinypatch)
