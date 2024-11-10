#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def gen_box_data(trainset, y_train, length=1000, image_size=32, box_size=4,
                 offset1=7, offset2=23, shiftdiv=6):

    np.random.seed(42)
    img = np.zeros([3, image_size, image_size], dtype=float)
    patch = np.ones([box_size, box_size], dtype=float)
    off_size = image_size - box_size

    for i in range(length):

        if i % 2 == 0:
            im = img.copy()
            offsetx = np.random.randint(off_size/2 - box_size)
            offsety = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1
            # obj 1
            im[0, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            # obj 2
            offsetx2 = np.random.randint(offsetx, off_size)
            offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1

            while ((abs(offsetx-offsetx2) < box_size+1)):
                offsetx2 = np.random.randint(offsetx, off_size)
                offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1

            im[1, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch

            trainset[i] = im
            y_train[i] = 0

        elif i % 2 == 1:
            im = img.copy()
            offsetx = np.random.randint(off_size/2 - box_size)
            offsety = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2
            # obj 1
            im[1, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            # obj 2
            offsetx2 = np.random.randint(offsetx, off_size)
            offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2

            while ((abs(offsetx-offsetx2) < box_size+1)):
                offsetx2 = np.random.randint(offsetx, off_size)
                offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2

            im[0, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch

            trainset[i] = im
            y_train[i] = 1
   
    return trainset, y_train

def gen_box_data_test(testset1, y_test1, testset2, y_test2, length=1000,
                      image_size=32, box_size=4, offset1=7, offset2=23,
                      shiftdiv=6):

    np.random.seed(42)
    img = np.zeros([3, image_size, image_size], dtype=float)
    patch = np.ones([box_size, box_size], dtype=float)
    off_size = image_size - box_size

    for i in range(length):

        if i % 2 == 0:
            im = img.copy()
            img2 = img.copy()
            offsetx = np.random.randint(off_size/2 - box_size)
            offsety = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1
            # obj 1
            im[0, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            offsety += 16
            img2[0, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            # obj 2
            offsetx2 = np.random.randint(offsetx, off_size)
            offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1

            while ((abs(offsetx-offsetx2) < box_size+1)):
                offsetx2 = np.random.randint(offsetx, off_size)
                offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1
    
            im[1, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch
            offsety2 += 16
            img2[1, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch

            testset1[i] = im
            y_test1[i] = 0
            testset2[i] = img2
            y_test2[i] = 0

        elif i % 2 == 1:
            im = img.copy()
            img2 = img.copy()
            offsetx = np.random.randint(off_size/2 - box_size)
            offsety = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2
            # obj 1
            im[1, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            offsety -= 16
            img2[1, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            # obj 2
            offsetx2 = np.random.randint(offsetx, off_size)
            offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2

            while ((abs(offsetx-offsetx2) < box_size+1)):
                offsetx2 = np.random.randint(offsetx, off_size)
                offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2

            im[0, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch
            offsety2 -= 16
            img2[0, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch

            testset1[i] = im
            y_test1[i] = 1
            testset2[i] = img2
            y_test2[i] = 1
     
    return testset1, y_test1, testset2, y_test2