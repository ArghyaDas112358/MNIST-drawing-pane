#!/usr/bin/env python
"""
pip install -U pygame numpy tensorflow keras opencv-python
"""
from datetime import datetime

import pygame
import os
import sys
import cv2
from numpy import unique, argmax
import numpy as np

fname = 'mnist.h5'


def get_model():
    from keras.models import Sequential, load_model
    if os.path.exists(fname):
        return load_model(fname)
    


if __name__ == "__main__":
    model = get_model()
    fps = 60
    fps_clock = pygame.time.Clock()

    pred = '0'
    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    screen.fill((0, 0, 0))
    start = datetime.now()
    drawing = False
    will_pred = False
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                # DRAWING
                if e.button == pygame.BUTTON_RIGHT:
                    screen.fill((0, 0, 0))
                else:
                    drawing = True
            elif e.type == pygame.MOUSEBUTTONUP:
                # STOPPED drawing
                drawing = False
                if e.button == 3:
                    screen.fill((0, 0, 0))

        if drawing:
            pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (255, 255, 255), pos, 12)
            will_pred = True
        elif will_pred:
        # make prediction
            small_img = (cv2.cvtColor(cv2.resize(np.flipud(np.rot90(pygame.surfarray.array3d(screen))), (28, 28)), cv2.COLOR_RGB2GRAY) / 255.0)
            small_img =small_img.reshape(28, 28, 1)
            pred = model.predict(np.array([small_img]))
            pred = str(argmax(pred))
            will_pred = False
        pygame.display.set_caption("MNIST Pred: {} at {:.2f} FPS".format(pred, fps_clock.get_fps()))
        pygame.display.flip()
        fps_clock.tick(fps)
