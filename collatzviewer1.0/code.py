import torchCode
import sys, os
andr = None
try:
    import android
    andr = True
except ImportError:
    andr = False
try:
    import pygame
    import sys
    import random
    import time
    from pygame.locals import *
    pygame.init() 
    fps = 1 / 3
except Exception as e:
    open('error.txt', 'w').write(str(e))


width=500
height=500
textOnImages=False

screen = pygame.display.set_mode((width, height), FULLSCREEN if andr else FULLSCREEN)

widthPygameWindow, heightPygameWindow = pygame.display.get_surface().get_size()

n=9000 #collatz n value

centerX=0.5
centerY=0.5
centerY=0.6666666666
offset=0.005

def displayNewImage():
    torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)

displayNewImage()

image = pygame.image.load('image.png')
image = pygame.transform.scale(image,(widthPygameWindow,heightPygameWindow))
screen.blit(image, (0,0))
pygame.display.update()

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                n=n+1
                torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)
            if event.button == 3:
                n=n-1
                torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)
            if event.button == 4:
                offset=offset*0.75
                torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)
            if event.button == 5:
                offset=offset*1.3333
                torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                abctemp=abctemp+1
            elif event.key == pygame.K_s:
                abctemp=abctemp+1
            elif event.key == pygame.K_a:
                abctemp=abctemp+1
            elif event.key == pygame.K_d:
                abctemp=abctemp+1  
            elif event.key == pygame.K_LEFT:
                centerY=centerY-(offset/5)
                torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)
            elif event.key == pygame.K_RIGHT:
                centerY=centerY+(offset/5)
                torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)
            elif event.key == pygame.K_UP:
                centerX=centerX+(offset/5)
                torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)
            elif event.key == pygame.K_DOWN:
                centerX=centerX-(offset/5)
                torchCode.generateImage(n, width, height, centerX, centerY, offset, textOnImages)
            elif event.key == pygame.K_ESCAPE:
                pygame.quit()

    image = pygame.image.load('image.png')
    image = pygame.transform.scale(image,(widthPygameWindow,heightPygameWindow))

    screen.blit(image,(0,0))

    pygame.display.update()
       
    time.sleep(fps)
    
