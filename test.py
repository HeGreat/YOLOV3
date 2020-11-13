import sys
import argparse
from yolo import YOLO,detect_video
from PIL import Image

def detect_img(yolo):
    img='demo.jpg'
    try:
        image=Image.open(img)
    except:
        print('Open Error! Try again')
    else:
        r_image=yolo.detect_image(image)
        r_image.show()

FLAGS=None

if __name__ == '__main__':
    image = 1
    video = 0
    if (image):
        x = {'image': True, 'input': './path2your_video', 'output': ''}
        yolo = YOLO(**x)
        image = Image.open('new_pic.jpg')
        r_image = yolo.detect_image(image)
        r_image.save("C:\PlL.jpg")
        r_image.show()
    elif (video):
        x = {'image': False, 'input': 'demo.mp4', 'output': ''}
        yolo = YOLO(**x)
        detect_video(yolo, 'demo.mp4', '')
