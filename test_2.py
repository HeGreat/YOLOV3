from functools import reduce
from PIL import Image
from matplotlib.colors import  rgb_to_hsv,hsv_to_rgb
import numpy as np
import cv2
from timeit import default_timer as timer
import numpy

image=Image.open('demo.jpg')
cv = cv2.imread('demo.jpg')
# print(cv)


# PIL
def letterbox_image(image,size):
    '''resize image with unchanged aspect ratio using padding'''
    start=timer()
    iw,ih=image.size   #1920,1080
    w,h=size           #416,416
    scale=min(w/iw,w/ih)
    nw=int(iw*scale)
    nh=int(ih*scale)
    image=image.resize((nw,nh),Image.BICUBIC)
    new_image = Image.new('RGB', size, (255, 255, 255))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    # top = int((h - nh) // 2)
    # bottom = top
    # left = int((w - nw) // 2)
    # right = left
    # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[128, 128, 128])

    print(f'cv2.resize耗费时间：{timer() - start}')
    return new_image


# opencv
def letterbox_image2(image,size):
    '''resize image with unchanged aspect ratio using padding'''
    start = timer()
    ih, iw, _ = image.shape  #(1080,1920,3)
    w, h = size              #(416,416)
    scale = min(w / iw, w / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh),interpolation=cv2.INTER_CUBIC)     #

    # image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    # new_image = Image.new('RGB', size, (128, 128, 128))
    # new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    top=int((h - nh) // 2)
    bottom=top
    left=int((w - nw) // 2)
    right=left
    #检查一下是否改变了原来的image图像
    image = cv2.copyMakeBorder(image, top, bottom, left, right,cv2.BORDER_CONSTANT,value=[255,255,255])
    print(f'cv2.resize耗费时间：{timer() - start}')
    return image

# cv=letterbox_image2(cv,(416,416))
# image=letterbox_image(image,(416,416))
# print(image)

# cv2.imwrite('C:\opencv.jpg',cv)
# cv=np.array(cv)
# image=np.array(image)
# x=cv-image
# print(cv-image)
# print((x==0).all())
# h,w,_=x.shape
# print(h,w)
# for i in range(h):
#     for j in range(w):
#         if((cv[i,j]!=image[i,j]).all()):
#             print("------",i)

time1=timer()
resized0=cv2.resize(cv,(416,234),interpolation=cv2.INTER_AREA)  #INTER_AREA

# image=Image.open('demo.jpg')
# image=image.resize((416,234),Image.BICUBIC)
# resized0=np.array(image)

gray0=numpy.zeros((416,416,3),dtype=np.uint8)
gray0[...]=128
# gray255=gray0[:,:]
#将灰度图转换成彩色图
# Img_rgb=cv2.cvtColor(gray255,cv2.COLOR_GRAY2RGB)
#将RGB通道全部置成0
# Img_rgb[:,:,0:3]=0
# resized1=Img_rgb

resized1=gray0

resized1[91:91+234,0:416]=resized0
time2=timer()
# print(time2-time1)

# resized1 = Image.fromarray(cv2.cvtColor(resized1,cv2.COLOR_BGR2RGB))
# resized1.save('new_image.jpg')
# resized1.show()

cv2.imwrite('new_image.jpg',resized1)
cv2.imshow('resized1',resized1)
cv2.waitKey(0)

