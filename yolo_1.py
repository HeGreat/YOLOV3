import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image,ImageFont,ImageDraw
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model

import settings
from yolo3.model import yolo_eval,yolo_body,tiny_yolo_body
from yolo3.utils_1 import letterbox_image
import cv2


class YOLO(object):
    _defaults={
        'model_path':settings.DEFAULT_MODEL_PATH,
        'anchors_path':settings.DEFAULT_ANCHORS_PATH,
        'classes_path':settings.DEFAULT_CLASSES_PATH,
        'score':settings.SCORE,
        'iou':settings.IOU,
        'model_image_size':settings.MODEL_IMAGE_SIZE,
        'gpu_num':settings.GPU_NUM,
    }
    @classmethod
    def get_defaults(cls,n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return 'Unrecognized attribute name'+n+"'"

    def __init__(self,**kwargs):
        self.__dict__.update(self._defaults)  #set up default values
        self.__dict__.update(kwargs)          # and update with user overrides
        self.class_names=self._get_class()
        self.anchors=self._get_anchors()
        self.load_yolo_model()

    def _get_class(self):
        classes_path=os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names=f.readlines()
        class_names=[c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path=os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors=f.readline()
        print(f'type(anchors):{type(anchors)}')
        anchors=[float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1,2)

    def load_yolo_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file'

        # Load model, or construct model and load weights
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors / 3, num_classes)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(self.yolo_model.output) * (
                        num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print(f'{model_path:}model,anchors,and classes loaded')

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [( 1.0, 1.0, x / len(self.class_names)) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)  # Fixed sedd for consistent colors across runs
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default

    # @tf.function
    def compute_output(self,image_data,image_shape):  #image_data:类型为：numpy，float32,image_shape:(1080,1920)
        #Generae output tensor targets for filtered bounding boxes
        #self.input_image_shape=K.placeholder(shape=(2,))
        self.input_image_shape=tf.constant(image_shape)
        if self.gpu_num>=2:
            self.yolo_model=multi_gpu_model(self.yolo_model,gpus=self.gpu_num)
        time1=timer()
        boxes,scores,classes=yolo_eval(self.yolo_model(image_data),self.anchors,len(self.class_names),self.input_image_shape,
                                       score_threshold=self.score,iou_threshold=self.iou)
        time2=timer()
        print(f'yolo_eval函数外部耗费时间：{time2-time1}')
        return boxes,scores,classes


    def detect_image(self, image):  #image是没有经过缩放的Mat格式
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of re required'
            #检查是否改变了原来的image图像
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            time4 = timer()
            print(f'letterbox_image函数外部耗费时间：{time4 - start}')
        # else:
        #     new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
        #     boxed_image = letterbox_image(image, new_image_size)

        image_data=np.array(boxed_image,dtype='float32')
        image_data/=255.0
        image_data=np.expand_dims(image_data,0) #Add batch dimension
        time2 = timer()
        print(f'image_detect函数start到compute_output之前耗费时间:{time2-start}')
        out_boxes,out_scores,out_classes=self.compute_output(image_data,[image.shape[0],image.shape[1]])
        time3=timer()
        print(f'image_detect函数compute_output函数外部耗费时间：{time3-time2}')
        # print(f'Found {len(out_boxes)} boxes for img')

        thickness=(image.shape[0]+image.shape[1])//300  #表示画框线的厚度
        time1 = timer()
        for i,c in reversed(list(enumerate(out_classes))):
            print(f'i:{i}')
            predicted_class=self.class_names[c]
            box=out_boxes[i]
            score=out_scores[i]

            text='{} {:.2f}'.format(predicted_class,score)
            top, left, bottom, right = box
            font=cv2.FONT_HERSHEY_SIMPLEX
            (w, h), x = cv2.getTextSize(text, font, 0.5, 1)

            top=max(0,np.floor(top+0.5).astype('int32'))
            left=max(0,np.floor(left+0.5).astype('int32'))
            bottom=min(image.shape[0],np.floor(bottom+0.5).astype('int32'))
            right=min(image.shape[1],np.floor(right+0.5).astype('int32'))
            for i in range(3):
                image=cv2.rectangle(image,(left+i,top+i),(right-i,bottom-i),self.colors[c][::-1],1)
            # image=cv2.rectangle(image,(left,top-h-x),(left+w,top),self.colors[c],-1)
            image=cv2.putText(image,text,(left,top-x),font,0.5,(0,0,255),1)
        end=timer()
        print('dectect_image内部后处理时间：',end-time1)
        return image

def detect_video(yolo,video_path,output_path=""):
    import cv2
    vid=cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC=int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps=vid.get(cv2.CAP_PROP_FPS)
    video_size=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput=True if output_path!="" else False
    if isOutput:
        print("!!! TYPE", type(output_path),type(video_FourCC),type(video_fps),type(video_size))
        out=cv2.VideoWriter(output_path,video_FourCC,video_fps,video_size)
    accum_time=0
    curr_fps=0
    fps="FPS:??"
    prev_time=timer()
    while True:
        time1=timer()
        return_value,frame=vid.read()
        # image=Image.fromarray(frame)
        time3 = timer()
        image=yolo.detect_image(frame)
        # image = yolo.detect_image(frame)
        time4=timer()
        print(f'dectect_image外部时间: {time4-time3}')
        result=np.asarray(image)  #是Mat格式
        curr_time=timer()
        exec_time=curr_time-prev_time
        prev_time=curr_time
        accum_time=accum_time+exec_time
        curr_fps=curr_fps+1
        if accum_time>1:
            accum_time=accum_time-1
            fps='FPS:'+str(curr_fps)
            curr_fps=0
        cv2.putText(result,text=fps,org=(3,15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.50,color=(255,0,0),thickness=2)
        cv2.namedWindow('result',cv2.WINDOW_NORMAL)
        cv2.imshow('result',result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time2 = timer()
        print(f'video detect one image time:{time2 - time1}')


