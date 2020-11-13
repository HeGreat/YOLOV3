import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import os
import settings
from yolo3.model import yolo_body,yolo_loss,preprocess_true_boxes
from yolo3.utils import get_random_data
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)


def _main():
    annotation_path=settings.TRAIN_DATA_PATH

    log_dir=settings.LOGS_DIR
    classes_path=settings.DEFAULT_CLASSES_PATH
    anchors_path=settings.DEFAULT_ANCHORS_PATH
    class_names=get_classes(classes_path)
    num_classes=len(class_names)      #coco数据集共80种类型
    anchors=get_anchors(anchors_path)
    # print(f'class_names:{class_names},  anchors:{anchors}')

    input_shape=settings.MODEL_IMAGE_SIZE

    is_tiny_version=len(anchors)==6
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path=settings.PRE_TRAINING_TINY_YOLO_WEIGHTS)
    else:
        model=create_model(input_shape,anchors,num_classes,freeze_body=2,weights_path=settings.PRE_TRAINING_YOLO_WEIGHTS)
    model.summary()
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #分配训练集和验证集
    val_split=settings.VALID_SPLIT
    with open(annotation_path)  as f:
        lines=f.readlines()
    num_train=int(len(lines))*0.6
    num_train=np.floor(num_train).astype(int)
    num_val=len(lines)-num_train


    #Train with frozen layers first, to get a stable loss.
    #Adjust num epoches to your dataset, This step is enough to obtain a not bad model
    if settings.FROZEN_TRAIN:
        model.compile(optimizer=Adam(lr=settings.FROZEN_TRAIN_LR),loss={
            #use custom yolo_loss Lambda layer.
            'yolo_loss':lambda y_true,y_pred:y_pred})
        # batch_size=settings.FROZEN_TRAIN_BATCH_SIZE
        batch_size=2
        print(f'Train on {num_train} samples, val on {num_val}, with batch size: {batch_size}')
        model.fit_generator(data_generator_wrapper(lines[:num_train],batch_size,input_shape,anchors,num_classes),
                            steps_per_epoch=max(1,num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:],batch_size,input_shape,anchors,num_classes),
                            validation_steps=max(1,num_val//batch_size),epochs=50,initial_epoch=0,callbacks=[logging,checkpoint])

        model.save_weights(os.path.join(log_dir,settings.FROZEN_TRAIN_OUTPUT_WEIGHTS))

    #Unfreeze and continue training, to fin-tune.
    #Train longer if the result is not good.
    if settings.UNFREEZE_TRAIN:
        for i in range(len(model.layers)):
            model.layers[i].trainable=True
        model.compile(optimizer=Adam(lr=settings.UNFREEZE_TRAIN_LR),
                      loss={'yolo_loss':lambda y_true,y_pred:y_pred})
        print('Unfreeze all of the layers')
        batch_size=settings.UNFREEZE_TRAIN_BATCH_SIZE # note that more GPU memory is required after unfreezing the body
        print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}')
        model.fit_generator(data_generator_wrapper(Epochlines[:num_train],batch_size,input_shape,anchors,num_classes),
                            steps_per_epoch=max(1,num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:],batch_size,input_shape,anchors,num_classes),
                            validation_steps=max(1,num_val//batch_size),
                            epochs=100,initial_epoch=50,callbacks=[logging,checkpoint,reduce_lr,early_stopping])
        model.save_weights(os.path.join(log_dir,settings.UNFREEZE_TRAIN_OUTPUT_WEIGHTS))

    #Further training if needed.

    #Save final weights
    model.save_weights(os.path.join(log_dir,settings.FINAL_OUTPUT_WEIGHTS))


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names=f.readlines()
    class_names=[c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors=f.readline()
    anchors=[float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1,2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,weights_path='model_data/yolo.h5'):
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for l in range(3)]
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    # model_body.summary()
    print(f'model_body.layers:{len(model_body.layers)}')
    print(f'Create YOLOv3 model with {num_anchors} anchor and {num_classes} classes')

    if load_pretrained:
        # model_body.summary()
        model_body.load_weights(weights_path,by_name=True,skip_mismatch=True)
        print(f'Load weights {weights_path}')
        if freeze_body in [1,2]:   #freeze_body:1为冻结Darknet53的层，2为冻结全部，只保留最后3层
            '''Freeze darknet53 body or freeze all but 3 output layers'''
            num=(185,len(model_body.layers)-3)[freeze_body-1]   #选择list中第几个数字
            for i in range(num):
                model_body.layers[i].trainable=False     #最后三个 predict：freeze
                # print(f'Freeze the first {num} layers of total {len(model_body.layers)} layers')

    model_loss=Lambda(yolo_loss,output_shape=(1,),name='yolo_loss',arguments={'anchors':anchors,'num_classes':num_classes,
                                                                              'ignore_thresh':0.5})([*model_body.output,*y_true])
    #model_body.output:共有3个y1,y2,y3
    model=Model([model_body.input,*y_true],model_loss)

    print(f'model_loss:{model_loss}')
    return model


def data_generator(annotation_lines,batch_size,input_shape,anchors,num_classes):
    '''data generator for fit_generator'''
    n=len(annotation_lines)
    i=0
    while True:
        image_data=[]
        box_data=[]
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image,box=get_random_data(annotation_lines[i],input_shape,random=settings.IMAGE_AUGMENTATION)
            # image:resize后的图片，box:resize后的框的位置,没有归一化处理
            image_data.append(image)
            box_data.append(box)
            i=(i+1)%n
        image_data=np.array(image_data) #resize后的图片，经过了归一化处理
        box_data=np.array(box_data)  #box_data:在resize后的图中box的位置与大小（值没有归一化）
        #将box_data的数据进行归一化处理，然后通过y_true传递回来(3,box_num,grid_h,grid_w,3,85)
        y_true=preprocess_true_boxes(box_data,input_shape,anchors,num_classes)
        yield [image_data,*y_true],np.zeros(batch_size)

def data_generator_wrapper(annotation_lines,batch_size,input_shape,anchors,num_classes):
    n=len(annotation_lines)
    if n==0 or batch_size<=0 :return None
    y = data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)
    return y


if __name__ == '__main__':
    _main()