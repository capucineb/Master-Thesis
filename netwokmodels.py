from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Dense, Flatten
import keras.optimizers
#import keras.backend import flatten
from keras import backend as K
import numpy as np
import keras.backend.tensorflow_backend as K2
import tensorflow as tf
import matplotlib.pyplot as plt


def my_model(name,input_image, input_shape, classes=2):

    def PrintDimensions(msg, x):
        print("Log *********", msg)
        shape = K.int_shape(x)
        print(shape)
        

    def model_5CNN(input_image,input_shape,classes):
        #"-------------MODEL--CNN----------------"
        x=Conv2D(32, (3, 3), input_shape=input_shape)(input_image)
        x=Activation('relu')(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)

        x=Conv2D(32, (3, 3))(x) #64 for full images 32 for cropped one
        x=Activation('relu')(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)

        x=Conv2D(64, (3, 3))(x)
        x=Activation('relu')(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3))(x)               
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3))(x)    
        #NEW  x = BatchNormalization(epsilon=2e-4, momentum=0.9)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        if classes==2:
            x = Dense(1,activation='sigmoid')(x) #4classes 4neurons
        else:
            x = Dense(classes, activation='softmax')(x)  # 4classes 4neurons

        return (x)

    def model_3CNN(input_image, input_shape, classes):
            # "-------------MODEL--CNN----------------"
        x = Conv2D(32, (3, 3), input_shape=input_shape)(input_image)
 #       x = BatchNormalization(axis=1, epsilon=2e-4, momentum=0.9)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3))(x)  # x=Conv2D(32, (3, 3))(x)  64 for full images 32 for cropped one
 #       x = BatchNormalization(axis=1, epsilon=2e-4, momentum=0.9)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3))(x)
 #       x = BatchNormalization(axis=1, epsilon=2e-4, momentum=0.9)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)


        x = Flatten()(x)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        if classes == 2:
            x = Dense(1, activation='sigmoid')(x)  # 4classes 4neurons
        else:
            x = Dense(classes, activation='softmax')(x)  # 4classes 4neurons


        return (x)

    def model_ResNet50_transfert(input_image, input_shape, classes):
    #"-------------MODEL--RESNET----------------"

        res=ResNet50(include_top=False, weights="imagenet", input_tensor=input_image, input_shape=input_shape, pooling=None, classes=classes)#classes=2)
        x = res.output
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        if classes == 2:
            x = Dense(1, activation='sigmoid')(x)  # 4classes 4neurons
        else:
            x = Dense(classes, activation='softmax')(x)  # 4classes 4neurons

        return x


    def model_ResNet50(input_image, input_shape, classes):
    #"-------------MODEL--RESNET----------------"

        res=ResNet50(include_top=False, weights=None, input_tensor=input_image, input_shape=input_shape, pooling=None, classes=classes)#classes=2)
        x = res.output
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        if classes == 2:
            x = Dense(1, activation='sigmoid')(x)  # 4classes 4neurons
        else:
            x = Dense(classes, activation='softmax')(x)  # 4classes 4neurons

        return x


    def model_VGG19(input_image, input_shape, classes):
    #"-------------MODEL--VGG----------------"
        vgg = VGG19(include_top=False, weights=None, input_tensor=input_image, input_shape=input_shape, pooling=None, classes=classes)#classes=2)
        x = vgg.output
        x = Flatten()(x)
    	#PrintDimensions("after flatten",x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return(x)


    def model_myVGG16(input_image, input_shape, classes):
      
      	def blocVGG(input_im,depth, thirdConv=True, input_shape = None):
          	if input_shape is not None:
        		x = Conv2D(depth, (3,3) ,activation='relu',
                       input_shape=input_shape)(input)
            else: 
              x= Conv2D(depth, (3,3), activation = 'relu')(input_im)
              
       		x = BatchNormalization(epsilon=2e-4, momentum=0.9)(x)
        	x = Conv2D(depth, (3,3) ,activation='relu')(x)
        	x = BatchNormalization(epsilon=2e-4, momentum=0.9)(x)
            
            if thirdConv:
              x = Conv2D(depth, (1,1), activation = 'relu')(x)
          
        	x = MaxPooling2D(pool_size=(2, 2))(x)
          	return (x)
         
        
        x= blocVGG(input_im = input_image, input_shape = input_shape, depth =32, 						thirdConv=False )
        x= blocVGG(input_im = x, depth = 64, thirdConv=False )
        x= blocVGG(input_im = x, depth = 128)
        x= blocVGG(input_im = x, depth = 256
        x= blocVGG(input_im = x, depth = 512)
        

        x = Flatten()(x)
        x = Dense(128,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128,activation='relu')(x)
        x = Dropout(0.2)(x)
        if classes==2:
            x = Dense(1)(x)
        else:
            x=Dense(classes)(x)

        x = Activation('sigmoid')(x)

        return (x)





    if name =="5CNN":
        x=model_5CNN(input_image,input_shape,classes)

    elif name =="3CNN":
        x=model_3CNN(input_image,input_shape,classes)

    elif name =="ResNet50":
        x = model_ResNet50(input_image, input_shape,classes)

    elif name == "VGG19":
        x = model_VGG19(input_image, input_shape, classes)

    elif name == "myVGG16":
        x = model_myVGG16(input_image, input_shape, classes)

    elif name == "transfert_ResNet50":
        x = model_ResNet50_transfert(input_image, input_shape, classes)

    return(x)