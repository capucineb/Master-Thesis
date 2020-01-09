all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Flatten
import keras.optimizers
#import keras.backend import flatten
from keras import backend as K
import numpy as np
import keras.backend.tensorflow_backend as K2
import tensorflow as tf
import matplotlib.pyplot as plt
from netwokmodels import my_model
from collections import Counter



# dimensions of our images.
img_width, img_height = 500, 320


#img_width, img_height = 800, 600
#train_data_dir ='/home/cbe/Documents/cbe/Data/cbr/echocardio/myframes_cropped/train'
train_data_dir = r'/home/cbe/Documents/cbe/Data/cbr/echocardio/4classes/train'  ####/mnt/Data/cbr/echocardio/myframes/train'
epochs = 50
batch_size = 1

if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)



def PrintDimensions (msg, x):
	print ("Log *********",msg )
	shape = K.int_shape(x)
	print( shape)
#	wait = input("Log message: PRESS ENTER generate TO CONTINUE.")


config = tf.ConfigProto()
# ask for GPU memory gracefully
config.gpu_options.allow_growth = True
# config.log_device_placement=True

# see only GPU 1
#config.gpu_options.visible_device_list = "0"  # 0 titan 1 GTX 1080
sess = tf.Session(config=config)

K2.set_session(sess)

input_shape = (img_width, img_height, 3)

input_image = Input(shape=input_shape, dtype = "float32")

x = my_model("transfert_ResNet50",input_image, input_shape,classes=4)

complete_model = Model(inputs=[input_image], outputs=[x])


#optimizer=keras.optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0) #lr=1e-4

#optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.0, decay=0.0, nesterov=False)

optimizer=keras.optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

complete_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
print(complete_model.summary())

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(validation_split=0.2)

# this is the augmentation configuration we will use for testing:
# only rescaling
#test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode = "rgb",#color_mode = "grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True)


validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",  # color_mode = "grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',shuffle=True)

nb_train_samples = int(np.ceil(train_generator.samples / batch_size))
nb_validation_samples = int(np.ceil(validation_generator.samples / batch_size))

counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

history=complete_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size),
 #   class_weight=class_weights)

complete_model.save_weights('CNN5 4classes ResNet transfert.h5')


# summarize history for accuracy
plt.figure(1)
plt.plot(history[0].history['acc'])
plt.plot(history[0].history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_CNN5  ResNet transfert 4 c.png')


# summarize history for loss
plt.figure(2)
plt.plot(history[0].history['loss'])
plt.plot(history[0].history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss CNN5  4 classes ResNet transfert.png')






