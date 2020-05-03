!pip install tensorflow-gpu==2.0.0
!pip install tqdm

!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O ./cats_and_dogs_filtered.zip

# Commented out IPython magic to ensure Python compatibility.
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %matplotlib inline

dataset_path="./cats_and_dogs_filtered.zip"
zip_object= zipfile.ZipFile(file=dataset_path,mode="r")
zip_object.extractall("./")
zip_object.close()

dataset_path_new="./cats_and_dogs_filtered/"
train_dir = os.path.join(dataset_path_new,"train")
validation_dir = os.path.join(dataset_path_new,"validation")

# Load pre trained model MobileNetV2
IMG_SHAPE=(128,128,3) # 128x128 colored images
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

#base_model.summary()

#Freezing base model
base_model.trainable=False

#analyse output size to determine the number of layers

#base_model.output 
#<tf.Tensor 'out_relu/Identity:0' shape=(None, 4, 4, 1280) dtype=float32>

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer = tf.keras.layers.Dense(units=1,activation="sigmoid")(global_average_layer)

model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),loss="binary_crossentropy",metrics=["accuracy"])
# changing the learning to be slower than default

data_gen_train=ImageDataGenerator(rescale=1/255.0)
data_gen_valid=ImageDataGenerator(rescale=1/255.0)

train_generator=data_gen_train.flow_from_directory(train_dir, target_size=(128,128),batch_size=128, class_mode="binary")# train
valid_generator=data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128),batch_size=128, class_mode="binary")# validation

model.fit(train_generator,epochs=5,validation_data=valid_generator) #using fit_generator() because of DataGenerator instead of fit()

valid_loss, valid_accuracy = model.evaluate(valid_generator)
print("Accuracy after transfer learning:{}".format(valid_generator))

# fine tuning: should be performed after training the custom head 
#fine tuning is done only on the few top layers
#steps:
base_model.trainable=True #unfreeze
print("number of layers in base_model: {}".format(len(base_model.layers)))
#155 layers
fine_tune_at = 100 #fine tune from 100-155
for layer in base_model.layers[:fine_tune_at]: #free layers before 100
  layer.trainable = False

#compiling the fine tuned model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),loss="binary_crossentropy",metrics=["accuracy"])

model.fit(train_generator,epochs=5,validation_data=valid_generator)

valid_loss, valid_accuracy = model.evaluate(valid_generator)
print("Accuracy after transfer learning:{}".format(valid_generator))