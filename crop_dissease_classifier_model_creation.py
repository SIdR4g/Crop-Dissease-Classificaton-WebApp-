import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.models import Model
import pickle

tfds = tf.data.Dataset
AUTOTUNE = tf.data.AUTOTUNE

def plot_image(batch):
    plt.figure(figsize=(20, 20))
    plt.rcParams.update({'font.size': 22})
    for image_batch, labels_batch in batch:
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(classes[labels_batch[i]])
            plt.axis("off")
    plt.show()


# @tf.autograph.experimental.do_not_convert
def get_label(file_path):
    return table.lookup(tf.strings.split(file_path, os.path.sep)[-2])

def process_image(file_path):
    
    label = get_label(file_path)
    
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [IMAGE_SIZE,IMAGE_SIZE])
    
    return img,label

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=10000):
    
    assert (train_split + val_split) == 1
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

if __name__ = '__main__':

    fruits = ["Apple", "Cherry", "Corn", "Grape", "Peach", "Potato", "Strawberry"]

    for fruit in fruits:
        path_train = 'train/'+fruit+'/*/*'
        path_test = 'test/'+fruit+'/*/*'
        
        train_images = tfds.list_files(path_train, shuffle = False)
        test_images = tfds.list_files(path_test, shuffle = False)
        
        train_images = train_images.shuffle(100)
        test_images = test_images.shuffle(100)
        

        classes = list(set(str(x.split("/")[-2]) for i,x in enumerate(glob(path_train, recursive = True))))
    
        with open(fruit+'_class.pkl', 'wb') as file:
            pickle.dump(classes, file)
            
        keys_tensor = tf.constant(classes)
        vals_tensor = tf.constant(range(len(classes)))

        init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)

        table = tf.lookup.StaticHashTable(
            init,
            default_value=-1)

        # print(keys_tensor, vals_tensor)

        IMAGE_SIZE = 128
        BATCH_SIZE = 64

        train_batch = train_images.map(process_image).batch(BATCH_SIZE)
        test_batch = test_images.map(process_image).batch(BATCH_SIZE)

        plot_image(train_batch.take(1))
        
        train_ds, val_ds = get_dataset_partitions_tf(train_batch)

        train_ds = train_ds.cache().prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)
        
        test_ds = test_batch.cache().prefetch(AUTOTUNE)

        
        data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2),
            ])

        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(buffer_size=AUTOTUNE)
        
        input_shape = ( IMAGE_SIZE, IMAGE_SIZE,3)# CHANNELS)
        n_classes = 4

        
        base_model = tf.keras.applications.MobileNetV2(input_shape = input_shape, include_top = False, weights = "imagenet")
        base_model.trainable = False
        model = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255),
                                base_model,
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(128, activation = 'relu'),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(n_classes, activation="softmax")                                     
                                ])        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=fruit+'.h5',
                monitor='val_loss', save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,verbose=1)
        ]

        history = model.fit(
            train_ds,
            batch_size=BATCH_SIZE,
            validation_data=val_ds,
            callbacks=callbacks_list,
            verbose=1,
            epochs=25,
        )


        from keras.models import load_model

        reconstructed_model = load_model(fruit+".h5")
        reconstructed_model.evaluate(test_ds)