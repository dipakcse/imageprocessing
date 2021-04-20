import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, TqdmDeprecationWarning
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

# declare some parameters
epochs_size = 50
img_width = 128
img_height = 128
border = 5
max_pool_parm_1 = 2
max_pool_parm_2 = 3


def conv2d(in_tensor, n_filters, kernel_size=3, batchnorm=True):
    # 2 convolutional layers with the parameters passed to it
    # 1st layer
    conv = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                  kernel_initializer='he_normal', padding='same')(in_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    # 2nd layer
    conv = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                  kernel_initializer='he_normal', padding='same')(in_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    return conv


def unet_model(input_image, n_filters, dropout, batchnorm):
    # define an UNET Model here

    c1 = conv2d(input_image, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((max_pool_parm_1, max_pool_parm_1))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((max_pool_parm_1, max_pool_parm_1))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((max_pool_parm_1, max_pool_parm_1))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((max_pool_parm_1, max_pool_parm_1))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    u6 = Conv2DTranspose(n_filters * 8, (max_pool_parm_2, max_pool_parm_2), strides=(max_pool_parm_1, max_pool_parm_1),
                         padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (max_pool_parm_2, max_pool_parm_2), strides=(max_pool_parm_1, max_pool_parm_1),
                         padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (max_pool_parm_2, max_pool_parm_2), strides=(max_pool_parm_1, max_pool_parm_1),
                         padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (max_pool_parm_2, max_pool_parm_2), strides=(max_pool_parm_1, max_pool_parm_1),
                         padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    output = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    ret_model = Model(inputs=[input_image], outputs=[output])
    return ret_model


def plot_samp_data(x, y, pred, binary_pred, ix=None):
    if ix is None:
        ix = random.randint(0, len(x))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(x[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Salt')

    ax[2].imshow(pred[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Salt Predicted')

    ax[3].imshow(binary_pred[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Salt Predicted binary')
    plt.show()


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="Variables are collinear", category=UserWarning
    )
    warnings.filterwarnings("ignore", message="grayscale is deprecated. Please use ", category=UserWarning)
    warnings.filterwarnings("ignore", category=TqdmDeprecationWarning)


# get list of all images
images_id = next(os.walk("train/images"))[2]
print("Total number of images found= ", len(images_id))
x = np.zeros((len(images_id), img_height, img_width, 1), dtype=np.float32)
y = np.zeros((len(images_id), img_height, img_width, 1), dtype=np.float32)
for n, id_ in tqdm_notebook(enumerate(images_id), total=len(images_id)):
    # Load all images
    img = load_img("train/images/" + id_, color_mode="grayscale")
    x_img = img_to_array(img)
    x_img = resize(x_img, (img_height, img_width, 1), mode='constant', preserve_range=True)
    # Load each masks
    mask = img_to_array(load_img("train/masks/" + id_, color_mode="grayscale"))
    mask = resize(mask, (img_height, img_width, 1), mode='constant', preserve_range=True)
    x[n] = x_img / 255.0
    y[n] = mask / 255.0

# Split train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

input_img = Input((img_height, img_width, 1), name='img')
model = unet_model(input_img, n_filters=16, dropout=0.1, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
# model.summary()

# model checkpoint call function to save best result
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
# fit the model
results = model.fit(x_train, y_train, batch_size=32, epochs=epochs_size, validation_data=(x_test, y_test),
                    callbacks=callbacks)

# loading the best model
model.load_weights('model-tgs-salt.h5')
score = model.evaluate(x_test, y_test, verbose=1)

print("Test loss = ", score[0])
print("Test Accuracy = ", score[1])

predic_train = model.predict(x_train, verbose=1)
predic_test = model.predict(x_test, verbose=1)
predic_train_t = (predic_train > 0.5).astype(np.uint8)
#pred_test_t = (predic_test > 0.5).astype(np.uint8)
plot_samp_data(x_train, y_train, predic_train, predic_train_t, ix=14)


