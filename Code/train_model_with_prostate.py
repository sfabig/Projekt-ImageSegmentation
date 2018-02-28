# Author: Sebastian Fabig
from image_gen import ImageDataGenerator
from load_data import loadDataGeneral
from build_model import build_UNet2D_4L

import pandas as pd
from keras.utils.vis_utils import plot_model
from keras.models import load_model

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = './prostate_training.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = csv_path[:csv_path.rfind('/')] + '/'

    df = pd.read_csv(csv_path)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = df.sample(frac=1)

    # Load training and validation data
    im_shape = (400, 400)
    X_train, y_train = loadDataGeneral(df, path, im_shape)

    # Build model
    layers_to_keep = ['conv2d_1', 'conv2d_10', 'conv2d_11', 'conv2d_12', 'conv2d_13', 'conv2d_22']
    inp_shape = X_train[0].shape
    lung_net = load_model('./trained_model.hdf5')
    UNet = build_UNet2D_4L(inp_shape)
    for name in layers_to_keep:
        UNet.get_layer(name=name).set_weights(lung_net.get_layer(name=name).get_weights())
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Visualize model
    plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

    batch_size = 8
    UNet.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                       epochs=100)

    UNet.save('./trained_prostate_model_some_layer.hdf5')
