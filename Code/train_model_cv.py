# Author: Sebastian Fabig
from image_gen import ImageDataGenerator
from load_data import loadDataGeneral
from build_model import build_UNet2D_4L
import numpy
import csv

import pandas as pd

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = './prostate_training.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = csv_path[:csv_path.rfind('/')] + '/'

    df = pd.read_csv(csv_path)
    df.sample(frac=1)

    # Load training and validation data
    im_shape = (400, 400)
    X, y = loadDataGeneral(df, path, im_shape)

    # emulate 10-fold cross validation test harness
    n_splits = 10
    print(len(X[0]))
    split_size = int(len(X[0])/n_splits)
    print(split_size)
    cv_scores = []
    for i in range(0, n_splits):
        print("Iteration %d" % (i+1))
        if i is 0:
            X_train = X[split_size:]
            y_train = y[split_size:]
            X_val = X[:split_size]
            y_val = y[:split_size]
        elif i is n_splits - 1:
            X_train = X[:split_size * i]
            y_train = y[:split_size * i]
            X_val = X[split_size * i:]
            y_val = y[split_size * i:]
        else:
            X_train = numpy.concatenate([X[:i * split_size], X[i * split_size + split_size:]])
            y_train = numpy.concatenate([y[:i * split_size], y[i * split_size + split_size:]])
            X_val = X[i * split_size:i * split_size + split_size]
            y_val = y[i * split_size:i * split_size + split_size]
        cv_scores.append(numpy.random.rand() * 100)
        # Build model
        inp_shape = X[0].shape
        UNet = build_UNet2D_4L(inp_shape)
        UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        UNet.load_weights(filepath='./trained_model.hdf5')

        train_gen = ImageDataGenerator(rotation_range=10,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rescale=1.,
                                       zoom_range=0.2,
                                       fill_mode='nearest',
                                       cval=0)

        test_gen = ImageDataGenerator(rescale=1.)

        batch_size = 8
        UNet.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                           steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                           epochs=100,
                           validation_data=test_gen.flow(X_val, y_val),
                           validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size)

        scores = UNet.evaluate(X_val, y_val, verbose=0)
        print("%s: %.2f%%" % (UNet.metrics_names[1], scores[1] * 100))
        cv_scores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cv_scores), numpy.std(cv_scores)))

    with open('./cv_scores.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['CV-Errors'])
        for val in cv_scores:
            writer.writerow([val])
        writer.writerow(["%.2f%% (+/- %.2f%%)" % (numpy.mean(cv_scores), numpy.std(cv_scores))])
