# Author: Sebastian Fabig
import numpy
import csv
from sklearn.metrics import mean_absolute_error
from keras .models import load_model

UNet_original = load_model('./trained_model.hdf5')
UNet_prostate = load_model('./trained_prostate_model_some_layer.hdf5')
layer = [UNet_original.layers[i] for i in range(len(UNet_original.layers))]
maes = [float(-1) for i in range(len(UNet_original.layers))]

for i in range(len(UNet_original.layers)):
    print(UNet_original.layers[i])
    print(UNet_prostate.layers[i])
    layer_weights_orig = UNet_original.get_layer(index=i).get_weights()
    layer_weights_prostate = UNet_prostate.get_layer(index=i).get_weights()
    if len(layer_weights_orig) > 0 and len(layer_weights_prostate) > 0:
        weights_orig = [float(weight) for weight in numpy.nditer(layer_weights_orig[0])]
        weights_prostate = [float(weight) for weight in numpy.nditer(layer_weights_prostate[0])]
        maes[i] = mean_absolute_error(weights_orig, weights_prostate)
        print(maes[i])
    
with open('./weights_comparison_some_layer.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Layer','MAE'])
    for i in range(len(maes)):
        writer.writerow([layer[i].name,maes[i]])