### Transfer Learning of the whole MobileNet model
### InceptionV3
### Dropouts
### Feature Extraction
### The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.
 
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
  layer.trainable = False
pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
