
import numpy as np
import tensorflow as tf

# Update path to local installation of mlpf library
import sys
sys.path += ["../mlpf"]

import tfmodel
from tfmodel.model import PFNetDense

# Training parameters
num_epochs = 100
num_batches = 5

# Dummy function for loading atlas data; to be implemented later
def load_data():
    pass

def transform_target(y):
    return {
        "cls": tf.one_hot(tf.cast(y[:, :, 0], tf.int32), num_output_classes),
        "charge": y[:, :, 1:2],
        "pt": y[:, :, 2:3],
        "eta": y[:, :, 3:4],
        "sin_phi": y[:, :, 4:5],
        "cos_phi": y[:, :, 5:6],
        "energy": y[:, :, 6:7],
    }

# Load data from pipeline
data = load_data()

input_classes = np.unique(data["X"][:, :, 0].flatten())
output_classes = np.unique(data["Y"][:, :, 0].flatten())
num_output_classes = len(output_classes)

combined_graph_layer = {
    "max_num_bins": 100,
    "bin_size": 128,
    "distance_dim": 128,
    "layernorm": "no",
    "num_node_messages": 1,
    "dropout": 0.0,
    "kernel": {
        "type": "NodePairGaussianKernel",
        "dist_mult": 0.1,
        "clip_value_low": 0.0
    },
    "node_message": {
        "type": "GHConvDense",
        "output_dim": 128,
        "activation": "gelu",
        "normalize_degrees": "yes"
    },
    "ffn_dist_hidden_dim": 128,
    "do_lsh": "no",
    "ffn_dist_num_layers": 0,
    "activation": "gelu",
    "dist_activation": "gelu"
}

model = PFNetDense(
    num_input_classes=len(input_classes),
    num_output_classes=len(output_classes),
    activation="elu",
    hidden_dim=128,
    bin_size=128,
    input_encoding="default",
    multi_output=True,
    combined_graph_layer=combined_graph_layer
)

# Temporal weight mode means each input element in the event can get a separate weight
model.compile(
    loss={
        "cls": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        "charge": tf.keras.losses.MeanSquaredError(),
        "pt": tf.keras.losses.MeanSquaredError(),
        "energy": tf.keras.losses.MeanSquaredError(),
        "eta": tf.keras.losses.MeanSquaredError(),
        "sin_phi": tf.keras.losses.MeanSquaredError(),
        "cos_phi": tf.keras.losses.MeanSquaredError()
    },
    optimizer="adam",
    sample_weight_mode="temporal"
)

# Train the model
model.fit(data["X"], transform_target(data["Y"]), epochs=num_epochs, batch_size=num_batches)