# convert tf model to .onnx

import tensorflow as tf
import tf2onnx
import onnx
import keras

# directory of generated tf model
GENERATORSAVELOCATION = "D:/work/2022/TechnoGAN/networks/Spect_512x256_v2/"
generator = keras.models.load_model(GENERATORSAVELOCATION + "generator", compile=False)

# not fully understanding what happens here, the 'x' names the layer(?) that takes the input values for inference
input_signature = [tf.TensorSpec([1, 64], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(generator, input_signature, opset=13)

# destination file
onnx.save(onnx_model, "D:/work/2022/TechnoGAN/networks/Spect_512x256_v2/onnx/generator_512x256_v2.onnx")