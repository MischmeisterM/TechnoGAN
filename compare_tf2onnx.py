# compare inference output of original tf model and converted onnx model to
# verify conversion and functionality

import onnxruntime as ort
import numpy as np

import tensorflow as tf

# Change shapes and types to match model
dummyseed = np.zeros(64)
tensorseed = np.zeros((1, len(dummyseed)), dtype='float32')
# tensorseed[0, :] = tf.convert_to_tensor(dummyseed)
input1 = np.zeros((1, 64), np.float32)
input2 = np.random.random_sample((1, 64))
input2 = np.float32(input2)

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# Following code assumes NVIDIA GPU is available, you can specify other execution providers or don't include providers parameter
# to use default CPU provider.

sess = ort.InferenceSession("D:/work/2022/TechnoGAN/networks/Spect_256x256/onnx/generator_256x256.onnx", providers=["CUDAExecutionProvider"])

# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.

results_ort = sess.run(["conv2d_transpose_3"], {"x": input2})[0]

model = tf.saved_model.load("D:/work/2022/TechnoGAN/networks/Spect_256x256/generator")
results_tf = model(input2, training=False)

for ort_res, tf_res in zip(results_ort, results_tf):
    np.testing.assert_allclose(ort_res, tf_res, rtol=1e-5, atol=1e-5)

print("Results match")
