# generates audio and sprectrum image (runs inference) from trained model converted to .onnx
# for model trained on 128x256 px spectrograms

import numpy as np
import time
from PIL import Image
import mmm_WaveToolbox as wtb
import onnxruntime as ort

print("Starting TechnoGAN generator server...")

# location of generator model
GENERATORSAVELOCATION = "data/model22.onnx"
print("Loading generator model: " + GENERATORSAVELOCATION)
sess = ort.InferenceSession(GENERATORSAVELOCATION, providers=["CPUExecutionProvider"])

# dimensions of generated images (needs to be same as generator output layer)
IMG_WIDTH = 128
IMG_HEIGHT = 256

# run inference on the generator model
# input/output names are defined when converting the tf model to onnx
# dimensons of input and output arrays are a bit wonky, not sure why additional dimensions are added at some point.
# TODO: figure out how to and implement proper naming of input and output layers on model conversion to .onnx
def generate(seed):
    #bring seed in correct input shape (1,64)
    input1 = np.zeros((1, 64), np.float32)
    input1[0, :] = seed

    #output should be shaped like (1, 128, 256, 1)
    #maybe first dim has to be cut off by adding [0]?,
    r = sess.run(["conv2d_transpose_3"], {"x": input1})[0]
    return r


# generate a dummy to test and initialize the network
# for some reason the first generation always takes a bit longer...
def generate_init_dummy(seedlength):
    dummyseed = np.zeros(seedlength)
    src_spect = generate(dummyseed)

# generate a single sample from given seed array (64 ints)
def generate_single_wave(seed, save_img=False):
    # take starting time
    st = time.time()

    print('generating single')

    # cast input seed to correct datatype and array dimension
    tensorseed = np.zeros((1, len(seed)), dtype='float32')
    tensorseed[0, :] = seed
    # roughly scale seed from signed int [-127,127] to float [-3.0,3.0]
    tensorseed = tensorseed * 3 / 128.

    # run inference
    src_spect = generate(tensorseed)

    # create proper 2-dimensional array
    values = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    # extract needed dimensions from output array and scale from [-1.,1.] to [0.,1.]
    values[:, :] = src_spect[0, :, :, 0] / 2 + 0.5

    # scale and cast values to int (from [0.,1.] to [0,256])
    values *= 256
    values = values.astype('int8')

    # create actual image from int array
    im = Image.fromarray(values, 'L')

    # recreate waveform from image
    destwave, wflags = wtb.powergram_to_wave(im=im, hops=4, spectpow=3)

    # print measured timespan to console for debug
    timestring = f'{"{:0.2f}".format(time.time() - st)} sec.'
    print(f'time: {timestring}')

    # return recreated waveform, generated image, timespan
    return destwave, im, timestring


# creates a chain of samples by linear interpolating each value of a start- and an end-seed.
# number of samples is defined by steps
# samples are appended and stored to a single wavefile
def generate_fade(startseed, endseed, steps):
    st = time.time()
    print('generating fade')
    tensorstartseed = np.zeros((1, len(startseed)), dtype='float32')
    tensorendseed = np.zeros((1, len(endseed)), dtype='float32')
    tensorstartseed[0, :] = startseed
    tensorendseed[0, :] = endseed
    tensorstartseed = tensorstartseed * 3 / 128.
    tensorendseed = tensorendseed * 3 / 128.

    values = np.zeros((IMG_HEIGHT, IMG_WIDTH))

    destwave = np.array(0, dtype='int16')

    for i in range(steps):
        tensorseed = wtb.lerp(tensorstartseed, tensorendseed, i / (steps - 1))
        src_spect = generate(tensorseed)
        values[:, :] = src_spect[0, :, :, 0] / 2 + 0.5
        values *= 256
        intvalues = values.astype('int8')
        im = Image.fromarray(intvalues, 'L')
        destwavestep, wflags = wtb.powergram_to_wave(im=im, hops=4, spectpow=3)
        destwave = np.append(destwave, destwavestep)

    timestring = f'{"{:0.2f}".format(time.time() - st)} sec.'
    print(f'time: {timestring}')
    return destwave, timestring

# creates a chain of samples by linear interpolating each value of a start- to an end-seed and back to the start.
# number of samples is defined by steps
# samples are appended and stored to a single wavefile
def generate_pingpong(startseed, endseed, steps):
    st = time.time()
    print('generating pingpong')
    tensorstartseed = np.zeros((1, len(startseed)), dtype='float32')
    tensorendseed = np.zeros((1, len(endseed)), dtype='float32')
    tensorstartseed[0, :] = startseed
    tensorendseed[0, :] = endseed
    tensorstartseed = tensorstartseed * 3 / 128.
    tensorendseed = tensorendseed * 3 / 128.

    values = np.zeros((IMG_HEIGHT, IMG_WIDTH))

    destwave = np.array(0, dtype='int16')

    i = 0
    for k in range(steps):
        if (k <= steps / 2):
            i = k
        else:
            i -= 1
        tensorseed = wtb.lerp(tensorstartseed, tensorendseed, i * 2 / (steps - 1))
        src_spect = generate(tensorseed)
        values[:, :] = src_spect[0, :, :, 0] / 2 + 0.5
        values *= 256
        intvalues = values.astype('int8')
        im = Image.fromarray(intvalues, 'L')
        destwavestep, wflags = wtb.powergram_to_wave(im=im, hops=4, spectpow=3)
        destwave = np.append(destwave, destwavestep)

    timestring = f'{"{:0.2f}".format(time.time() - st)} sec.'
    print(f'time: {timestring}')
    return destwave, timestring

# called on startup
def init_generator():
    print("Initializing generator...")
    generate_init_dummy(64)
    print("Generator initialized.")
