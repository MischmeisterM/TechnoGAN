# generates audio and sprectrum image (runs inference) from trained model converted to .onnx
# for model trained on 256x256 px constant-q spectrograms

# for more comments on functionality refer to mmm_onnxInference_128x256.py

import numpy as np
import time
from PIL import Image
import mmm_WaveToolbox as wtb
import onnxruntime as ort

print("Starting TechnoGAN generator server...")

GENERATORSAVELOCATION = "data/generator_256x256_cq2.onnx"
print("Loading generator model: " + GENERATORSAVELOCATION)
sess = ort.InferenceSession(GENERATORSAVELOCATION, providers=["CPUExecutionProvider"])

IMG_WIDTH = 256
IMG_HEIGHT = 256
#PREVIEWSCALE = 2
OUTPUT_SR = 44100

# number of frequency bins and starting frequency
# should reflect the numbers of the used training data
CQ_BINS = 24
CQ_BASE = 8.175

# run inference on the generator model
# input/output names are defined when converting the tf model to onnx
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


def generate_single_wave(seed, save_img=False):
    st = time.time()
    print('generating single')
    tensorseed = np.zeros((1, len(seed)), dtype='float32')
    tensorseed[0, :] = seed
    tensorseed = tensorseed * 3 / 128.

    src_spect = generate(tensorseed)

    values = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    values[:, :] = src_spect[0, :, :, 0] / 2 + 0.5

    values *= 256
    values = values.astype('int8')

    im = Image.fromarray(values, 'L')

    destwave, wflags = wtb.cqpowergram_to_wave(im=im, sr=OUTPUT_SR, hops=4, spectpow=2, bins_per_octave=CQ_BINS, fmin=CQ_BASE)

    timestring = f'{"{:0.2f}".format(time.time() - st)} sec.'
    print(f'time: {timestring}')
    return destwave, im, timestring


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
        destwavestep, wflags = wtb.cqpowergram_to_wave(im=im, sr=OUTPUT_SR, hops=4, spectpow=2, bins_per_octave=CQ_BINS, fmin=CQ_BASE)
        destwave = np.append(destwave, destwavestep)

    timestring = f'{"{:0.2f}".format(time.time() - st)} sec.'
    print(f'time: {timestring}')
    return destwave, timestring


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
        destwavestep, wflags = wtb.cqpowergram_to_wave(im=im, sr=OUTPUT_SR, hops=4, spectpow=2, bins_per_octave=CQ_BINS, fmin=CQ_BASE)
        destwave = np.append(destwave, destwavestep)

    timestring = f'{"{:0.2f}".format(time.time() - st)} sec.'
    print(f'time: {timestring}')
    return destwave, timestring


def init_generator():
    print("Initializing generator...")
    generate_init_dummy(64)
    print("Generator initialized.")
