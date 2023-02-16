# training tf model for 128x256 pixel spectrograms

import tensorflow as tf

import numpy as np
import os
from keras import layers
import time
import pathlib
from PIL import Image, ImageDraw

import mmm_WaveToolbox as wtb
import mmm_keyboardUI
import mmm_consoleToolbox as ctb

from scipy.io import wavfile


print("Checking for GPU support...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



KUI = mmm_keyboardUI.keyboardUI()


# project directory
PROJECTDIR = "D:/work/2022/TechnoGAN/networks/Spect_128x256/"
# subdir to save preview .png files during training
PREVIEWIMAGEDIR = PROJECTDIR + "previewimages/"
# subdir to store training progress
SAVEPOINTDIR = PROJECTDIR + "savepoints/"
# subdir to save preview .wav files during training
PREVIEWWAVDIR = PROJECTDIR + "previewwavs/"
# subdir to store traied model
GENERATORDIR = PROJECTDIR + "generator/"

AUTOTUNE = tf.data.experimental.AUTOTUNE

# subdir containing all training images
data_dir = pathlib.Path(f'{PROJECTDIR}dataset')
imageCount = len(list(data_dir.glob('*.png')))
imageList = list(data_dir.glob('*'))

# dimensions of training images
IMG_WIDTH = 128
IMG_HEIGHT = 256


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # scale to [-1,1] range
    img = (img - 0.5) * 2
    # resize the image to the desired size.
    # return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def prepare_for_training(ds, cache=True, shuffle_buffer_size=60000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(shuffle_buffer_size)

    # very small datasets can be repeated to fill the buffer???
    # ds = ds.repeat(50)
    # Repeat forever
    # ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


list_ds = tf.data.Dataset.list_files(str(data_dir / '*'))
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# number of training data files???
BUFFER_SIZE = 5500

# if running out of v-ram during startup, try a smaller batch-size
BATCH_SIZE = 32

# number of seed-values (size of first generator layer)
noise_dim = 64

# number of preview images/wavs per example step
num_examples_to_generate = 4

# Batch and shuffle the data
train_dataset = prepare_for_training(labeled_ds, True, BUFFER_SIZE)

# Convolution kernels (different sizes need different stride steps or produce different output dimensions
FILTER5 = (5, 5)
FILTER3 = (3, 3)


def make_generator_model():
    print('Building generator model:')
    model = tf.keras.Sequential()
    model.add(layers.Dense(64 * 32 * 128, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)

    model.add(layers.Reshape((64, 32, 128)))
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(96, FILTER5, strides=(1, 1), padding='same', use_bias=False))
    # 32x64
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(48, FILTER5, strides=(2, 2), padding='same', use_bias=False))
    # 64x128
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(48, FILTER5, strides=(2, 2), padding='same', use_bias=False))
    # 128x256
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, FILTER5, strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    # 128x256
    print(model.output_shape)
    return model


generator = make_generator_model()

noise = tf.random.normal([1, noise_dim])
generated_image = generator(noise, training=False)


def make_discriminator_model():
    print('Building discriminator model:')
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, FILTER5, strides=(2, 2), padding='same', input_shape=[256, 128, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    model.add(layers.Conv2D(96, FILTER5, strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    model.add(layers.Conv2D(192, FILTER5, strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    model.add(layers.Conv2D(192, FILTER5, strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    print(model.output_shape)

    return model


discriminator = make_discriminator_model()
decision = discriminator(generated_image)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_prefix = os.path.join(SAVEPOINTDIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# for preview files, the same seeds are used during training
# as long as the training script is not restartet, previews can be directly compared to monitor training progress
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# after how many training runs a preview batch is generated
PREVIEWSTEPS = 10

# after how many training runs a backup checkpoint is saved
CHECKPOINTSTEPS = 200


def train(dataset):
    # make sure to not overwrite previews from previous runs
    previewcounter = len(list(pathlib.Path(PREVIEWIMAGEDIR).glob('*.png')))
    print(f'{previewcounter} previews found from previous runs')
    maxSteps = 0
    maxSteps = np.ceil(BUFFER_SIZE/BATCH_SIZE)
    epoch = 0
    while True:
        epoch += 1
        # print (f'start epoch {epoch}')
        start = time.time()
        step = 0
        for image_batch in dataset:
            step += 1
            if (maxSteps == 0):
                print("\rstep " + str(step) + " of ?", end="")
            else:
                #print("\r" + "{:0.2f}".format(100 * step / maxSteps) + "% of epoch " + str(epoch), end="")
                ctb.printProgressBar(step, maxSteps, '', "of epoch " + str(epoch))
            train_step(image_batch)
            # pause learning with 'p'
            while KUI.PAUSED:
                time.sleep(0.2)
        if KUI.SAVENEXT:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print('checkpoint saved!')
            generator.save(GENERATORDIR, include_optimizer=True)
            print('generator saved!')
            KUI.donesaving()
        maxSteps = step
        print(f' - Time: {"{:0.2f}".format(time.time() - start)} sec')

        # output preview image and wavs every n epochs
        if (epoch + 1) % PREVIEWSTEPS == 0:
            generate_and_save_images(generator, previewcounter, seed)
            previewcounter += 1

        # Save the model every n epochs
        if (epoch + 1) % CHECKPOINTSTEPS == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print('checkpoint saved!')
            generator.save(GENERATORDIR, include_optimizer=True)
            print('generator saved!')


# preview images are upscaled for convenience
PREVIEWSCALE = 2
# samplerate that is stored with preview waves, does not influence amount of samples generated
# just so files are played at an aproximate proper speed when previewing them
OUTPUT_SR = 34952


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    collIm = Image.new('L', (IMG_WIDTH, (IMG_HEIGHT + 2) * 4))
    wi = collIm.width
    he = collIm.height

    for i in range(predictions.shape[0]):
        if (i < 4):
            values = np.zeros((IMG_HEIGHT, IMG_WIDTH))
            values[:, :] = predictions[i, :, :, 0] / 2 + 0.5
            values *= 256
            values = values.astype('int8')  # .swapaxes(0,1)

            im = Image.fromarray(values, 'L')

            collIm.paste(im, (0, i * (IMG_HEIGHT + 2)))

            destwave, wflags = wtb.powergram_to_wave(im=im, hops=4, spectpow=3)
            try:
                wavfile.write(f'{PREVIEWWAVDIR}/wav_ep{epoch}_{i}.wav', OUTPUT_SR, destwave)
            except:
                print(f'ERROR: unable to save wav_ep{epoch}_{i}.wav')
            else:
                print(
                    f'saved powergram-to-wave: wav_ep{epoch}_{i}.wav, length:{len(destwave)} lowsignal:{wflags[0]}, clipping:{wflags[1]}')

    collIm = collIm.resize((collIm.width * PREVIEWSCALE, collIm.height * PREVIEWSCALE), Image.Resampling.NEAREST)
    collIm.save(f'{PREVIEWIMAGEDIR}/collection_{epoch}.png')


def generate_single_wave(seed, save_img=False):
    st = time.time()
    print('generating single')
    tensorseed = np.zeros((1, len(seed)), dtype='float32')
    tensorseed[0, :] = tf.convert_to_tensor(seed)
    tensorseed = tensorseed * 3 / 128.

    src_spect = generator(tensorseed, training=False)

    values = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    values[:, :] = src_spect[0, :, :, 0] / 2 + 0.5
    values *= 256
    values = values.astype('int8')

    im = Image.fromarray(values, 'L')

    destwave, wflags = wtb.powergram_to_wave(im=im, hops=4, spectpow=3)

    print(f'time {"{:0.2f}".format(time.time() - st)} sec.')
    return destwave, im


def generate_fade(startseed, endseed, steps):
    st = time.time()
    print('generating fade')
    tensorstartseed = np.zeros((1, len(startseed)), dtype='float32')
    tensorendseed = np.zeros((1, len(endseed)), dtype='float32')
    tensorstartseed[0, :] = tf.convert_to_tensor(startseed)
    tensorendseed[0, :] = tf.convert_to_tensor(endseed)
    tensorstartseed = tensorstartseed * 3 / 128.
    tensorendseed = tensorendseed * 3 / 128.

    values = np.zeros((IMG_HEIGHT, IMG_WIDTH))

    destwave = np.array(0, dtype='int16')

    for i in range(steps):
        tensorseed = wtb.lerp(tensorstartseed, tensorendseed, i / (steps - 1))
        src_spect = generator(tensorseed, training=False)
        values[:, :] = src_spect[0, :, :, 0] / 2 + 0.5
        values *= 256
        intvalues = values.astype('int8')
        im = Image.fromarray(intvalues, 'L')
        destwavestep, wflags = wtb.powergram_to_wave(im=im, hops=4, spectpow=3)
        destwave = np.append(destwave, destwavestep)

    print(f'time {"{:0.2f}".format(time.time() - st)} sec.')
    return destwave


def generate_pingpong(startseed, endseed, steps):
    st = time.time()
    print('generating pingpong')
    tensorstartseed = np.zeros((1, len(startseed)), dtype='float32')
    tensorendseed = np.zeros((1, len(endseed)), dtype='float32')
    tensorstartseed[0, :] = tf.convert_to_tensor(startseed)
    tensorendseed[0, :] = tf.convert_to_tensor(endseed)
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
        tensorseed = wtb.lerp(tensorstartseed, tensorendseed, i / (steps - 1))
        src_spect = generator(tensorseed, training=False)
        values[:, :] = src_spect[0, :, :, 0] / 2 + 0.5
        values *= 256
        intvalues = values.astype('int8')
        im = Image.fromarray(intvalues, 'L')
        destwavestep, wflags = wtb.powergram_to_wave(im=im, hops=4, spectpow=3)
        destwave = np.append(destwave, destwavestep)

    print(f'time {"{:0.2f}".format(time.time() - st)} sec.')
    return destwave


def load_checkpoint():
    print("Loading trained model data...")
    cp = checkpoint.restore(tf.train.latest_checkpoint(SAVEPOINTDIR))
    wtb.SPECTRUMRESCALE = 160


if __name__ == '__main__':
    cp = checkpoint.restore(tf.train.latest_checkpoint(SAVEPOINTDIR))
    wtb.SPECTRUMRESCALE = 160

    # TRAIN
    train(train_dataset)

