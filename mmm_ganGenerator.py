# wrapper for communication with osc-server and different generator models

from scipy.io import wavfile
import numpy as np
import os
import tempfile

# IMPORT THE NETWORK HERE ####################################################
import mmm_onnxInference_512x256 as gan
# use training scripts like GAN_SPECT_128x256 to run inference directly with tensorflow models
##############################################################################

# create temp dir on default os temp folder
# this will store generated files
# TODO: delete files on quit (not too much space is used and generated files overwrite each other, but would be cleaner)
TARGET_DIR = tempfile.gettempdir() + "\\TechnoGAN\\"
os.makedirs(TARGET_DIR, exist_ok=True)
TARGET_DIR = TARGET_DIR.replace("\\", "/")
print("Temp directory set to ", TARGET_DIR)
TARGET_SR = 22000

IS_INITIALIZED = False


def isalive(*params):
    return 'alive', True, ''


def init_generator(*nix):
    global IS_INITIALIZED
    if not IS_INITIALIZED:
        # gan.load_checkpoint()
        gan.init_generator()
        IS_INITIALIZED = True
        return 'msg', 'Init Generator: network loaded', ''
    else:
        return 'msg', 'Init Generator: network already up', ''


def set_target_sr(sr):
    global TARGET_SR
    TARGET_SR = sr[0]
    return 'msg', f'target sampling rate set to: {TARGET_SR}', ''


def set_target_dir (dir):
    global TARGET_DIR
    TARGET_DIR = dir[0]
    return 'msg', f'target directory set to: {TARGET_DIR}', ''


def get_target_dir (*nix):
    return 'msg', f'target dir is: {TARGET_DIR}', ''


def generate_single_wav (seed):
    fn = 'testo.wav'
    outwav, outimg, timestring = gan.generate_single_wave(seed)
    print (f'writing to {TARGET_DIR}{fn}')
    try:
        wavfile.write(TARGET_DIR + fn, TARGET_SR, outwav)
    except:
        return 'msg', f'ERROR writing to {TARGET_DIR}{fn}', ''

    return 'wav', f'{TARGET_DIR}{fn}', f'Generated single in {timestring}: {fn}'


def generate_single_wav_id (params):
    fn = 'single'
    id = params[0]
    seed = list(params[1:])

    outwav, outimg, timestring = gan.generate_single_wave(seed)

    fn = f'{TARGET_DIR}{fn}{id}.wav'
    print (f'writing to {fn}')
    try:
        wavfile.write(fn, TARGET_SR, outwav)
    except:
        return 'msg', f'ERROR writing to {fn}', ''

    return 'wav_id', f'{id} {fn}', f'Generated single in {timestring}: {fn}'


def generate_single_wav_id_img (params):
    fn = 'single'
    id = params[0]
    seed = list(params[1:])

    outwav, outimg, timestring = gan.generate_single_wave(seed)

    fn = f'{TARGET_DIR}{fn}{id}'
    img_fn = f'{fn}.png'
    fn = f'{fn}.wav'
    print (f'writing to {fn} and {img_fn}')
    try:
        wavfile.write(fn, TARGET_SR, outwav)
    except:
        return 'msg', f'ERROR writing to {fn}', ''

    try:
        outimg.save(img_fn)
    except:
        return 'msg', f'ERROR writing to {img_fn}', ''

    return 'wav_id_img', f'{id} {fn} {img_fn}', f'Generated single in {timestring}: {fn}'


def generate_random_fade (seed):
    fn = 'testo.wav'
    steps = seed[0]
    morph = seed[1]
    startseed = list(seed[2:])
    endseed = startseed.copy()
    for i in range(len(startseed)):
        r = np.random.randint(-morph , morph + 1)
        endseed[i] += r

    outwav, timestring = gan.generate_fade(startseed, endseed, steps)
    print (f'writing to {TARGET_DIR}{fn}')
    try:
        wavfile.write(TARGET_DIR + fn, TARGET_SR, outwav)
    except:
        return 'msg', f'ERROR writing to {TARGET_DIR}{fn}', ''
    return 'wav', f'{TARGET_DIR}{fn}', f'Generated randomfade in {timestring}: {fn}'


def generate_fade (seed):
    fn = 'fade'
    id = seed[0]
    steps = seed[1]
    l = int((len(seed) - 2) / 2)
    startseed = list(seed[2:l+2])
    endseed = list(seed[l+2:2*l+2])

    outwav, timestring = gan.generate_fade(startseed, endseed, steps)

    fn = f'{TARGET_DIR}{fn}{id}.wav'
    print (f'writing to {fn}')
    try:
        wavfile.write(fn, TARGET_SR, outwav)
    except:
        return 'msg', f'ERROR writing to {fn}', ''
    return 'wav_id', f'{id} {fn}', f'Generated fade in {timestring}: {fn}'


def generate_pingpong (seed):
    fn = 'pingpong'
    id = seed[0]
    steps = seed[1]
    l = int((len(seed) - 2) / 2)
    startseed = list(seed[2:l+2])
    endseed = list(seed[l+2:2*l+2])

    outwav, timestring = gan.generate_pingpong(startseed, endseed, steps)
    fn = f'{TARGET_DIR}{fn}{id}.wav'
    print(f'writing to {fn}')
    try:
        wavfile.write(fn, TARGET_SR, outwav)
    except:
        return 'msg', f'ERROR writing to {fn}', ''
    return 'wav_id', f'{id} {fn}', f'Generated pingpong in {timestring}: {fn}'
