# tools for resampling, converting, saving and slicing wave files
# not all functions in this collection are currently in use, many of the conversions are remains of failed experiments
# as many of the tried conversions didn't work for training the GAN

from scipy.io import wavfile # scipy library to read wav files
from scipy import fft
from librosa import core as lc
import scipy.signal as sps
import numpy as np
import os
from PIL import Image, ImageDraw

# exponential factor used when storing spectrograms as pngs, to preserve detail in lower magnitudes
# should remain the same throughout dataset creation, training and inference of a single model
SPECTRUMEXPFACTOR = 4

# convert polar to carthesian
def pol2car(r,theta):
    return r * np.exp( 1j * theta )

# convert carthesian to polar
def car2pol(z):
    return ( np.abs(z), np.angle(z) )


# CONVERT WAVEFORM DATA TO SPECTRUM via FFT
#       input: waveform as array
#       output: magnitude and phase as array (half length of input)
def wave_to_spect (src_wave):
    # create fft
    src_spect = fft.rfft(src_wave)
    # convert to polar
    src_spect_mag, src_spect_phase = car2pol (src_spect)
    SrcSpectMag = src_spect_mag ** (1 / SPECTRUMEXPFACTOR)
    return SrcSpectMag, src_spect_phase


# CONVERT SPECTRUM DATA TO WAVEFORM via IFFT
#       input: magnitude and phase as array
#       output: waveform as array (double length of input)
def spect_to_wave (mag, phase, power_change = 1):
    #reconvert to cartesian
    mag = mag ** SPECTRUMEXPFACTOR
    dest_spect = pol2car(mag, phase)
    dest_wave = fft.irfft(dest_spect)
    # volume refitting (should roughly be 1/downsamplefactor)
    n = 1 / power_change
    # cut away complex parts and convert to int
    l = len(dest_wave)
    dest_wave_m = np.zeros(l, dtype=np.int16)
    dest_wave_m[:] = dest_wave[:] * n
    return dest_wave_m

# calculate number of samples for one slice
def calc_slice_sample_length(samplerate, bpm, beats):
    beatspersecond = bpm / 60.0
    samplesperbeat = samplerate / beatspersecond
    samplesperslice = samplesperbeat * beats
    return int(samplesperslice)


# SLICE WAVEFILE AND SAVE TO SEPARATE FILES
#       input:  samplelength - length of single segment in samples
#               jump - save every n-th segment
#               filenames are path and filename without extensions (.wav is assumed)
def slice_wave_and_save(input_filename, output_filename, bpm, jump=1, beats=1):
    src_sr, src_wave = wavfile.read(input_filename + ".wav")
    samplelength = calc_slice_sample_length(src_sr, bpm, beats)

    available_slices = int(len(src_wave) / samplelength)
    print(f'{input_filename} sr:{src_sr} samples:{len(src_wave)} beats_per_slice:{beats} skip:{jump} slices:{available_slices}')

    counter = 0
    for s in range(0, available_slices, jump):
        current_wave = src_wave[s * samplelength:(s + 1) * samplelength]
        wavfile.write(output_filename + f'_{counter}.wav', src_sr, current_wave)
        counter += 1
    print(f'created {counter} slices')




# down- and upsample spectrums(?)
# currently not used anywhere, not sure why i needed it at some point
# TODO: cleanup and delete old unused conversion functions
def downsample_spect (mag, phase, size):
    # lowpass filter / downsample
    res_mag = mag[0:size]
    res_phase = phase[0:size]
    return (res_mag, res_phase)

def upsample_spect (mag, phase, size):
    l = len(mag)
    dest_mag = np.zeros(size, dtype=np.float64)
    dest_phase = np.zeros(size, dtype=np.float64)
    dest_mag[0:l] = mag[0:l]
    dest_phase[0:l] = phase[0:l]
    return dest_mag, dest_phase


# generates spectrogram for full audiofile along x-axis without phase
# currently unused
# most likely used for testing/experimentation at some point
def generateSpectrogramLong(wave):
    wave = wave[0:2752]
    l = len(wave)
    offset = int(np.floor(l/2))
    im = Image.new('L', (l,1))

    destMag, destPhase = wave_to_spect(wave)
    maxmag = 0
    for currFreq in range(len(destMag)-1):
        mVal = int(destMag[currFreq]) * 4 # * 255 / 20)
        mPhase = int((destPhase[currFreq] + np.pi) * 255 / (2 * np.pi))
        maxmag = np.max((mVal, maxmag))
        im.putpixel((currFreq, 0), mVal)
        im.putpixel((offset + currFreq, 0), mPhase)
    print(maxmag)
    return im

# reverse function of spectrogram to full audiofile along x-axis without phase
# currently unused
# most likely used for testing/experimentation at some point
def spectImageLong2Wav(im = Image.new('L',(16,1))):
    l = im.width
    offset = int(np.floor(l/2))
    print (offset)

    sMag = np.zeros(offset, dtype=np.float64)
    sPhase = np.zeros(offset, dtype=np.float64)

    for i in range(offset):
        sMag[i] = im.getpixel((i,0)) / 4 #* 20 / 255
        currPhase = im.getpixel((offset + i,0))
        sPhase[i] = (currPhase * 2 * np.pi / 255) - np.pi

    destWave = spect_to_wave(sMag, sPhase).real
    return destWave

# convert spectrogram image (top half of image = magnitude, bottom half = phase)
# back to waveform
def spectImage2Wav(im = Image.new('I',(8,8))):
    bucketWidth = im.height
    spectWidth = int(im.height / 2)
    buckets = im.width
    sMag = np.zeros(spectWidth, dtype=np.float64)
    sPhase = np.zeros(spectWidth, dtype=np.float64)
    destWave = np.zeros(bucketWidth * buckets, dtype=np.int16)

    #fftFreqs = fftfreq(bucketWidth, 1.0 / 6000)[0:8]
    #contPhase = np.zeros(spectWidth) + np.pi
    #contPhase[0] = 0
    for currBucket in range (0, buckets):
        for currFreq in range (0, spectWidth):
            currMag = im.getpixel((currBucket, currFreq))
            currPhase = im.getpixel((currBucket, currFreq + spectWidth))
            sMag[currFreq] = currMag * 20 / 255
            sPhase[currFreq] = (currPhase * 2 * np.pi / 255) - np.pi

        #contPhase = fftFreqs * currBucket * 2 * np.pi * (bucketWidth / 6000)
        #contPhase = contPhase % (2 * np.pi)
        #contPhase = contPhase - np.pi
        #print(contPhase)
        #wavBucket = convert2Wave(sMag, contPhase)

        wavBucket = spect_to_wave(sMag, sPhase)
        destWave[currBucket * bucketWidth:(currBucket + 1) * bucketWidth - 2] = wavBucket[0:bucketWidth].real
        # HACK to fill missing samples from ifft
        destWave[(currBucket + 1) * bucketWidth - 2] = destWave[(currBucket + 1) * bucketWidth - 3]
        destWave[(currBucket + 1) * bucketWidth - 1] = destWave[(currBucket + 1) * bucketWidth - 3]

    return destWave


WAVEIMAGEX = 172
WAVEIMAGEY = 16
# convert waveform directly to grayscale image
def wave_to_waveimage(wave, size):
    l = len(wave)
    im = Image.new('L', (size, size))
    for y in range (size):
        for x in range (size):
            val = int((wave[x+y*size] + 32767)/256)
            im.putpixel((x, y), val)
    return im

# convert grayscale image directly to waveform
def waveimage_to_wave(img = Image.new('I',(WAVEIMAGEX,16))):
    size = img.width
    destWave = np.zeros(size*size, dtype=np.int16)
    mini = 256*256
    maxi = -256*256
    for y in range (size):
        for x in range (size):
            val = img.getpixel((x,y))
            #print (val)
            val = (val - 128) * 256
            maxi = np.max([val, maxi])
            mini = np.min([val, mini])
            destWave[x+y*size] = val
            #print(f'   {val}')
    print(f'min {mini} - max {maxi}')
    return destWave


# resample wave and cast to int16
def resample_int16(wave, samples):
    resampled = sps.resample(wave, samples)
    dest_wave = np.zeros(samples, dtype='int16')
    for i in range (len(resampled)):
        dest_wave[i] = int(resampled[i])
    return dest_wave


# linear interpolation of two values
def lerp(v1, v2, d):
    return v1 * (1 - d) + v2 * d


# generate spectrogram greyscale image from wave
# top half - magnitude, bottom half - phase
# currently unused.
def generate_spectrogram(wave, spect_width):
    spect_width2 = spect_width * 2
    buckets = int(np.floor(len(wave)/spect_width2))
    #print(f'buckets:{buckets}')
    im = Image.new('L', (buckets, spect_width2))

    for currBucket in range (0, buckets):
        bucket = wave[currBucket * spect_width2:((currBucket+1) * spect_width2)]
        bMag, bPhase = wave_to_spect(bucket)
        #print(f'bucket:{currBucket} lenBmag: {len(bMag)}')
        maxVal = 255
        for currFreq in range (0, spect_width):
            #im.putpixel((currBucket,currFreq),100)
            mVal = int(bMag[currFreq] * 255 / 20)
            pVal = int((bPhase[currFreq] + np.pi) * 255 / (2 * np.pi))
            maxVal = np.min((pVal, maxVal))
            im.putpixel((currBucket, currFreq), mVal)
            im.putpixel((currBucket, currFreq + spect_width), pVal)
        #print (f'{np.min(bPhase)} - {maxVal}')
    return im


# convert waveform to "powergram" (spectrogram disregarding phase, only magnitude is stored in the image -
# when converting back to wave, griffin-lim algo will be used to approximate missing phase data.

# scale down spectrum to keep values around [0., 1.]
SPECTRUMRESCALE = 160
# default power factor when converting to int values, to keep details in lower magnitudes
# both values should be kept constant during training data generation and using the trained model
SPECTRUMPOWER = 4
def wave_to_powergram(wave, spect_width, hops = int(1), spectpow = SPECTRUMPOWER):

    if (hops == 1):
        win = 'boxcar'
    else:
        win = 'hann'

    chunksize = spect_width * 2
    chunks = int(np.floor(len(wave)/chunksize))
    slices = chunks * hops
    hoplength = int (chunksize/hops)
    im = Image.new('L', (slices, spect_width))
    print(f'generating powergram size({slices},{spect_width}) from {len(wave)} samples in {chunks} chunks of size {chunksize}')


    # convert wave to float
    floatwave = np.zeros(len(wave), dtype=np.float64)
    for i in range (len(wave)):
        floatwave[i] = float(wave[i]) / np.power(2, 15)

    new_spect = lc.stft(floatwave, n_fft=chunksize, hop_length=hoplength, win_length=chunksize, window=win, center=True, pad_mode='wrap')
    new_mag, new_phase = car2pol(new_spect)

    new_mag /= SPECTRUMRESCALE

    # normalize if necessary (make sure there is no clipping in the resulting image),
    # set warning flags [weaksignal, clipping]
    warnings = [False, False]
    mag_max = np.max(new_mag)
    if (mag_max < 0.5):
        warnings[0] = True
    if (mag_max > 1):
        print (f'warning: magnitude max is {mag_max} (>1) scaling down...')
        new_mag /= mag_max
        warnings[1] = True

    new_mag = np.power(new_mag, 1/spectpow) * 256

    values = new_mag.astype('int8')[:-1,:-1]
    im = Image.fromarray(values, 'L')  # way faster than putpixel-loop
    return im, warnings



# convert 'powergram' image to waveform using grifin-lim algorithm from librosa.core
# this is the best way i found to recreate a waveform from a magnitude-only spectrogram
def powergram_to_wave(im = Image.new('I',(8,8)), hops = int(1), spectpow = SPECTRUMPOWER):
    if (hops == 1):
        win = 'boxcar'
    else:
        win = 'hann'

    spect_width = im.height
    chunks = int(im.width/hops)
    chunksize = spect_width * 2
    hoplength = int(chunksize/hops)
    #s_mag = np.zeros((spect_width + 1, chunks * hops), dtype=np.float64)
    dest_wave = np.zeros(chunksize * chunks, dtype=np.int16)

    #print(f'powergram_to_wave: imagesize {im.width} x {im.height}, to wavesize {len(dest_wave)}')


    s_mag2 = np.asarray(im) / 256
    s_mag2 = np.power(s_mag2, spectpow) * SPECTRUMRESCALE
    s_mag2 = np.append(s_mag2, np.zeros((1, chunks*hops), dtype=np.float64), axis=0)


    dest_wave = lc.griffinlim(s_mag2, hop_length=hoplength, win_length=chunksize, window = win, center = True, pad_mode='wrap', n_iter=32, init=None)


    dest_wave_int = np.zeros(len(dest_wave), dtype='int16')
    # normalize if necessary, (make sure resulting wave isn't clipping
    # set warning flags [weaksignal, clipping]
    overhead = max((max(dest_wave), abs(min(dest_wave))))
    warnings = [False, False]

    if (overhead < 0.5):
        warnings[0] = True
    if (overhead > 1):
        #print (f'warning: converted wave exceeds range ({overhead}), normalizing...')
        dest_wave *= 0.999 / overhead  # using 1. might still clip the signal when values are exactly 1
        warnings[1] = True

    # rescale float wave to int16 [-1., 1.] to [-2^15, 2^15]
    dest_wave *= np.power(2, 15)
    dest_wave_int = dest_wave.astype(np.int16)

    return dest_wave_int, warnings


# convert waveform to constant-q 'powergram'
# with this method, frequency bins of the transformation are kept in constant harmonic intervals
CQ_BASE = 16.35 # default frequency of the lowest bin (in Hz)
def wave_to_cqpowergram(wave, spect_width, hops=int(1), spectpow=SPECTRUMPOWER, fmin=CQ_BASE, bins_per_octave=12):

    if (hops == 1):
        win = 'boxcar'
    else:
        win = 'hann'

    chunksize = spect_width # * 2
    chunks = int(np.floor(len(wave)/chunksize))
    slices = chunks * hops
    hoplength = int (chunksize/hops)
    im = Image.new('L', (slices, spect_width))
    print(f'generating constant q powergram, size({slices},{spect_width}) from {len(wave)} samples in {chunks} chunks of size {chunksize}')


    # convert wave to float
    floatwave = np.zeros(len(wave), dtype=np.float64)
    for i in range (len(wave)):
        floatwave[i] = float(wave[i]) / np.power(2, 15)

    new_spect = lc.cqt(floatwave, sr=44100, hop_length=hoplength, n_bins=spect_width, fmin=fmin, bins_per_octave=bins_per_octave, window=win, pad_mode='wrap', tuning=0.0)
    new_mag, new_phase = car2pol(new_spect)

    new_mag /= SPECTRUMRESCALE

    # normalize if necessary, set warning flags [weaksignal, clipping]
    warnings = [False, False]
    mag_max = np.max(new_mag)
    if (mag_max < 0.5):
        warnings[0] = True
    if (mag_max > 1):
        print (f'warning: magnitude max is {mag_max} (>1) scaling down...')
        new_mag /= mag_max
        warnings[1] = True

    new_mag = np.power(new_mag, 1/spectpow) * 256

    values = new_mag.astype('int8')[:,:-1]
    im = Image.fromarray(values, 'L')  # way faster than putpixel-loop
    return im, warnings


# reconvert constant-q powergram to waveform, using griffin-lim constant-q algorithm from librosa.core to
# approximate missing phase data.
def cqpowergram_to_wave(im = Image.new('I', (8, 8)), sr=16384, hops=int(1), spectpow=SPECTRUMPOWER, bins_per_octave=12, fmin=CQ_BASE):
    if (hops == 1):
        win = 'boxcar'
    else:
        win = 'hann'

    spect_width = im.height
    chunks = int(im.width/hops)
    chunksize = spect_width * 2
    hoplength = int(chunksize/hops)
    s_mag = np.zeros((spect_width + 1, chunks * hops), dtype=np.float64)

    # print(f'powergram_to_wave: imagesize {im.width} x {im.height}, to wavesize {len(dest_wave)}')

    for x in range(chunks*hops):
        for y in range(spect_width):
            curr_mag = float(im.getpixel((x, y))) / 256
            curr_mag = pow(curr_mag, spectpow) * SPECTRUMRESCALE
            s_mag[y, x] = curr_mag
    dest_wave = lc.griffinlim_cqt(s_mag, hop_length=hoplength, sr=sr, window=win, fmin=fmin, pad_mode='wrap', bins_per_octave=bins_per_octave, tuning=0.0)
    dest_wave_int = np.zeros(len(dest_wave), dtype='int16')


    #normalize if necessary, set warning flags [weaksignal, clipping]
    overhead = max((max(dest_wave), abs(min(dest_wave))))
    warnings = [False, False]

    if (overhead < 0.5):
        warnings[0] = True
    if (overhead > 1):
        #print (f'warning: converted wave exceeds range ({overhead}), normalizing...')
        dest_wave *= 0.999 / overhead  # using 1. might still clip the signal when values are exactly 1
        warnings[1] = True

    for i in range(len(dest_wave)):
        dest_wave_int[i] = int(dest_wave[i] * np.power(2, 15))
    return dest_wave_int, warnings


# resample waveform and save to file
def resample_wav(src_fn, dest_fn, samples):
    # load wav
    src_sr, src_wave = wavfile.read(src_fn)
    src_len = len(src_wave)
    print (f'loading {src_fn} with sr: {src_sr} and {len(src_wave)} samples')

    # resample source wave
    dest_wave = resample_int16(src_wave, samples)

    # save downsampled wavefile
    dest_sr = int(samples * src_sr / src_len)
    print (f"saving resampled: {dest_fn} at sr: {dest_sr} and {len(dest_wave)} samples, valrange {min(dest_wave)}/{max(dest_wave)}")
    wavfile.write(dest_fn, dest_sr, dest_wave)


# convert wavefile to powergram image and save to file
def convert_and_save (src_filename, dest_filename, dest_ctrl_filename, dest_spectimg_filename, dest_waveimg_filename, size, beats = 1, spectpow = 2):
    # load wav
    src_sr, src_wave = wavfile.read(src_filename)
    src_len = len(src_wave)
    slice_len = src_len / (size * beats)
    ctrl_wave = np.zeros(len(src_wave))

    src_max = max(src_wave)
    src_min = min(src_wave)

    print (f'loading {src_filename} with sr: {src_sr} and {len(src_wave)} samples, slicelength: {slice_len}, valrange {src_min}/{src_max}')
    slice_len = int(slice_len)

    dest_len = 2*64*64*beats
    ctrl_wave = np.array(0, dtype='int16')
    dest_wave = np.zeros(dest_len, dtype='int16')

    ds_rate = (2 * size/slice_len)
    dest_sr = int(src_sr * ds_rate)
    wavimg_sr = int(src_sr * ds_rate/2)

    #resample source wave
    dest_wave = resample_int16(src_wave, dest_len)

    im, warn = wave_to_powergram(dest_wave, size, hops=4, spectpow=spectpow)
    #im, warn = wave_to_cqpowergram(dest_wave, size, 4, spectpow = spectpow)
    print (f"saving spectrum img: {dest_spectimg_filename}.png, warnings[(low/clp)] {warn}, size {im.width}x{im.height}")
    im.save(dest_spectimg_filename + ".png")

    dest_wave = resample_int16(src_wave, size*size)


    # save several other conversions/formats for testing/debugging

    #save waveimage
    #wave_img = wave_to_waveimage(dest_wave, size)
    #print (f'saving waveimg: {dest_waveimg_filename}.png')
    #wave_img.save(dest_waveimg_filename + ".png")

    #im2 = Image.open(dest_spectimg_filename + ".png")
    #ctrl_wave, warn2 = powergram_to_wave(im = im2, hops = 4)

    #ctrl_wave = resample_int16(ctrl_wave, src_len)

    #save downsampled wavefile
    #print (f"saving resampled: {dest_filename} at sr: {dest_sr} and {len(dest_wave)} samples, valrange {min(dest_wave)}/{max(dest_wave)}")
    #wavfile.write(dest_filename, int(dest_sr/2), dest_wave)

    # save control-wavefile
    #print (f"saving ctrlwave: {dest_ctrl_filename} at sr: {src_sr} and {len(ctrl_wave)} samples, valrange {min(ctrl_wave)}/{max(ctrl_wave)}")
    #wavfile.write(dest_ctrl_filename, dest_sr, ctrl_wave)
    return warn


# convert wavefile to constant-q powergram image and save to file
def convert_and_save_cq (src_filename, dest_filename, dest_ctrl_filename, dest_spectimg_filename, dest_waveimg_filename, size, beats=1, spectpow=2, fmin=CQ_BASE, bpo=4, hops=4):
    # load wav
    src_sr, src_wave = wavfile.read(src_filename)
    src_len = len(src_wave)
    slice_len = src_len / (size * beats)
    ctrl_wave = np.zeros(len(src_wave))

    src_max = max(src_wave)
    src_min = min(src_wave)

    print (f'loading {src_filename} with sr: {src_sr} and {len(src_wave)} samples, slicelength: {slice_len}, valrange {src_min}/{src_max}')
    slice_len = int(slice_len)

    dest_len = 2*64*64*beats
    ctrl_wave = np.array(0, dtype='int16')
    dest_wave = np.zeros(dest_len, dtype='int16')

    ds_rate = (2 * size/slice_len)
    dest_sr = int(src_sr * ds_rate)
    wavimg_sr = int(src_sr * ds_rate/2)

    #resample source wave
    dest_wave = resample_int16(src_wave, dest_len)

    im, warn = wave_to_cqpowergram(dest_wave, size, hops=hops, spectpow=spectpow, fmin=fmin, bins_per_octave=bpo)
    #im, warn = wave_to_cqpowergram(dest_wave, size, 4, spectpow = spectpow)
    print (f"saving spectrum img: {dest_spectimg_filename}.png, warnings[(low/clp)] {warn}, size {im.width}x{im.height}")
    im.save(dest_spectimg_filename + ".png")

    dest_wave = resample_int16(src_wave, size*size)

    # save several other conversions/formats for testing/debugging

    #save waveimage
    #wave_img = wave_to_waveimage(dest_wave, size)
    #print (f'saving waveimg: {dest_waveimg_filename}.png')
    #wave_img.save(dest_waveimg_filename + ".png")

    #im2 = Image.open(dest_spectimg_filename + ".png")
    #ctrl_wave, warn2 = powergram_to_wave(im = im2, hops = 4)

    #ctrl_wave = resample_int16(ctrl_wave, src_len)

    #save downsampled wavefile
    #print (f"saving resampled: {dest_filename} at sr: {dest_sr} and {len(dest_wave)} samples, valrange {min(dest_wave)}/{max(dest_wave)}")
    #wavfile.write(dest_filename, int(dest_sr/2), dest_wave)

    # save control-wavefile
    #print (f"saving ctrlwave: {dest_ctrl_filename} at sr: {src_sr} and {len(ctrl_wave)} samples, valrange {min(ctrl_wave)}/{max(ctrl_wave)}")
    #wavfile.write(dest_ctrl_filename, dest_sr, ctrl_wave)
    return warn


# convert a whole directory of .wav files to powergram images
def convert_and_save_dir(srcDir, destWavDir, destWavImgDir, destSpectImgDir, fileType, spectWidth, beats = 1, spectpow = 2):
    filecounter = 0
    clipcounter = 0
    lowsignalcounter = 0

    # find and load WAV-Files
    for file in os.listdir(srcDir):
        if file.endswith(fileType):
            #print("working on " + os.path.join(srcDir + srcFilename))
            srcFilename = os.path.splitext(file)[0]
            destFilename = srcFilename

            wflags = convert_and_save(srcDir + srcFilename + fileType,
                           destWavDir + destFilename + "downs" + fileType,
                           destWavDir + destFilename + "ctrl" + fileType,
                           destSpectImgDir + destFilename,
                           destWavImgDir + destFilename,
                           spectWidth,
                           beats = beats,
                           spectpow = spectpow)
            filecounter += 1
            if wflags[0]:
                lowsignalcounter += 1
            if wflags[1]:
                clipcounter += 1
    print (f'converted {filecounter} files, {lowsignalcounter} with low signal, {clipcounter} normalized due to clipping')


# convert a whole directory of .wav files to constant-q powergram images
def convert_and_save_dir_cq(srcDir, destWavDir, destWavImgDir, destSpectImgDir, fileType, spectWidth, beats=1, spectpow=2, bins_per_octave=4, fmin=CQ_BASE):
    filecounter = 0
    clipcounter = 0
    lowsignalcounter = 0

    # find and load WAV-Files
    for file in os.listdir(srcDir):
        if file.endswith(fileType):
            # print("working on " + os.path.join(srcDir + srcFilename))
            srcFilename = os.path.splitext(file)[0]
            destFilename = srcFilename

            wflags = convert_and_save_cq(srcDir + srcFilename + fileType,
                                         destWavDir + destFilename + "downs" + fileType,
                                         destWavDir + destFilename + "ctrl" + fileType,
                                         destSpectImgDir + destFilename,
                                         destWavImgDir + destFilename,
                                         spectWidth,
                                         beats=beats,
                                         spectpow=spectpow,
                                         bpo=bins_per_octave,
                                         fmin=fmin)
            filecounter += 1
            if wflags[0]:
                lowsignalcounter += 1
            if wflags[1]:
                clipcounter += 1
    print (f'converted {filecounter} files, {lowsignalcounter} with low signal, {clipcounter} normalized due to clipping')


# resample a whole direcotry of wavefiles
def resample_dir(src_dir, dest_dir, samples):
    filecounter = 0
    # find and load WAV-Files
    for file in os.listdir(src_dir):
        if file.endswith('.wav'):
            #print("working on " + os.path.join(srcDir + srcFilename))
            src_fn = os.path.splitext(file)[0]

            resample_wav(src_dir + src_fn + '.wav', dest_dir + src_fn + '.wav', samples)
            filecounter += 1
    print (f'resampled {filecounter} files')




def saveSpectImage2Wav(srcFile, destFile, sr):
    im = Image.open(srcFile + ".png")
    destWave = spectImage2Wav(im)
    wavfile.write(destFile, sr, destWave)

def saveSpectImageLong2Wav(srcFile, destFile, sr):
    im = Image.open(srcFile + ".png")
    destWave = spectImageLong2Wav(im)
    wavfile.write(destFile, sr, destWave)

def save_waveimage_to_wave(srcFile, destFile, sr):
    im = Image.open(srcFile + ".png")
    destWave = waveimage_to_wave(im)
    wavfile.write(destFile, sr, destWave)

# convert powergram image (.png) to wave (.wav)
def save_powergram_image_to_wav(src_file, dest_file, sr, hops = 4, spectpow = 4):
    im = Image.open(src_file + ".png")
    destWave, warnflags = powergram_to_wave(im, hops = hops, spectpow = spectpow)
    print (f'saving powergram to wave: {dest_file} - low Sig:{warnflags[0]} clipping:{warnflags[1]}')
    wavfile.write(dest_file, sr, destWave)


# convert constant-q powergram image (.png) to wave (.wav)
def save_cqpowergram_image_to_wav(src_file, dest_file, sr, hops = 4, spectpow = 4, bins_per_octave=4, fmin=CQ_BASE):
    im = Image.open(src_file + ".png")
    destWave, warnflags = cqpowergram_to_wave(im, hops=hops, sr=sr, spectpow=spectpow, bins_per_octave=bins_per_octave, fmin=fmin)
    print (f'saving powergram to wave: {dest_file} - low Sig:{warnflags[0]} clipping:{warnflags[1]}')
    wavfile.write(dest_file, sr, destWave)


# convert wavefile (.wav) to constant-q powergram image (.png)
# TODO: figure out why there are 2 such functions (convert_and_save_cq)
def save_wave_to_cqpowergram(src_file, dest_file, hops=4, spectpow=4, fbins=64, slices=64, bins_per_octave=4, fmin=CQ_BASE):
    # load wav
    src_sr, src_wave = wavfile.read(src_file)
    src_len = len(src_wave)
    print(f'loading {src_file} with sr: {src_sr} and {len(src_wave)} samples')

    bucketsize = 2 * fbins

    dest_len = bucketsize * slices
    dest_wave = resample_int16(src_wave, dest_len)


    src_max = max(dest_wave)
    src_min = min(dest_wave)

    print(f'slicelength: {bucketsize}, valrange {src_min}/{src_max}')

    #dest_wave = dest_wave.repeat(2)

    im, warn = wave_to_cqpowergram(dest_wave, fbins, hops=hops, spectpow=spectpow, bins_per_octave=bins_per_octave, fmin=fmin)
    print (f"saving spectrum img: {dest_file}.png, warnings[(low/clp)] {warn}, size {im.width}x{im.height}")
    im.save(dest_file + ".png")

    return warn






