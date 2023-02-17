# functions to create training data from sliced samples
#
# use the convert_singe* functions to test conversion parameters:
#   convert a single wave to spectrum image and reconvert to wave
#   to test if transformation and retransformation work - the resulting wave should
#   sound like the source as much as possible
#   the same functions and params should then be used when
#   creating images for the dataset and generating wavs through inference
#
# then convert a whole directory of samples to training images
#
# the parameters spectWidth and beats define the pixel size of the resulting images
#   size is the number of frequency bins
#   beats is the length of the sample in quarter notes
#       in theory a different number can be used, smaller numbers mean
#       smaller images and less quality
#       (basically the wavefile will be downsampled to beats*8192 samples to get a
#       proper power of 2 spectrum size after the transformation)


import mmm_WaveToolbox as wtb
import os

# parameters for constant-q transformation - need to be reused when retransforming generated images
CQ_BINS = 30        # bins per octave (highest frequency is clamped by spectrum width)
CQ_BASE = 32.7      # frequency of lowest bin (in Hz)


# CONVERT WHOLE DIRECTORY OF WAVEFILES TO DOWNSAMPLED WAVES AND SPECTRUM-PNGs
def convert_dir():
    src_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/slices_160bpm_4beat/"
    dest_wav_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/dataset/"
    dest_wav_img_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/dataset/"
    dest_spect_img_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/dataset/"
    fileType = ".wav"

    wtb.convert_and_save_dir(src_dir, dest_wav_dir, dest_wav_img_dir, dest_spect_img_dir, fileType, 256, beats=8, spectpow=3)
    #wtb.resample_dir(src_dir, dest_wav_dir, samples = 16384)

def convert_dir_cq():
    src_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/slices_160bpm_2beat/"
    dest_wav_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/dataset/"
    dest_wav_img_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/dataset/"
    dest_spect_img_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/dataset/"
    fileType = ".wav"

    wtb.convert_and_save_dir_cq(src_dir, dest_wav_dir, dest_wav_img_dir, dest_spect_img_dir, fileType, 256, beats=4, spectpow=3, bins_per_octave=CQ_BINS, fmin=CQ_BASE)
    #wtb.resample_dir(src_dir, dest_wav_dir, samples = 16384)

def convert_single():
    # TEST CONVERSION AND RECONVERSION ON SINGLE FILE
    fn = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/test/2022_160bpm_full01_324"
    wtb.convert_and_save(fn + ".wav", fn + "_dest.wav", fn + "_ctrl.wav", fn + "_spect", fn + "_wavimg", 256, beats=8, spectpow=3)
    wtb.save_powergram_image_to_wav(fn + "_spect", fn + "_reconv.wav", 44100, spectpow=3, hops=4)

    # wtb.saveSpectImageLong2Wav(fn + "_spect", fn + "_imgwav.wav", 6000)
    # wtb.saveWaveImage2Wav(fn + "_dest", fn + "_imgwav.wav", 6000)

def convert_single_cq():
    # TEST CONVERSION AND RECONVERSION ON SINGLE FILE
    fn = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/test/2022_160bpm_full01_324"
    wtb.convert_and_save_cq(fn + ".wav", fn + "_dest.wav", fn + "_ctrl.wav", fn + "_spect", fn + "_wavimg", 256, beats=4, spectpow=3, bpo=CQ_BINS, fmin=CQ_BASE, hops=4)
    wtb.save_cqpowergram_image_to_wav(fn + "_spect", fn + "_reconv.wav", 44100, spectpow=3, hops=4, bins_per_octave=CQ_BINS, fmin=CQ_BASE)

    # wtb.saveSpectImageLong2Wav(fn + "_spect", fn + "_imgwav.wav", 6000)
    # wtb.saveWaveImage2Wav(fn + "_dest", fn + "_imgwav.wav", 6000)



convert_dir()
#convert_single()

#convert_dir_cq()
#convert_single_cq()

