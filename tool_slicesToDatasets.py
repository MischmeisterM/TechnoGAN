import mmm_WaveToolbox as wtb
import os

CQ_BINS = 30
CQ_BASE = 32.7


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
    #wtb.saveWaveImage2Wav(fn + "_dest", fn + "_imgwav.wav", 6000)

def convert_single_cq():
    # TEST CONVERSION AND RECONVERSION ON SINGLE FILE
    fn = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/test/2022_160bpm_full01_324"
    wtb.convert_and_save_cq(fn + ".wav", fn + "_dest.wav", fn + "_ctrl.wav", fn + "_spect", fn + "_wavimg", 256, beats=4, spectpow=3, bpo=CQ_BINS, fmin=CQ_BASE, hops=4)
    wtb.save_cqpowergram_image_to_wav(fn + "_spect", fn + "_reconv.wav", 44100, spectpow=3, hops=4, bins_per_octave=CQ_BINS, fmin=CQ_BASE)

    # wtb.saveSpectImageLong2Wav(fn + "_spect", fn + "_imgwav.wav", 6000)
    #wtb.saveWaveImage2Wav(fn + "_dest", fn + "_imgwav.wav", 6000)

convert_dir()
#convert_single()
#convert_dir_cq()
#convert_single_cq()

