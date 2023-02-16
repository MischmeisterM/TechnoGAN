import mmm_WaveToolbox as wtb
import os

SQUARESIZE = 64



fn = "E:/work/2020/MA/experiments/DATASETS/beats/col128bpm/newtest4"

# wav to image
wtb.convert_and_save(fn + ".wav", fn + "_dest.wav", fn + "_ctrl.wav", fn + "_spect", fn + "_wavimg", size = 128, beats = 2, spectpow=3)
#wtb.save_wave_to_cqpowergram(fn + '.wav', fn + '_cq', hops=4, fbins=128, spectpow=4, slices=64, bins_per_octave=24, fmin=110)

# image to wav

#wtb.save_cqpowergram_image_to_wav(fn + '_cq', fn + '_reconv.wav', sr = 16384, hops = 4, spectpow = 4, bins_per_octave=24, fmin=55 )
wtb.save_powergram_image_to_wav(fn + '_spect', fn + '_reconv.wav', sr = 34840, hops = 4, spectpow = 3 )
#saveSpectImageLong2Wav(fn + "_dest", fn + "_imgwav.wav", 6000)
#saveWaveImage2Wav(fn + "_dest", fn + "_imgwav.wav", 6000)
