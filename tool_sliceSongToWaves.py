# helper script that takes a directory of .wav files and slices each file into individual samples

import mmm_WaveToolbox as wtb
import os

src_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/tracks_160bpm/"
dest_wav_dir = "D:/work/2022/TechnoGAN/networks/Source/2022_160bpm/slices_160bpm_4beat/"
#src_filename = "col_128bpm_05"
#slice_wave_and_save(src_dir + src_filename, dest_wav_dir + src_filename, 22500, 3)
for file in os.listdir(src_dir):
    if file.endswith(".wav"):
        src_filename = os.path.splitext(file)[0]
        # slice_wave_and_save(<sourcefile>, <destinationfile>, <BPM>, jump = <store only every Nth slice> beats = <beats per slice>)
        wtb.slice_wave_and_save(src_dir + src_filename, dest_wav_dir + src_filename, 160, beats=4)

