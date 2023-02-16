import mmm_WaveToolbox as wtb


src_len = 22500
size = 64
src_sr = 48000

slice_len = src_len / size
ds_rate = (size / slice_len)
out_sr = int(src_sr * ds_rate)

print (f'outsr: {out_sr}')

#CONVERT SINGLE WAV PNG TO WAVEFILE
if (0):
    ep = 11
    for i in range(4):
        infn = f'x_{i}_{ep}'
        outfn = f'xy{ep}_{i}'
        sdir = "E:/work/2020/MA/experiments/TF_tuts2/images/" + infn
        ddir = "E:/work/2020/MA/experiments/DATASETS/beats/output/" + outfn
        wtb.save_waveimage_to_wave(sdir, ddir + ".wav", out_sr)

#CONVERT RANGE OF WAVPNGs TO WAVEFILES
if (1):
    for ep in  range(15,42,4):
        for i in range(4):
            infn = f'x_{i}_{ep}'
            outfn = f'xx{ep}_{i}'
            sdir = "E:/work/2020/MA/experiments/TF_tuts2/images/" + infn
            ddir = "E:/work/2020/MA/experiments/DATASETS/beats/output/" + outfn
            wtb.save_waveimage_to_wave(sdir, ddir + ".wav", out_sr)


