import mmm_WaveToolbox as wtb


src_len = 22500
size = 64
src_sr = 48000

slice_len = src_len / size
ds_rate = (2 * size / slice_len)
out_sr = int(src_sr * ds_rate)

print (f'outsr: {out_sr}')

#CONVERT SINGLE WAV PNG TO WAVEFILE
if (1):
    ep = 40
    for i in range(4):
        infn = f'x_{i}_{ep}'
        outfn = f'wz{ep}_{i}'
        sdir = "E:/work/2020/MA/experiments/TF_tuts2/images/" + infn
        ddir = "E:/work/2020/MA/experiments/DATASETS/beats/output/" + outfn
        wtb.save_powergram_image_to_wav(sdir, ddir + ".wav", out_sr)

#CONVERT RANGE OF WAVPNGs TO WAVEFILES
if (0):
    for ep in  range(20,34,3):
        for i in range(4):
            infn = f'x_{i}_{ep}'
            outfn = f'yx{ep}_{i}'
            sdir = "E:/work/2020/MA/experiments/TF_tuts2/images/" + infn
            ddir = "E:/work/2020/MA/experiments/DATASETS/beats/output/" + outfn
            wtb.save_powergram_image_to_wav(sdir, ddir + ".wav", out_sr)


