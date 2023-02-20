# TechnoGAN

## Introduction
TechnoGAN is a Neural Network that generates short loopable techno-like samples. 

The project was initially part of my master thesis while studying composition with focus on Computer music.

While I have some programming background, I am all but a professional coder, thus I'm sure the code is in many ways terribly written, badly documented and not conforming to any standards. I apologize for that, but as i actually just learned python in the course of this project, I think i got quite some way so far. 
The next steps in developing the project further require extensive experience and knowledge in software development and algorythms that I simply don't have the capacity to aquire all on my own.

#### Listen / Demo Videos:
* 'smol' standalone interface version: https://www.youtube.com/watch?v=YlCn5rSRvO8
* Max 4 Live sampler interface: https://www.youtube.com/watch?v=-VWpd4WY7DQ
* full Max 4 Live performance interface: https://www.youtube.com/watch?v=YGJx9dgsVYM

Project webpage: https://www.mischmeisterm.com/projects/technogan/

For questions, comments, requests: mischa@mischmeisterm.com

---
## Project structure
### Basic structure
While the original input and final output is audio data, The heart of TechnoGAN is a Generative Adversarial Network that works on grayscale images.

Audio samples are transformed into spectrograms that serve as training data. Vice versa, Spectrograms are created from the Network and then transformed back into 1-dimensional waveforms.
Currently, phase information is discarded to keep the data to a single 2d-image. In retransformation, missing phase data is approximately reconstructed with the griffin-lim algorythm.
This allows to work with a (by now) rather simple GAN model.

Tensorflow in connection with CUDA is used For training the model. For inference, the trained model is converted into ONNX format and the ONNXRuntime library is used for a slightly slimmer and more compatible application.

There also is an OSC-interface for actually jamming, composing and performing with the trained model and several Max and Max for Live applications that offer a graphical interface and some mixing and arrangement tools.
See the project webpage for downloads: https://www.mischmeisterm.com/projects/technogan/

---
### TL:DR
* Training the model: 
  * Run any of the `GAN_Spect_*.py` scripts or create your own.
  * Make sure to point the path variables in there to the correct locations.
* Running Inference via OSC:
  * Run `mmm_liveOSCServer`.
    * -> imports `mmm_ganGenerator.py`
      * -> imports the inference script (the `GAN_Spect_*.py` used to train or the 'mmm_onnxInference_*.py that points to the right model')
      
---
### GAN Model
The root for the Network model and script is a rather basic Tensorflow tutorial. (https://www.tensorflow.org/tutorials/generative/dcgan)

For each different input format (pixel size of training dataset), a script handles network shapes and training. The script is also able to be used for inference later. During training, the script also continuously generates inference samples to document and monitor progress.

The Generator network starts with a seed aray of 64 values. This (rather arbitrary amount) could be modified but the change would also be made in the OSC-interface and its connected client applications.

There are several models for different pixel sizes included. Also an experimental one that works on spectrograms created with a constant-q transformation. 

#### How to create your own Model:
* Copy one of the existing model scripts `GAN_Spect_*.py`.
* Change the project directory variables to your desired locations.
* Adapt `make_generator_model()` and `make_discriminator_model()` to reflect the pixel size of your training data.
* Configure other constants to your liking.

After finishing training you can convert the model into .onnx format with `convert_tf2onnx.py`.
Make sure you also copy and adapt one of the `mmm_onnxInference*.py` scripts to run inference on the converted .onnx model (mainly IMG_WIDTH and IMG_HEIGHT).

---
### Inference interface
`mmm_liveOSCServer.py` starts a process that listens to OSC messages on 10102 and sends responses to 127.0.0.1:10101.

`mmm_ganGenerator.py` contains all commands that the OSC server can handle and the name of the actual generator script used to run inference. The scripts used to train the Model can also be used here, as they should implement all necessary functionality to run inference as well. 

The OSC messages have to be sent to the server in following format: `/cc<command> <parameters>`
#### List of currently implemented OSC commands:
* `cc//ìsAlive` check if the OSC-server is running and responding, should return `/alive`.
* `/cc/initGenerator` initializes generator (should be done once). Returns `/msg` if succesfull or already initialized.
* `/cc/set_target_sr <int>` sets samplerate to be stored with newly generated wavefiles. returns confirmation `/msg`.
* `/cc/set_target_dir <string>` sets destination directory to store newly generated wavefiles in. returns confirmation `/msg`.
* `/cc/set_target_dir` returns current destination directory in `/msg`. 
* `/cc/generate_single_wav <float[64] seed>` generates sample with given seed parameters (an array of 64 floats). Returns filename and generation time in `/wav`. Mostly used for testing purposes. 
* `/cc/generate_single_wav_id <int id> <float[64] seed>` generates sample with given seed parameters (an array of 64 floats). Returns id, filename and generation time in `/wav_id`.
  * Since most client apps use several sample slots to load the generated sample, the ID is used to remember which slot actually requested the generation of the current sample. 
* `/cc/generate_single_wav_id_img <int id> <float[64] seed>` generates sample with given seed parameters (an array of 64 floats). Additionaly stores the generated spectrum as .png image for debug or visualisation purposes. Returns id, filenames and generation time in `/wav_id_img`.
* `/cc/generate_random_fade <int steps> <float morph> <float[64] startseed>` generates <steps> samples appended in a single wavefile, starting with <startseed> and continuously randomizing(drunken walk) the startseed in the range of <morph>. Returns filename and generation time in `/wav`. Mostly used for testing purposes.
* `/cc/generate_fade <int steps> <float[64] startseed> <float[64] endseed>` generates <steps> samples appended in a single wavefile, starting with <startseed> and lerping to <endseed> to calculate seeds in between the steps. Returns id, filename and generation time in `/wav_id`.
* `/cc/generate_pingpong <int steps> <float[64] startseed> <float[64] endseed>` Same as generate_fade but interpolates from <startseed> to <endseed> back to <starseed> to create a looping mutation of the seed. Returns id, filename and generation time in `/wav_id`.

Responses are always prefixed with `/ccret`.
#### List of currently expectable response commands:
* `/ccret/msg <string msg>` a message to display in the client for debugging.
* `/ccret/alive` tells the client that the server is up and running.
* `/ccret/init` tells potential clients that the server is starting up (or has been restartet)
* `/ccret/wav <string wavfile> <string dbgmess>` lccation of generated file that has been requested via `/cc/generate_single_wav` (and an additional message string for debug). 
* `/ccret/wav_id <int id> <string wavfile> <string dbgmess>` id and location of sample requested with `..._wav_id` commands (and an additional message string for debug).
* `/ccret/wav_id_img <int id> <string wavfile> <string imagefile> <string dbgmess>` id and file locations of sample requested with `..._wav_image_id` commands (and an additional message string for debug).


---
## How to prepare a training dataset:

### tools for dataset creation
`tool_*.py` contain a few handy scripts to convert audiofiles into processable datasets.

---
### Step-by-Step guide to create your own set of training data:
1. Curate audiofiles that you would like in your training set:
   * Songs should be roughly in the same tempo (and genre maybe, but that's up to you).
   * Rhythmic and atonal sounds work best. Melodies and tonal structures don't transform and train well with the current models. 
   * Think about the implications of using possibly copyrighted material in your datasets.

2. Conform tracks to same tempo
   * use your DAW of choice to conform the tracks to exactly the same tempo
   * make sure it is really exact and transients don't start to slip into the previous beat over time. This can be a problem with music created with analogue equipment or digitized from analogue media like tape or vinyl.

3. Slice tracks into same-length samples
   * you can use `tool_sliceSongToWaves.py` for that. Make sure to give it the right BPM in the parameters.
   * for the model I trained, I used about 5500 single samples - more should be better. 

4. transform wave-slices to greyscale images
   * use one of the scripts in `tool_slicesToDatasets.py`
   * check converting and reconverint a single file to see if it works
   * if you fiddle with the parameters, make sure that the image dimensions conform to your training model.
   
---
## How to train the model:
When a model script is run as the main process, the model is trained. After a few iterations example waveforms and images are created to monitor progress. Every few iterations the model is saved and training progress is saved in a checkpoint. When the script is restarted, training resumes from the last checkpoint.

Keyboard shortcuts during training:
* Ctrl-Shift-P - Pause/Resume training without halting the script.
* Ctrl-Shift-S - Save model and progress and create a resume checkpoint after current iteration is finished.

### Requirements/Dependencies
To make use of your GPU while training (__highly recommended__), install:
 - nVidia CUDA toolkit
   - https://developer.nvidia.com/cuda-downloads
 - nVidia CUDAnn library
   - https://developer.nvidia.com/cudnn

Versions I used that worked for me (new versions might require modifications/refactoring in the scripts):
tensorflow 2.10.0 (tensorflow python package)
CUDA v11.7
cudaNN 11.3

---
## How to use a trained model:
After training, you can convert the model to the ONNX file format that is a bit easier to handle and doesn't require the bulky tensorflow library (requires OnnxRuntime instead).

1. Converting to .onnx
   - To convert the model into ONNX format, use the script in `convert_tf2onnx.py`.
2. Create an inference script
   - Copy one of the `mmm_onnxInference_*.py` scripts
     - adapt the parameters (mainly `IMG_WIDTH` and `IMG_HEIGHT`) to your model.
     - reference the right model file in `GENERATORSAVELOCATION`
   - reference (import) the new inference script in `mmm_ganGenerator.py`
3. run `mmm_liveOSCServer.py`

---
### Freezing the generator app:
By freezing the generator app you can create a standalone application that should run on any machine without needing to install anything else.

#### *Windows:*
Use setup_cx_win.py to freeze the generator script into a console app. 

**After freezing, you have to manually copy** the '\_soundfile_data' folder into your builds \lib folder (most likely found in '\venv\Lib\site-packages\_soundfile_data'). 
Not sure why cx-freeze doesn't catch this one, and I'm pretty sure there is a prettier solution for this.  

#### *Mac:*
Remove the slash/backslash replacement for the temp dir in ```TARGET_DIR = TARGET_DIR.replace("\\", "/")``` 'in mmm_ganGenerator.py'.

Use setup_cx_mac.py to freeze the generator script into a console app.

For now, the main script for Mac is 'mmm_ganGenerator.py', there is no proper wrapper for the console app as I don't know how to do this. 

**After freezing, you have to manually copy** the '\_soundfile_data' folder into your builds \lib folder (most likely found in '\venv\Lib\site-packages\_soundfile_data'). 
Not sure why cx-freeze doesn't catch this one, and I'm pretty sure there is a prettier solution for this.  

---
## TODOs and next steps
### freezing and codesigning server app for Mac/M1
While I somehow managed to generate a running freeze of the generator app on Mac, there are still a lot of issues:
* The app needs to be properly signed with xCode so the user isn't confronted with hundreds of warnings for every python component after downloading the app.
* There seem to be lots of compatibility issues, where the app won't run at all, especially with newer M1 Macs.
* Right now the app is started as a pure console app that needs to stay open, there is probably a better way to put the process a bit more into the background.

### creating a max external for inference
In the long run, the generator app should go away alltogether and be integrated in the interface app (i.e. Max 4 live Plugin or VST). So far I have tried to create a Max external with Max's min devkit (http://cycling74.github.io/min-devkit/
) and use the OnnxRuntime c++ library to run inference there. Sadly I couldn't get to run both things together (I didn't get around to learn cmake yet). Another major issue with this goal is, that the Griffin-Lim algorythm from librosa would need to be re-implemented entirely in c++.

### refactor hacky code and approximated transformations
If you have read this far, you've probably figured out that I'm not a professional programmer. Much of the code used in this project looks that way because after some experimentation I got it to do what I needed it to do and not because it is the way such things should usually be done.
* Many processes are crude aproximations and might not be mathimatically or scientifically correct.
* Many things could surely be done in a more performant or clear way.
