# TechnoGAN
### Project introduction


## Basic structure
### GAN Model
### Inference interface
### tools for dataset creation


## How to prepare a training dataset:
#### Conform tracks to same tempo
#### Slice tracks into same-length samples
#### transform wave-slices to greyscale images


## How to train the model:
### Requirements/Dependencies


## How to use a trained model:
### Converting to .onnx
### Freezing the generator app:

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

## TODOs and next steps
### freezing server app for Mac/M1
### creating a max external for inference
### refactor hacky code and approximated transformations
