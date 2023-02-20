import sys
from cx_Freeze import setup, Executable

sys.path.append("D:\\work\\2020\\MA\\experiments\\python_tests\\test5")

includes = []

# Include your files and folders
#includefiles = ['images/','outwavs/','mmm_liveOSCServer.py', 'mmm_ganGenerator.py', 'main.py', 'GAN_Spect_128x256.py', 'icon.png']
includefiles = ['icon.png', 'data/']
# Exclude unnecessary packages
excludes = ['cx_Freeze','pydoc_data','tkinter']

# Dependencies are automatically detected, but some modules need help.
packages = ["scipy.optimize", "scipy.integrate", 'numpy', 'imageio', 'pythonosc', 'librosa', 'soundfile', 'onnxruntime']

base = None
shortcut_name = None
shortcut_dir = None
#if sys.platform == "win32":
#    base = "Win32GUI"
#    shortcut_name='My App'
#    shortcut_dir="DesktopFolder"

setup(
    name = 'TechnoGAN alpha',
    version = '0.2',
    description = 'TechnoGAN Generator Server',
    author = 'Michael Mayr',
    author_email = '',
    options = {'build_exe': {
        'includes': includes,
        'excludes': excludes,
        'packages': packages,
        'include_files': includefiles}
        }, 
    executables = [Executable('mmm_liveOSCServer.py', 
    base = base, # "Console", base, # None
    icon='icon.png', 
    shortcut_name = shortcut_name, 
    shortcut_dir = shortcut_dir)]
)