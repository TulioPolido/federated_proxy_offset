# Jupyter Notebooks and python script for the Viva Bem Software Deliveries
This model results from the Federated Learning (FL) research line. In this deliverable, we present a federated simulation model for activity recognition. The model uses three accelerometer axis (x, y and z) transformed into recurrence plots as input.
The objective of this research is to explore new solutions to HAR problems that demonstrate strong performance and contribute new knowledge to the field.

## Getting Started

**The Viva Bem software delivery** is a Jupyter Notebook. This material encapsulates all the process steps and important details.

Before you running our code, you must go to the [Bart repository](aiunica-generic-local/data_delivery/6.0.0/A7/viva_bem_dataset/data_flmodel/) and download the corresponding dataset in the path: data/.

[Jupyter notebooks](https://jupyter.org/) allow a high-level of interactive learning, as code, text description, and visualization combined in one place.

## Prerequisites

You will require Jupyter notebook to run this code.

Install Jupyter by
download from the [official website](https://jupyter.org/), or
using automatic installer of, e.g., [Anaconda](https://www.anaconda.com/distribution/).

Start the Jupyter server.


### Installation of the Conda Environment

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
Open command prompt
Run the following code
`conda update --all`
`conda create --name <env_name> python=3.9.20 --file requirements.txt`
`conda activate <env_name>`

To finish the installation of jupyter extensions, run:
`jupyter contrib nbextension install --user`

Select the folder with notebooks and run `jupyter notebook`

### Dependencies

_libgcc_mutex=0.1
_openmp_mutex=4.5
asttokens=2.4.1
ca-certificates=2024.9.24
comm=0.2.2
contourpy=1.3.0
cycler=0.12.1
debugpy=1.8.7
decorator=5.1.1
exceptiongroup=1.2.2
executing=2.1.0
filelock=3.16.1
fonttools=4.54.1
fsspec=2024.10.0
importlib-metadata=8.5.0
importlib-resources=6.4.5
ipykernel=6.29.5
ipython=8.18.1
jedi=0.19.1
jinja2=3.1.4
joblib=1.4.2
jupyter_client=8.6.3
jupyter_core=5.7.2
kiwisolver=1.4.7
krb5=1.21.3
ld_impl_linux-64=2.40
libedit=3.1.20230828
libffi=3.4.4
libgcc=14.2.0
libgcc-ng=14.2.0
libgomp=14.2.0
libsodium=1.0.20
libstdcxx=14.2.0
libstdcxx-ng=11.2.0
markupsafe=3.0.2
matplotlib=3.9.2
matplotlib-inline=0.1.7
mpmath=1.3.0
ncurses=6.4
nest-asyncio=1.6.0
networkx=3.2.1
numpy=2.0.2
nvidia-cublas-cu12=12.4.5.8
nvidia-cuda-cupti-cu12=12.4.127
nvidia-cuda-nvrtc-cu12=12.4.127
nvidia-cuda-runtime-cu12=12.4.127
nvidia-cudnn-cu12=9.1.0.70
nvidia-cufft-cu12=11.2.1.3
nvidia-curand-cu12=10.3.5.147
nvidia-cusolver-cu12=11.6.1.9
nvidia-cusparse-cu12=12.3.1.170
nvidia-nccl-cu12=2.21.5
nvidia-nvjitlink-cu12=12.4.127
nvidia-nvtx-cu12=12.4.127
openssl=3.3.2
packaging=24.1
pandas=2.2.3
parso=0.8.4
pexpect=4.9.0
pickleshare=0.7.5
pillow=11.0.0
pip=24.2
platformdirs=4.3.6
prompt-toolkit=3.0.48
psutil=6.1.0
ptyprocess=0.7.0
pure_eval=0.2.3
pygments=2.18.0
pyparsing=3.2.0
python=3.9.20
python-dateutil=2.9.0
python_abi=3.9
pytz=2024.2
pyzmq=26.2.0
readline=8.2
scikit-learn=1.5.2
scipy=1.13.1
seaborn=0.13.2
setuptools=75.1.0
six=1.16.0
sqlite=3.45.3
stack_data=0.6.2
sympy=1.13.1
threadpoolctl=3.5.0
tk=8.6.14
torch=2.5.1
torchsummary=1.5.1
torchvision=0.20.1
tornado=6.4.1
traitlets=5.14.3
triton=3.1.0
typing_extensions=4.12.2
tzdata=2024.2
wcwidth=0.2.13
wheel=0.44.0
xz=5.4.6
zeromq=4.3.5
zipp=3.20.2
zlib=1.2.13


### Running Jupyter Notebook


To run Jupyter Notebook, open an Anaconda prompt and navigate to the directory in which the cloned repository is. By default Jupyter is setup to open your home directory.

 Once you are in the correct folder, run Jupyter:

**jupyter notebook** or **jupyter-notebook** (depending on your operating system).

This should open Jupyter Notebooks in a browser window. On occasion, Jupyter may not be able to open a window and will give you a URL to paste in your browser. Please do so, if required.

### Running the python script

To run the Python script, open an Anaconda prompt and navigate to the directory where the cloned repository is located. Next, go into the <code>src</code> folder to run the following command with the parameter values that gave us the best performance:

python3 main.py
