# PaliGemma 2 Caption Batch
This tool uses the [PaliGemma 2]([https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48)) model from Google to caption images in an input folder.

It's a very fast and fairly robust captioning model that can produce good outputs with custom prompting and additional functionality.

## Requirements
* Python 3.10 or above.
  * Tested with 3.10.

* Cuda 12.1.
  * It may work with other versions. Untested.
 
To use CUDA / GPU speed captioning, you'll need ~21GB VRAM or more.

On an Nvidia 3090 (24gb VRAM), you can use a batch size of 7. Possibly 8.

## Setup
1. Create a virtual environment. Use the included `venv_create.bat` to automatically create it. Use python 3.10 or above.
2. Install the libraries in requirements.txt. `pip install -r requirements.txt`. This is done by step 1 when asked if you use `venv_create`.
3. Install [Pytorch for your version of CUDA](https://pytorch.org/). It's only been tested with version 12.1 but may work with others.
4. Open `batch.py` in a text editor and change the BATCH_SIZE = 7 value to match the level of your GPU.


## How to use
1. Activate the virtual environment. If you installed with `venv_create.bat`, you can run `venv_activate.bat`.
2. Run `python batch.py` from the virtual environment.

This runs captioning on all images in the /input/-folder.

## Example
![birb](https://github.com/user-attachments/assets/4a1aa16c-e805-4e3f-bafe-4044aa6682e9)

> A front view of a fluffy cartoon penguin standing on white snow. The bird is round and looks like it's smiling because its mouth is open, showing its pink tongue sticking out. It has dark eyes, rosy cheeks, orange feet, an orange beak and black pupils. Snowballs are lying in front of it. Two tall trees covered in white fur are behind to the left and right of the penguin in the distance. Falling snowflakes are around them and little glowing lights can be seen floating through the air as well. A darker blue sky is visible above all of them
