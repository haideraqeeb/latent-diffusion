# latent-diffusion

## To install dependencies run the command:
```
pip install -r requirements.txt
```
## Download weights and tokenizer files:
1. Download ```vocab.json``` and ```merges.txt``` from https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/files and save them in the data folder in the root directory.
2. Download the ```v1-5-pruned-emaonly.ckpt``` and save it in the data folder from the same link.

## To use the model:
1. Go to the ```script``` folder.
2. Change the ```prompt``` to the image you want to generate(you can also add ```uncond_prompt``` for the unconditional prompt if you want to exclude something).
3. If you have CUDA support or MPS support then change the ```ALLOW_CUDA``` or ```ALLOW_MPS``` constant to ```True``` for faster processing.

## Special Thanks
Special thanks to the following repositories:

1. https://github.com/CompVis/stable-diffusion/
2. https://github.com/divamgupta/stable-diffusion-tensorflow
3. https://github.com/kjsman/stable-diffusion-pytorch
4. https://github.com/huggingface/diffusers/
5. https://github.com/hkproj/pytorch-stable-diffusion/tree/main