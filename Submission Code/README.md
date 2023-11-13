
> Firstly you need to install all package from requirements:
```
pip3 install ...
```
> You can configure the hyperparameters by setting them in the config.py file, as follows:

```python
lr = 0.001
batch_size = 64
epoch  = 30
save_path = "model.pth"
image_path = ""  #image dataset root path
model = "resnet"
```

1. lr : learning rate
2. batch_size : batch size
3. save_path : output path to save model parameters
4. model : if it is resnet, run resnet, if it is resnext, run resnext

## Run model
using following command in your shell

```
python models.py
```