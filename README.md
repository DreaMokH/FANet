# FANet

The repository is for the paper "[FANet: Features Adaptation Network for 360$^{\circ}$ Omnidirectional Salient Object Detection](https://ieeexplore.ieee.org/document/9211754)", IEEE Signal Processing Letters, 2020.

## Codes

* The code is trained and tested with Python3.7, PyTorch1.6 and CUDA10.1. The required packages include ```PyTorch```, ```torchvision```, ```Numpy```, ```SciPy```, ```PIL```, ```OpenCV``` and ```Tensorboard```.

* The pretrained weight of backbone ResNet-50 can be downloaded from official PyTorch [link](https://download.pytorch.org/models/resnet50-19c8e357.pth). The datasets can be downloaded from [360-SOD](http://cvteam.net/projects/JSTSP20_DDS/DDS.html) and [F-360iSOD](https://github.com/PanoAsh/F-360iSOD).

* The paths in the [config.yaml](./config.yaml) should be reset when you need to [train](./train.py) the model or [predict](./inference.py) the saliency maps.

* The eval code can be found in [http://dpfan.net/](http://dpfan.net/).

## Results

Our results can be downloaded at [results](./SalMap.zip) or [BaiduYun CloudDrive](https://pan.baidu.com/s/1RfjZM73D472W6KO5n-8v1w)(Extraction Code: alab).

## Citation

If you find this repo useful, please cite the following paper:

```bibtex
@ARTICLE{Huang_2020_SPL,
  author={M. {Huang} and Z. {Liu} and G. {Li} and X. {Zhou} and O. {Le Meur}},
  journal={IEEE Signal Processing Letters}, 
  title={FANet: Features Adaptation Network for 360$^{\circ}$ Omnidirectional Salient Object Detection}, 
  year={2020},
  volume={27},
  pages={1819-1823},
  doi={10.1109/LSP.2020.3028192}}
```

## Contact

Any questions, please contact [huangmengke@shu.edu.cn](huangmengke@shu.edu.cn).

