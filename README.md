# DA-CenterNet

PyTorch Implementation of Domain Adaptive CenterNet([Object as Points](https://arxiv.org/abs/1904.07850))
- This repo adds only image-level adaptation component in [Domain Adaptive Faster R-CNN for Object Detection in the Wild](https://arxiv.org/abs/1803.03243)
- You don't need to bulid some cpp code to use Deformable Convolution used in CenterNet.

## Performance

## On Cityscape &rarr; Cityscape Foggy
DA Faster R-CNN is with only image-level adaptation component.

|Repo|  mAP    |  
|:-------------:|:------:|
|DA Faster R-CNN(image-level)| 25.7   | 
| **This Repo** | 24.6   |

## Training

### Cityscape &rarr; Cityscape Foggy

```
python train.py --source ./data/cityscape.yaml --target ./data/cityscape_foggy.yaml --batch-size 4 --total-epoch 50
```

If your gpu memory is too lower to train the model, you should try to reduce batch-size.


## Evaluation

### Cityscape
```
python eval.py --data ./data/cityscape.yaml --weights your_model.pth --batch-size 8 --flip
```

### Cityscape Foggy
```
python eval.py --data ./data/cityscape_foggy.yaml --weights your_model.pth --batch-size 8 --flip
```

## Reference

- https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch