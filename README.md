# Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose & Fall detection of human pose is improved by yolov4

My project focuses on providing a solution to detect human falls in real time and to send a warning message during the fall, rather than the result of the fall.

# 2020/12/26 Modified by [YOU-JEN SHIEH](https://github.com/a5372935)

## Python Demo <a name="python-demo"/>

We provide python demo just for the quick results preview. Please, consider c++ demo for the best performance. To run the python demo from a webcam:
* `python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video 0`
* `python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video <video path>`

## IF you want send alarm-email when the human fall 
Please uncomment lines 205 to 215 of demo.py & Modified your email, target email, app password in line 78 & 79

## Reference
* Lightweight OpenPose : [Daniil-Osokin](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
* yolov4-tiny-pytorch : [Bubbliiiing](https://github.com/bubbliiiing/yolov4-tiny-pytorc)
