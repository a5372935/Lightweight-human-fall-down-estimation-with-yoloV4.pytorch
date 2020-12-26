# Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose & Fall detection of human pose is improved by yolov4

My project focuses on providing a solution to detect human falls in real time and to send a warning message during the fall, rather than the result of the fall.

# 2020/12/26 Modified by [YOU-JEN SHIEH](https://github.com/a5372935)

## Environment
* windows10, python 3.6.10, torch 1.4 
* CPU : i5-9400f , CPU usage up to 100%
* GPU : GTX1650 , GPU only use 1G RAM

## Download weights
* Pre-trained on COCO model is available at: [checkpoint_iter_370000.pth](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth), it has 40% of AP on COCO validation set (38.6% of AP on the val *subset*)
* Pre-trained on COCO-Train2017 [yolov4_tiny_weights_coco.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_coco.pth) and move it to model_data, image input size 416 * 416 

## Python Demo <a name="python-demo"/>

We provide python demo just for the quick results preview. Please, consider c++ demo refer [Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) for the **best performance**. To run the python demo from a webcam or videos:
* `python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video 0`
* `python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video <video_path>`

## IF you want send alarm-email when the human fall 
Please uncomment lines 205 to 215 of demo.py & Modified your email, target email, app password in line 78 & 79

## Reference
### If you want retrain model please refer here
* Lightweight OpenPose : [Daniil-Osokin](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
* yolov4-tiny-pytorch : [Bubbliiiing](https://github.com/bubbliiiing/yolov4-tiny-pytorc)
