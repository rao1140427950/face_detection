## Face detection using Keras and Tensorflow 2.x

### Backbones:
- ResNet
- ResNet + CBAM

### Detectors:
- SSD
- SSD + FPN

### Known Issues:
- OOM while training. (Fixed)
- Still OOM while training with a large batch size or image size, caused by generating anchor boxes.

### TODOs:
- Optimize input data pipeline. (Done)
- How to detect small faces.
- Add data argumentation methods. (Done)
- Analyse distribution of bboxes in WiderFace and optimize `scales` and `aspect_ratios_per_layer`. (Done)
- Multi-scale training.

### Logs:
- Optimize input data pipeline. Now data generator can be seperated from training and run on a different device. Fix OOM while training.
- Change `scales` and `aspect_ratios_per_layer` in `SSD_CONFIG` to suit small faces.
- Add random resize data argumentation method.
- Add attention mechanism (CBAM) in ResNet backbone.
- Add FPN to SSD detector.
- Analyse distribution of bboxes in dataset and adjust `scales` and `aspect_ratios_per_layer`.
- Training images now are resized while keeping ratios.

### Test results:
![image](https://github.com/rao1140427950/face_detection/blob/main/images/test_image_1_results.jpg)
![image](https://github.com/rao1140427950/face_detection/blob/main/images/test_image_2_results.jpg)