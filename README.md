## Face detection using Keras and Tensorflow 2.x

### Backbones:
- ResNet

### Detectors:
- SSD

### Known Issues:
- OOM while training. (Fixed)

### TODOs:
- Optimize input data pipeline. (Done)
- How to detect small faces.
- Add data argumentation methods.
- Analyse distribution of bboxes in WiderFace and optimize 'scales' and 'aspect_ratios_per_layer'.

### Logs:
- Optimize input data pipeline. Now data generator can be seperated from training and run on a different device. Fix OOM while training.
- Change 'scales' and 'aspect_ratios_per_layer' in `SSD_CONFIG` to suit small faces.
- Add random resize data argumentation method.