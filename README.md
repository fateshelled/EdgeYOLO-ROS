# EdgeYOLO-ROS

[EdgeYOLO](https://github.com/LSH9832/edgeyolo) + ROS2 Inference DEMO

## Supported List
- ONNXRuntime C++

※ OpenVINO, TensorRT and TFLite inference demo code has been uploaded but not tested.

## Build
```bash
cd ros2_ws/src
git clone https://github.com/Ar-Ray-code/bbox_ex_msgs
git clone https://github.com/fateshelled/EdgeYOLO-ROS
cd ../

colcon build --symlink-install --packages-up-to edgeyolo_ros_cpp
```

## Model Download
- Model download from [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/356_EdgeYOLO/download_nopost.sh)

## Usage
### ONNXRuntime
```bash
ros2 launch edgeyolo_ros_cpp edgeyolo_onnxruntime.launch.py model_path:=edgeyolo_tiny_coco_416x416.onnx
```

## Reference
- [EdgeYOLO](https://github.com/LSH9832/edgeyolo)
- [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS)
- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
