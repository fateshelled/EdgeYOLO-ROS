# EdgeYOLO-ROS

[EdgeYOLO](https://github.com/LSH9832/edgeyolo) + ROS 2 Inference DEMO

## Supported List
- ONNXRuntime C++ and Python
- TensorRT C++
- OpenVINO C++

※ TFLite inference demo code has been uploaded but not tested.

## Requirements
- ROS 2 Foxy or Humble
- OpenCV
- ONNXRuntime or TensorRT or OpenVINO
- bbox_ex_msgs
- v4l2-camera (for Webcam Demo)

## Build
```bash
cd ros2_ws/src
git clone https://github.com/Ar-Ray-code/bbox_ex_msgs
git clone https://github.com/fateshelled/EdgeYOLO-ROS
cd ../

# for C++
colcon build --symlink-install --packages-up-to edgeyolo_ros_cpp bbox_ex_msgs

# for python
colcon build --symlink-install --packages-up-to edgeyolo_ros_py bbox_ex_msgs
```

## Model Download
- Model download from [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/356_EdgeYOLO/download_nopost.sh)
  - use nopost model

## Webcam DEMO
### ONNXRuntime
```bash
# Run C++ Node
ros2 launch edgeyolo_ros_cpp edgeyolo_onnxruntime.launch.py \
      model_path:=edgeyolo_tiny_coco_416x416.onnx

# Run Python Node
ros2 launch edgeyolo_ros_py edgeyolo_onnxruntime.launch.py \
      model_path:=edgeyolo_tiny_coco_416x416.onnx
```

### OpenVINO
```bash
# model convert
mo.py --input_model edgeyolo_tiny_coco_416x416.onnx \
      --input_shape [1,3,416,416] \
      --output_dir ./openvino_model

# Run C++ Node
ros2 launch edgeyolo_ros_cpp edgeyolo_openvino.launch.py \
      model_path:=./openvino_model/edgeyolo_tiny_coco_416x416.xml
```

### TensorRT
```bash
# model convert
trtexec --onnx=edgeyolo_tiny_coco_416x416.onnx \
        --saveEngine=edgeyolo_tiny_coco_416x416.engine

# Run C++ Node
ros2 launch edgeyolo_ros_cpp edgeyolo_tensorrt.launch.py \
      model_path:=edgeyolo_tiny_coco_416x416.engine
```

## Reference
- [EdgeYOLO](https://github.com/LSH9832/edgeyolo)
- [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS)
- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
