import launch
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "video_device",
            default_value="/dev/video0",
            description="input video source"
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="./src/EdgeYOLO-ROS/weights/edgeyolo_tiny_coco_416x416.onnx",
            description="edgeyolo model path."
        ),
        DeclareLaunchArgument(
            "conf_th",
            default_value="0.30",
            description="edgeyolo confidence threshold."
        ),
        DeclareLaunchArgument(
            "nms_th",
            default_value="0.45",
            description="edgeyolo nms threshold"
        ),
        DeclareLaunchArgument(
            "imshow",
            default_value="true",
            description=""
        ),
    ]
    nodes = [
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            parameters=[{
                "video_device": LaunchConfiguration("video_device"),
                "image_size": [640,480]
            }]),
        Node(
            package='edgeyolo_ros_py',
            executable='edgeyolo_onnx_node',
            parameters=[{
                "model_path": LaunchConfiguration("model_path"),
                "conf_th": LaunchConfiguration("conf_th"),
                "nms_th": LaunchConfiguration("nms_th"),
                "imshow": LaunchConfiguration("imshow"),
            }],
        ),
    ]


    return launch.LaunchDescription(
        launch_args + nodes
    )
