#ifndef _EdgeYOLO_CPP_EdgeYOLO_TFLITE_HPP
#define _EdgeYOLO_CPP_EdgeYOLO_TFLITE_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
// #include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
// #include "tensorflow/lite/delegates/gpu/delegate.h"

#include "core.hpp"
#include "coco_names.hpp"

namespace edgeyolo_cpp{
    #define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                  \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

    class EdgeYOLOTflite: public AbcEdgeYOLO{
        public:
            EdgeYOLOTflite(file_name_t path_to_model, int num_threads,
                        float nms_th=0.45, float conf_th=0.3, std::string model_version="0.1.1rc0",
                        int num_classes=80, bool p6=false, bool is_nchw=true);
            ~EdgeYOLOTflite();
            std::vector<Object> inference(const cv::Mat& frame) override;

        private:
            int doInference(float* input, float* output);

            int input_size_;
            int output_size_;
            bool is_nchw_;
            std::unique_ptr<tflite::FlatBufferModel> model_;
            std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
            std::unique_ptr<tflite::Interpreter> interpreter_;
            TfLiteDelegate* delegate_;

    };
} // namespace edgeyolo_cpp

#endif