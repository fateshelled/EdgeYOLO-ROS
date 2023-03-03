#ifndef _EdgeYOLO_CPP_EdgeYOLO_HPP
#define _EdgeYOLO_CPP_EdgeYOLO_HPP

#include "config.h"

#ifdef ENABLE_OPENVINO
    #include "edgeyolo_openvino.hpp"
#endif

#ifdef ENABLE_TENSORRT
    #include "edgeyolo_tensorrt.hpp"
#endif

#ifdef ENABLE_ONNXRUNTIME
    #include "edgeyolo_onnxruntime.hpp"
#endif

#ifdef ENABLE_TFLITE
    #include "edgeyolo_tflite.hpp"
#endif


#endif
