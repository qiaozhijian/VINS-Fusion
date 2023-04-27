cmake_minimum_required(VERSION 2.8.3)
project(camera_models)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    )

find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
if (${OpenCV_VERSION} VERSION_GREATER "4.0.0")
    set(OpenCV_DIR  /home/qzj/Downloads/tool/opencv-3.4.10/build)
    FIND_PACKAGE(OpenCV 	REQUIRED )
    set(OpenCV_LIB_DIR ${OpenCV_DIR}/lib)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV library path: ${OpenCV_LIB_DIR}")
    link_directories(${OpenCV_LIB_DIR})
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_libraries(${OpenCV_LIB_DIR}/libopencv_core.so ${OpenCV_LIB_DIR}/libopencv_imgproc.so ${OpenCV_LIB_DIR}/libopencv_highgui.so
            ${OpenCV_LIB_DIR}/libopencv_imgcodecs.so ${OpenCV_LIB_DIR}/libopencv_videoio.so ${OpenCV_LIB_DIR}/libopencv_video.so
            ${OpenCV_LIB_DIR}/libopencv_calib3d.so ${OpenCV_LIB_DIR}/libopencv_features2d.so ${OpenCV_LIB_DIR}/libopencv_flann.so
            ${OpenCV_LIB_DIR}/libopencv_ml.so ${OpenCV_LIB_DIR}/libopencv_objdetect.so ${OpenCV_LIB_DIR}/libopencv_photo.so
            ${OpenCV_LIB_DIR}/libopencv_stitching.so ${OpenCV_LIB_DIR}/libopencv_superres.so ${OpenCV_LIB_DIR}/libopencv_videostab.so)
endif()
# set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


catkin_package(
    INCLUDE_DIRS include
    LIBRARIES camera_models
    CATKIN_DEPENDS roscpp std_msgs
#    DEPENDS system_lib
    )

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

include_directories("include")

add_executable(Calibrations 
    src/intrinsic_calib.cc"
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/PinholeFullCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

add_library(camera_models
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/PinholeFullCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(Calibrations ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(camera_models ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
