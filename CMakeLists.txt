cmake_minimum_required(VERSION 2.8.3)
project(sli_gpd)
 add_compile_options(-std=c++11)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS cmake_modules eigen_conversions tf tf_conversions geometry_msgs message_generation roscpp
  sensor_msgs std_msgs ros_gpg)

# PCL
find_package(Armadillo REQUIRED)
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenCV
find_package(OpenCV)

# custom messages
find_package(message_generation)

#find_library(GENERATOR_LIB_SLI grasp_candidates_generator_ros HINTS "/homeL/demo/ws_grasp/devel/lib/")
find_path(GENERATOR_LIB_SLI_INCLUDE_DIR gpg/grasp.h HINTS "/homeL/demo/ws_grasp/src/ros_gpg/include")
include_directories(${GENERATOR_LIB_SLI_INCLUDE_DIR})
#message("ros_gpg_DIR: " ${GENERATOR_LIB_SLI})
message("ros_gpg_include_DIR: " ${GENERATOR_LIB_SLI_INCLUDE_DIR} )
# CAFFE
IF(NOT EXISTS ${CAFFE_DIR})
  SET(CAFFE_DIR "$ENV{CAFFE_DIR}")
  SET(CAFFE_DIR "/home/andreas/software/caffe/build")
ENDIF()
MESSAGE("CAFFE_DIR: " ${CAFFE_DIR})
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

## Set compiler optimization flags
set(CMAKE_CXX_FLAGS "-DNDEBUG -O3 -fopenmp -flto -Wno-deprecated -Wenum-compare")
# set(CMAKE_CXX_FLAGS "-DNDEBUG -O3 -fopenmp -flto -mavx -mfma -Wno-deprecated -Wenum-compare")


## Generate messages in the 'msg' folder
add_message_files(FILES CloudIndexed.msg CloudSamples.msg CloudSources.msg GraspConfig.msg GraspConfigList.msg
  SamplesMsg.msg)


## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES geometry_msgs sensor_msgs std_msgs)

catkin_package(
INCLUDE_DIRS include
CATKIN_DEPENDS cmake_modules eigen_conversions geometry_msgs message_runtime roscpp sensor_msgs std_msgs ros_gpg
DEPENDS Eigen OpenCV PCL
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


## Declare a C++ library
add_library(${PROJECT_NAME}_caffe_classifier src/${PROJECT_NAME}/caffe_classifier.cpp)
add_library(${PROJECT_NAME}_clustering src/${PROJECT_NAME}/clustering.cpp)
add_library(${PROJECT_NAME}_data_generator src/${PROJECT_NAME}/data_generator.cpp)
add_library(${PROJECT_NAME}_grasp_detector src/${PROJECT_NAME}/grasp_detector.cpp)
add_library(${PROJECT_NAME}_learning src/${PROJECT_NAME}/learning.cpp)
add_library(${PROJECT_NAME}_sequential_importance_sampling src/${PROJECT_NAME}/sequential_importance_sampling.cpp)
add_library(${PROJECT_NAME}_grasp_image src/${PROJECT_NAME}/grasp_image.cpp)
add_library(${PROJECT_NAME}_grasp_image_15_channels src/${PROJECT_NAME}/grasp_image_15_channels.cpp)

add_library(${PROJECT_NAME}_grasp_detector_gmm src/${PROJECT_NAME}/grasp_detector_gmm.cpp)
## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(grasp_candidates_classifier ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare C++ executables
add_executable(${PROJECT_NAME}_detect_grasps src/nodes/grasp_detection_node.cpp)
add_executable(${PROJECT_NAME}_generate_candidates src/nodes/generate_candidates.cpp)
add_executable(${PROJECT_NAME}_create_grasp_images src/nodes/create_grasp_images.cpp)
add_executable(${PROJECT_NAME}_create_training_data src/nodes/create_training_data.cpp)
add_executable(${PROJECT_NAME}_classify_candidates src/nodes/classify_candidates.cpp)
add_executable(${PROJECT_NAME}_test_occlusion src/tests/test_occlusion.cpp)

add_executable(${PROJECT_NAME}_detect_grasps_gmm src/nodes/grasp_detection_node_gmm.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(grasp_candidates_classifier_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_caffe_classifier
                      ${Caffe_LIBRARIES}
                      ${OpenCV_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_clustering
                      ros_gpg_grasp_candidates_generator)

target_link_libraries(${PROJECT_NAME}_create_grasp_images
                      ${PROJECT_NAME}_learning
                      ros_gpg_grasp_candidates_generator
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_create_training_data
                      ${PROJECT_NAME}_data_generator)

target_link_libraries(${PROJECT_NAME}_data_generator
                      ${PROJECT_NAME}_learning
                      ros_gpg_grasp_candidates_generator
                      ${Caffe_LIBRARIES}
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_generate_candidates
                      ros_gpg_grasp_candidates_generator
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_classify_candidates
                      ${PROJECT_NAME}_grasp_detector
                      ${PROJECT_NAME}_sequential_importance_sampling
                      ros_gpg_grasp_candidates_generator
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${ARMADILLO_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_grasp_detector
                      ${PROJECT_NAME}_caffe_classifier
                      ${PROJECT_NAME}_clustering
                      ${PROJECT_NAME}_learning
                      ros_gpg_grasp_candidates_generator
                      ${catkin_LIBRARIES}
                      ${Caffe_LIBRARIES}
                      ${ARMADILLO_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_grasp_detector_gmm
                      ${PROJECT_NAME}_caffe_classifier
                      ${PROJECT_NAME}_clustering
                      ${PROJECT_NAME}_learning
                      ros_gpg_grasp_candidates_generator
                      ${catkin_LIBRARIES}
                      ${Caffe_LIBRARIES}
                      ${ARMADILLO_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_grasp_image
                      ${OpenCV_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_grasp_image_15_channels
                      ${PROJECT_NAME}_grasp_image)

target_link_libraries(${PROJECT_NAME}_learning
                      ${PROJECT_NAME}_grasp_image_15_channels
                      ros_gpg_grasp_candidates_generator)

target_link_libraries(${PROJECT_NAME}_detect_grasps
                      ${PROJECT_NAME}_grasp_detector
                      ${PROJECT_NAME}_sequential_importance_sampling
                      ros_gpg_grasp_candidates_generator
                      ${PCL_LIBRARIES}
                      ${ARMADILLO_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_detect_grasps_gmm
                      ${PROJECT_NAME}_grasp_detector_gmm
                      ${PROJECT_NAME}_sequential_importance_sampling
                      ros_gpg_grasp_candidates_generator
                      ${PCL_LIBRARIES}
                      ${ARMADILLO_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_sequential_importance_sampling
                      ${PROJECT_NAME}_grasp_detector
                      ${ARMADILLO_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_test_occlusion
                      ${PROJECT_NAME}_learning
                      ros_gpg_grasp_candidates_generator
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${Caffe_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_learning
                      ros_gpg_grasp_candidates_generator
                      ${catkin_LIBRARIES}
                      ${Caffe_LIBRARIES})
# Rename targets to simplify their names.
set_target_properties(${PROJECT_NAME}_detect_grasps
                      PROPERTIES OUTPUT_NAME detect_grasps_sli
                      PREFIX "")

set_target_properties(${PROJECT_NAME}_detect_grasps_gmm
                      PROPERTIES OUTPUT_NAME detect_grasps_sli
                      PREFIX "")

set_target_properties(${PROJECT_NAME}_classify_candidates
                      PROPERTIES OUTPUT_NAME classify_candidates_sli
                      PREFIX "")

set_target_properties(${PROJECT_NAME}_create_grasp_images
                      PROPERTIES OUTPUT_NAME create_grasp_images_sli
                      PREFIX "")

set_target_properties(${PROJECT_NAME}_create_training_data
                      PROPERTIES OUTPUT_NAME create_training_data_sli
                      PREFIX "")

set_target_properties(${PROJECT_NAME}_test_occlusion
                      PROPERTIES OUTPUT_NAME test_occlusion_sli
                      PREFIX "")
