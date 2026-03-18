cmake_minimum_required(VERSION 3.16)
project(stitch_map)

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# ────────────────────────────────────────────────
#             ROS 2 dependencies
# ────────────────────────────────────────────────
find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ROS 2 core & messages
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(stitch_map_msgs REQUIRED)          # package chứa StitchCommand.msg và StitchResults.msg

# ────────────────────────────────────────────────
#             Other dependencies
# ────────────────────────────────────────────────
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options date_time)
find_package(Iridescence REQUIRED)
find_package(LASlib REQUIRED)
find_package(OpenMP)

# ────────────────────────────────────────────────
#             OpenMP flags
# ────────────────────────────────────────────────
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CUDA (nếu bật VGICP_CUDA)
if(BUILD_VGICP_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARY_DIRS})
  add_definitions(-DUSE_VGICP_CUDA)
endif()

# ────────────────────────────────────────────────
#             stitch_map library (core logic)
# ────────────────────────────────────────────────
add_library(stitch_map_library SHARED
  src/camera/create_camera.cpp
  src/vlcal/common/frame.cpp
  src/vlcal/common/frame_cpu.cpp
  src/vlcal/common/ivox.cpp
  src/vlcal/common/cloud_covariance_estimation.cpp
  src/vlcal/common/time_keeper.cpp
  src/vlcal/common/estimate_fov.cpp
  src/vlcal/common/estimate_pose.cpp
  src/vlcal/common/points_color_updater.cpp
  src/vlcal/common/visual_lidar_data.cpp
  src/vlcal/common/visual_lidar_visualizer.cpp
  src/vlcal/preprocess/preprocess.cpp
  src/vlcal/preprocess/generate_lidar_image.cpp
  src/vlcal/preprocess/static_point_cloud_integrator.cpp
  src/vlcal/preprocess/dynamic_point_cloud_integrator.cpp
  src/vlcal/calib/cost_calculator_nid.cpp
  src/vlcal/calib/view_culling.cpp
  src/vlcal/calib/visual_camera_calibration.cpp
  # Nếu bạn đã di chuyển PreprocessMap và InitialGuessAuto vào đây thì thêm file .cpp tương ứng
)

target_include_directories(stitch_map_library PUBLIC
  include
  thirdparty/json/include
  thirdparty/Bonxai/include
  thirdparty/nanoflann/include
  thirdparty/Sophus
  ${Boost_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${LASlib_INCLUDE_DIRS}
)

target_link_libraries(stitch_map_library
  fmt
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${GTSAM_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Iridescence::Iridescence
  LASlib
)

# ────────────────────────────────────────────────
#             fast_gicp library
# ────────────────────────────────────────────────
add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
)

target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)

if(OPENMP_FOUND AND TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
endif()

if(BUILD_VGICP_CUDA)
  cuda_add_library(fast_vgicp_cuda SHARED
    src/fast_gicp/cuda/fast_vgicp_cuda.cu
    src/fast_gicp/cuda/brute_force_knn.cu
    src/fast_gicp/cuda/covariance_estimation.cu
    src/fast_gicp/cuda/covariance_estimation_rbf.cu
    src/fast_gicp/cuda/covariance_regularization.cu
    src/fast_gicp/cuda/gaussian_voxelmap.cu
    src/fast_gicp/cuda/find_voxel_correspondences.cu
    src/fast_gicp/cuda/compute_derivatives.cu
    src/fast_gicp/cuda/compute_mahalanobis.cu
    src/fast_gicp/cuda/ndt_cuda.cu
    src/fast_gicp/cuda/ndt_compute_derivatives.cu
  )

  target_include_directories(fast_vgicp_cuda PRIVATE
    include
    thirdparty/Eigen
    thirdparty/nvbio
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
  )

  target_link_libraries(fast_vgicp_cuda ${PCL_LIBRARIES})
  cuda_add_cublas_to_target(fast_vgicp_cuda)

  target_sources(fast_gicp PRIVATE
    src/fast_gicp/gicp/fast_vgicp_cuda.cpp
    src/fast_gicp/ndt/ndt_cuda.cpp
  )

  target_link_libraries(fast_gicp fast_vgicp_cuda)
  add_dependencies(fast_gicp fast_vgicp_cuda)
endif()

# ────────────────────────────────────────────────
#             Executables
# ────────────────────────────────────────────────

# 1. preprocess_map 
add_executable(preprocess_map src/preprocess_map.cpp)
target_link_libraries(preprocess_map stitch_map_library fast_gicp)

# 2. stitch_node 
add_executable(stitch_node src/stitch_node.cpp)
ament_target_dependencies(stitch_node
  rclcpp
  std_msgs
  stitch_map_msgs
)
target_link_libraries(stitch_node
  stitch_map_library
  fast_gicp
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  LASlib
)

# 3. Các executable cũ (nếu vẫn muốn giữ)
add_executable(estimate_transform src/estimate_transform.cpp)
target_link_libraries(estimate_transform
  stitch_map_library
  fast_gicp
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)

# add_executable(test_estimate_transform src/test_estimate_transform.cpp)
# target_link_libraries(test_estimate_transform
#   stitch_map_library
#   fast_gicp
#   Eigen3::Eigen
#   ${PCL_LIBRARIES}
# )

# ────────────────────────────────────────────────
#             Install
# ────────────────────────────────────────────────
install(
  TARGETS
    stitch_map_library
    fast_gicp
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
    preprocess_map
    stitch_node
    estimate_transform
    # test_estimate_transform
  DESTINATION lib/${PROJECT_NAME}
)

# Python script SuperGlue
install(PROGRAMS
  scripts/find_matches_superglue.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch files, config,... (nếu có)
# install(DIRECTORY launch config
#   DESTINATION share/${PROJECT_NAME}/
# )

# Export & finalize
ament_export_libraries(stitch_map_library)
ament_export_dependencies(rclcpp std_msgs stitch_map_msgs)
ament_auto_package()