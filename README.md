# SE_WorkStation

## 📌 Overview
`SE_WorkStation` is a workspace for developing modules related to stitching/merging point cloud fragments from SE devices or from multiple scans. The project integrates powerful libraries such as PCL, OpenCV, GTSAM, Ceres, SuperPoint, and SuperGlue to process point clouds, perform optimization, and build a complete pipeline for point cloud stitching.

---

## ⚙️ Dependencies

- ROS2 (Humble)
- PCL (Point Cloud Library)
- OpenCV
- GTSAM
- Ceres Solver
- SuperGlue (Feature Matching)

---

## 💻 System Requirements

- Ubuntu 22.04
- ROS2 Humble / Iron
- GCC >= 9
- CMake >= 3.16

---

## 🚀 Getting Started
### 1. Install dependencies
```sh
sudo apt install libomp-dev libboost-all-dev libglm-dev libglfw3-dev libpng-dev libjpeg-dev
```
```sh
git clone https://github.com/borglab/gtsam
cd gtsam && git checkout 4.2a9
mkdir build && cd build
# For Ubuntu 22.04, add -DGTSAM_USE_SYSTEM_EIGEN=ON
cmake .. -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
         -DGTSAM_BUILD_TESTS=OFF \
         -DGTSAM_WITH_TBB=OFF \
         -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
make -j$(nproc)
sudo make install
```
```sh
git clone --recurse-submodules https://github.com/ceres-solver/ceres-solver
cd ceres-solver
git checkout e47a42c2957951c9fafcca9995d9927e15557069
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DUSE_CUDA=OFF
make -j$(nproc)
sudo make install
```
```sh
git clone https://github.com/koide3/iridescence --recursive
mkdir iridescence/build && cd iridescence/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```
```sh
pip3 install numpy opencv-python torch matplotlib
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git

echo 'export PYTHONPATH=$PYTHONPATH:/path/to/SuperGluePretrainedNetwork' >> ~/.bashrc
source ~/.bashrc
```
### 2. Clone repository

```sh
cd <your_ws>/src
git clone https://github.com/Luong-Tran-Duc/SE_WorkStation.git
cd ..
colcon build
```
### 3. Quick Start Guide
```sh
ros2 run stitch_map stitch_node
```
