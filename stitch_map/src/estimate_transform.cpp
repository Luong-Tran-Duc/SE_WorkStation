#include <fstream>
#include <iostream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

#include <nlohmann/json.hpp>
#include <gtsam/geometry/SO3.h>
#include <gtsam/geometry/Pose3.h>

#include <dfo/nelder_mead.hpp>
#include <vlcal/common/console_colors.hpp>

#include <camera/create_camera.hpp>
#include <vlcal/common/estimate_fov.hpp>
#include <vlcal/common/estimate_pose.hpp>
#include <vlcal/common/visual_lidar_data.hpp>

#include <glk/primitives/primitives.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <cuda_runtime.h>
#include <fast_gicp/gicp/impl/fast_vgicp_cuda_impl.hpp>
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vlcal {

class InitialGuessAuto {
public:
  InitialGuessAuto(const std::string& data_path) : data_path(data_path) {
    std::ifstream ifs(data_path + "/calib.json");
    if (!ifs) {
      std::cerr << vlcal::console::bold_red << "error: failed to open " << data_path << "/calib.json" << vlcal::console::reset << std::endl;
      abort();
    }

    const int pick_window_size = 1;
    for (int i = -pick_window_size; i <= pick_window_size; i++) {
      for (int j = -pick_window_size; j <= pick_window_size; j++) {
        if (i == 0 && j == 0) {
          continue;
        }
        pick_offsets.emplace_back(i, j);
      }
    }
    std::sort(pick_offsets.begin(), pick_offsets.end(), [](const auto& lhs, const auto& rhs) { return lhs.squaredNorm() < rhs.squaredNorm(); });

    ifs >> config;

    file_names = config["meta"]["file_names"];
  }

  std::vector<std::pair<Eigen::Vector4d, Eigen::Vector4d>> read_correspondences(const std::string& data_path, const std::string& target_name, const std::string& source_name, const cv::Mat& target_point_indices, const Frame::ConstPtr& target_points, const Frame::ConstPtr& source_points) {
    cv::Mat source_point_indices_8uc4 = cv::imread(data_path + "/" + source_name + "/" + source_name + "_lidar_indices.png", -1);
    cv::Mat source_point_indices = cv::Mat(source_point_indices_8uc4.rows, source_point_indices_8uc4.cols, CV_32SC1, reinterpret_cast<int*>(source_point_indices_8uc4.data));

    std::ifstream matches_ifs(data_path + "/" + target_name + "_vs_" + source_name + "_matches.json");
    if (!matches_ifs) {
      std::cerr << vlcal::console::bold_red << "error: failed to open " << data_path + "/" + target_name + "_vs_" + source_name + "_matches.json" << vlcal::console::reset
                << std::endl;
      abort();
    }

    nlohmann::json matching_result;
    matches_ifs >> matching_result;

    std::vector<int> kpts0 = matching_result["kpts0"];
    std::vector<int> kpts1 = matching_result["kpts1"];
    std::vector<int> matches = matching_result["matches"];
    std::vector<double> confidence = matching_result["confidence"];

    std::vector<std::pair<Eigen::Vector4d, Eigen::Vector4d>> correspondences;
    for (int i = 0; i < matches.size(); i++) {
      if (matches[i] < 0) {
        continue;
      }

      const Eigen::Vector2i kp0(kpts0[2 * i], kpts0[2 * i + 1]);
      const Eigen::Vector2i kp1(kpts1[2 * matches[i]], kpts1[2 * matches[i] + 1]);

      std::int32_t source_point_index = source_point_indices.at<std::int32_t>(kp1.y(), kp1.x());

      if (source_point_index < 0) {
        for (const auto& offset : pick_offsets) {
          source_point_index = source_point_indices.at<std::int32_t>(kp1.y() + offset.y(), kp1.x() + offset.x());

          if (source_point_index >= 0) {
            break;
          }
        }

        if (source_point_index < 0) {
          std::cerr << vlcal::console::bold_yellow << "warning: ignore keypoint in a blank region!!" << vlcal::console::reset << std::endl;
        }
        continue;
      }

      std::int32_t target_point_index = target_point_indices.at<std::int32_t>(kp0.y(), kp0.x());

      if (target_point_index < 0) {
        for (const auto& offset : pick_offsets) {
          target_point_index = target_point_indices.at<std::int32_t>(kp0.y() + offset.y(), kp0.x() + offset.x());

          if (target_point_index >= 0) {
            break;
          }
        }

        if (target_point_index < 0) {
          std::cerr << vlcal::console::bold_yellow << "warning: ignore keypoint in a blank region!!" << vlcal::console::reset << std::endl;
        }
        continue;
      }

      correspondences.emplace_back(target_points->points[target_point_index], source_points->points[source_point_index]);
    }

    return correspondences;
  }

  void estimate_and_save(const boost::program_options::variables_map& vm) {
    PoseEstimationParams params;
    params.ransac_iterations = vm["ransac_iterations"].as<int>();
    params.ransac_error_thresh = vm["ransac_error_thresh"].as<double>();
    params.ransac_angle_error_thresh = vm["ransac_angle_error_thresh"].as<double>();
    params.robust_kernel_delta = vm["robust_kernel_delta"].as<double>();
    params.robust_kernel_width = vm["robust_kernel_width"].as<double>();

    PoseEstimation pose_estimation(params);

    for (size_t i = 0; i < file_names.size() - 1; i++) {
      std::string target_name = file_names[i];
      std::string source_name = file_names[i + 1];

      std::cout << "estimating pose between " << target_name << " and " << source_name << std::endl;

      auto target_data = std::make_shared<VisualLiDARData>(data_path, target_name);
      cv::Mat target_point_indices_8uc4 = cv::imread(data_path + "/" + target_name + "/" + target_name + "_lidar_indices.png", -1);
      cv::Mat target_point_indices = cv::Mat(target_point_indices_8uc4.rows, target_point_indices_8uc4.cols, CV_32SC1, reinterpret_cast<int*>(target_point_indices_8uc4.data));

      auto source_data = std::make_shared<VisualLiDARData>(data_path, source_name);

      auto corrs = read_correspondences(data_path, target_name, source_name, target_point_indices, target_data->points, source_data->points);

      std::vector<bool> inliers;
      Eigen::Isometry3d T_target_source = pose_estimation.estimate(corrs, &inliers);

      const Eigen::Isometry3d T_source_target = T_target_source.inverse();
      const Eigen::Vector3d trans = T_target_source.translation();
      const Eigen::Quaterniond quat = Eigen::Quaterniond(T_target_source.linear()).normalized();
      const std::vector<double> values = {trans.x(), trans.y(), trans.z(), quat.x(), quat.y(), quat.z(), quat.w()};
      config["results"][target_name + "_" + source_name]["init_T_target_source"] = values;
    }

    std::ofstream ofs(data_path + "/calib.json");
    if (!ofs) {
      std::cerr << vlcal::console::bold_red << "error: failed to open " << data_path + "/calib.json" << " for writing" << vlcal::console::reset << std::endl;
      return;
    }

    ofs << config.dump(2) << std::endl;
  }

  template <typename Registration>
  Eigen::Matrix4f FineAlignment(Registration &reg, const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &target, const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &source)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZRGB>);

    double fitness_score = 0.0;

    auto t1 = std::chrono::high_resolution_clock::now();

    reg.clearTarget();
    reg.clearSource();
    reg.setInputTarget(target);
    reg.setInputSource(source);
    reg.setMaxCorrespondenceDistance(0.5);
    reg.setMaximumIterations(1000);
    reg.setTransformationEpsilon(1e-6);
    reg.align(*aligned);
    auto t2 = std::chrono::high_resolution_clock::now();
    fitness_score = reg.getFitnessScore();
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans = reg.getFinalTransformation();
    double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

    std::cout << "fitness score:" << fitness_score << std::endl;
    std::cout << "Fine alignment:" << single << "[msec] " << std::endl;

    return trans;
  }

private:
  const std::string data_path;
  nlohmann::json config;

  std::vector<Eigen::Vector2i> pick_offsets;

  std::vector<std::string> file_names;
};
}  // namespace vlcal

int main(int argc, char** argv) {
  using namespace boost::program_options;
  options_description description("initial_guess_auto");

  // clang-format off
  description.add_options()
    ("help", "produce help message")
    ("data_path", value<std::string>(), "directory that contains preprocessed data")
    ("ransac_iterations", value<int>()->default_value(8192), "iterations for RANSAC")
    ("ransac_error_thresh", value<double>()->default_value(10), "reprojection error threshold")
    ("ransac_angle_error_thresh", value<double>()->default_value(0.35), "angle error threshold")
    ("robust_kernel_delta", value<double>()->default_value(0.3), "Cauchy kernel delta for fine estimation")
    ("robust_kernel_width", value<double>()->default_value(10.0), "Cauchy kernel width for fine estimation")
  ;
  // clang-format on

  positional_options_description p;
  p.add("data_path", 1);

  variables_map vm;
  store(command_line_parser(argc, argv).options(description).positional(p).run(), vm);
  notify(vm);

  if (vm.count("help") || !vm.count("data_path")) {
    std::cout << description << std::endl;
    return 0;
  }

  const std::string data_path = vm["data_path"].as<std::string>();
  vlcal::InitialGuessAuto init_guess(data_path);
  init_guess.estimate_and_save(vm);

  return 0;
}