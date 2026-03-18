#define PCL_NO_PRECOMPILE

#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <rclcpp/rclcpp.hpp>

#include <stitch_map_msgs/msg/stitch_command.hpp>
#include <stitch_map_msgs/msg/stitch_results.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <cuda_runtime.h>
#include <fast_gicp/gicp/impl/fast_vgicp_cuda_impl.hpp>
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


#include <nlohmann/json.hpp>
#include <gtsam/geometry/SO3.h>
#include <gtsam/geometry/Pose3.h>

#include <dfo/nelder_mead.hpp>
#include <vlcal/common/console_colors.hpp>

#include <camera/create_camera.hpp>
#include <vlcal/common/frame_cpu.hpp>
#include <vlcal/common/estimate_fov.hpp>
#include <vlcal/common/estimate_pose.hpp>
#include <vlcal/common/visual_lidar_data.hpp>
#include <vlcal/preprocess/generate_lidar_image.hpp>

#include <glk/io/ply_io.hpp>
#include <glk/texture_opencv.hpp>
#include <glk/primitives/primitives.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <pcl/register_point_struct.h>
#include <pcl/io/auto_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>


#include <LASlib/lasreader.hpp>

#include <LASlib/laswriter.hpp>
#include <LASlib/laswriter_las.hpp>
#include <LASlib/laspoint.hpp>
#include <LASlib/lasquantizer.hpp>
#include <LASlib/lasdefinitions.hpp>



namespace fs = std::filesystem;

// =================== CUSTOM POINT ===================
namespace pcl {
struct PointXYZIRGB {
  PCL_ADD_POINT4D;
  float intensity;
  PCL_ADD_RGB;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT(
  pcl::PointXYZIRGB,
  (float, x, x)(float, y, y)(float, z, z)
  (float, intensity, intensity)
  (float, rgb, rgb)
)

// =================== FILE INFO ===================
struct FileInfo {
  std::string name;
  int device_id;
  long long timestamp;
};

// =================================================
// ================= PreprocessMap =================
// =================================================
namespace vlcal {

  class PreprocessMap {
  public:
    PreprocessMap() {}

    int run(const boost::program_options::variables_map& vm) {
      const std::string map_path = vm["map_path"].as<std::string>();
      const std::string dst_path = vm["dst_path"].as<std::string>();
      boost::filesystem::create_directories(dst_path);

      // Support processing a single file or all supported files inside a directory.
      std::vector<std::string> file_names;
      std::vector<std::pair<FileInfo, size_t>> files_with_indices;  // FileInfo + index in pointclouds

      boost::filesystem::path mp(map_path);
      if (boost::filesystem::is_directory(mp)) {
        for (auto& entry : boost::filesystem::directory_iterator(mp)) {
          if (!boost::filesystem::is_regular_file(entry.path())) continue;
          std::string ext = boost::algorithm::to_lower_copy(entry.path().extension().string());
          if (ext != ".pcd" && ext != ".ply" && ext != ".las" && ext != ".laz") continue;

          const std::string file_path = entry.path().string();
          std::cout << "Processing: " << file_path << std::endl;
          auto lidar_points = load_lidar_points(file_path, vm["voxel_resolution"].as<double>(), vm["min_distance"].as<double>());
          if (!lidar_points) {
            std::cerr << "warning: failed to load " << file_path << std::endl;
            continue;
          }

          const std::string base = entry.path().stem().string();
          const std::string out_dir = dst_path + "/" + base;
          boost::filesystem::create_directories(out_dir);
          if (!save_lidar_data(out_dir, lidar_points, base)) {
            std::cerr << "warning: failed to save data for " << file_path << std::endl;
            continue;
          }
          
          FileInfo info;
          if (parse_file_name(base, info)) {
            files_with_indices.emplace_back(info, pointclouds.size() - 1);  // Store FileInfo with its pointcloud index
          } else {
            std::cerr << "warning: invalid file name format: " << base << std::endl;
          }
          file_names.push_back(base);
        }
      }

      nlohmann::json config;
      config["meta"]["data_path"] = map_path;
      if (!files_with_indices.empty()) {
        // Sort by device_id and timestamp
        std::sort(files_with_indices.begin(), files_with_indices.end(),
          [](const std::pair<FileInfo, size_t>& a, const std::pair<FileInfo, size_t>& b) {
            if (a.first.device_id != b.first.device_id)
              return a.first.device_id < b.first.device_id;        
            return a.first.timestamp < b.first.timestamp;          
          }
        );

        // Reorder pointclouds according to sorted order
        std::vector<pcl::PointCloud<pcl::PointXYZIRGB>::Ptr> sorted_pointclouds;
        sorted_pointclouds.reserve(files_with_indices.size());
        for (const auto& pair : files_with_indices) {
          sorted_pointclouds.push_back(pointclouds[pair.second]);
        }
        pointclouds = sorted_pointclouds;

        auto& arr = config["meta"]["file_names"];
        arr = nlohmann::json::array();

        for (size_t i = 0; i < files_with_indices.size(); ++i) {
          arr.push_back(files_with_indices[i].first.name);
        }
      } else {
        std::cerr << "warning: no valid point cloud files found in " << map_path << std::endl;
      }

      std::ofstream ofs(dst_path + "/meta_data.json");
      ofs << config.dump(2) << std::endl;

      return 0;
    }

    bool parse_file_name(const std::string& name, FileInfo& info) {
      // expected: device_<id>_<timestamp>
      std::vector<std::string> tokens;
      boost::split(tokens, name, boost::is_any_of("_"));

      if (tokens.size() != 3 || tokens[0] != "device")
        return false;

      try {
        info.name = name;
        info.device_id = std::stoi(tokens[1]);
        info.timestamp = std::stoll(tokens[2]);
      } catch (...) {
        return false;
      }
      return true;
    }

    vlcal::Frame::ConstPtr load_lidar_points(const std::string& path, double voxel_resolution, double min_distance) {
      pcl::PointCloud<pcl::PointXYZIRGB>::Ptr map_points(
        new pcl::PointCloud<pcl::PointXYZIRGB>());
      
      std::string ext = boost::filesystem::extension(path);
      boost::algorithm::to_lower(ext);

      if (ext == ".pcd" || ext == ".ply") {
        if (pcl::io::load(path, *map_points) != 0) {
          std::cerr << "error: failed to load point cloud " << path << std::endl;
          return nullptr;
        }
      }
      else if (ext == ".las" || ext == ".laz") {
        map_points = extract_points_from_laz(path);
        if (!map_points || map_points->empty()) {
          std::cerr << "error: failed to load LAZ/LAS " << path << std::endl;
          return nullptr;
        }
      }
      else {
        std::cerr << "error: unsupported file format: " << path << std::endl;
        return nullptr;
      }

      pcl::ApproximateVoxelGrid<pcl::PointXYZIRGB> voxelgrid;
      voxelgrid.setLeafSize(voxel_resolution, voxel_resolution, voxel_resolution);
      voxelgrid.setInputCloud(map_points);
      auto filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZIRGB>>();
      voxelgrid.filter(*filtered);

      std::cout << "map_points=" << map_points->size() << " filtered=" << filtered->size() << std::endl;
      map_points = filtered;

      pointclouds.push_back(map_points);

      std::vector<Eigen::Vector4d> points(map_points->size());
      std::vector<double> intensities(map_points->size());
      std::vector<Eigen::Vector4d> colors(map_points->size());

      for (int i = 0; i < map_points->size(); i++) {
        const auto& pt = map_points->points[i];
        points[i] << static_cast<double>(pt.x),
                static_cast<double>(pt.y),
                static_cast<double>(pt.z),
                1.0;
        intensities[i] = static_cast<double>(pt.intensity);

        uint32_t rgb = *reinterpret_cast<const uint32_t*>(&pt.rgb);
        uint8_t r = (rgb >> 16) & 0xFF;
        uint8_t g = (rgb >> 8)  & 0xFF;
        uint8_t b = rgb & 0xFF;
        colors[i] << r / 255.0,
                g / 255.0,
                b / 255.0,
                1.0;   // alpha
      }

      auto frame = std::make_shared<FrameCPU>(points);
      if (!intensities.empty()) {
        frame->add_intensities(intensities);
      }

      if (!colors.empty() && colors.size() == points.size()) {
        frame->add_colors(colors);
      }

      if (frame->intensities != nullptr && frame->size() > 0) {
        // histrogram equalization
        std::vector<int> indices(frame->size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](const int lhs, const int rhs) { return frame->intensities[lhs] < frame->intensities[rhs]; });

        const int bins = 256;
        for (int i = 0; i < indices.size(); i++) {
          const double value = std::floor(bins * static_cast<double>(i) / indices.size()) / bins;
          frame->intensities[indices[i]] = value;
        }
      }

      return frame;
    }

    pcl::PointCloud<pcl::PointXYZIRGB>::Ptr extract_points_from_laz(const std::string& laz_filename) {
      auto t_start = std::chrono::steady_clock::now();
      LASreadOpener opener;
      opener.set_file_name(laz_filename.c_str());

      LASreader* reader = opener.open();
      if (!reader) {
          std::cerr << "failed to open " << laz_filename << std::endl;
          return nullptr;
      }

      const LASheader& header = reader->header;

      std::cout << "LAS version: "
                << int(header.version_major) << "."
                << int(header.version_minor) << std::endl;
      std::cout << "Point format: " << header.point_data_format << std::endl;

      pcl::PointCloud<pcl::PointXYZIRGB>::Ptr cloud(
          new pcl::PointCloud<pcl::PointXYZIRGB>());
      cloud->reserve(header.number_of_point_records);
      cloud->is_dense = false;

      while (reader->read_point()) {
          const LASpoint& p = reader->point;
          pcl::PointXYZIRGB pt;

          // ---- XYZ ----
          pt.x = p.get_x();
          pt.y = p.get_y();
          pt.z = p.get_z();

          // ---- Intensity ----
          pt.intensity = static_cast<float>(p.intensity);

          // ---- RGB ----
          const U16* rgb = p.get_RGB();
          if (rgb) {
              pt.r = static_cast<uint8_t>(rgb[0]);
              pt.g = static_cast<uint8_t>(rgb[1]);
              pt.b = static_cast<uint8_t>(rgb[2]);
          } else {
              pt.r = pt.g = pt.b = 255;
          }

          cloud->push_back(pt);
      }

      reader->close();
      delete reader;

      cloud->width  = cloud->size();
      cloud->height = 1;

      std::cout << "Loaded " << cloud->size()
                << " points from " << laz_filename << std::endl;

      auto t_end = std::chrono::steady_clock::now();

      const double total_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

      std::cout << "[LAZ loader] total time : " <<  total_ms << "ms" << std::endl;


      return cloud;
    }

    bool save_lidar_data(const std::string& dst_path, const Frame::ConstPtr& lidar_points, const std::string& basename) {
      glk::PLYData ply;
      ply.vertices.resize(lidar_points->size());
      ply.intensities.resize(lidar_points->size());
      ply.colors.resize(lidar_points->size());
      for (int i = 0; i < lidar_points->size(); i++) {
        ply.vertices[i] = lidar_points->points[i].cast<float>().head<3>();
        ply.intensities[i] = lidar_points->intensities[i];
        ply.colors[i] = lidar_points->colors[i].cast<float>();
      }
      glk::save_ply_binary(dst_path + "/" + basename + ".ply", ply);

      // Generate LiDAR images
      const double lidar_fov = vlcal::estimate_lidar_fov(lidar_points);
      std::cout << "LiDAR FoV: " << lidar_fov * 180.0 / M_PI << "[deg]" << std::endl;
      Eigen::Vector2i lidar_image_size;
      std::string lidar_camera_model;
      std::vector<double> lidar_camera_intrinsics;

      Eigen::Isometry3d T_lidar_camera = Eigen::Isometry3d::Identity();

      if (lidar_fov < 150.0 * M_PI / 180.0) {
        lidar_image_size = {1024, 1024};
        T_lidar_camera.linear() = (Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        const double fx = lidar_image_size.x() / (2.0 * std::tan(lidar_fov / 2.0));
        lidar_camera_model = "plumb_bob";
        lidar_camera_intrinsics = {fx, fx, lidar_image_size[0] / 2.0, lidar_image_size[1] / 2.0};
      } else {
        lidar_image_size = {1920, 960};
        T_lidar_camera.linear() = (Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitX())).toRotationMatrix();
        lidar_camera_model = "equirectangular";
        lidar_camera_intrinsics = {static_cast<double>(lidar_image_size[0]), static_cast<double>(lidar_image_size[1])};
      }

      auto lidar_proj = camera::create_camera(lidar_camera_model, lidar_camera_intrinsics, {});

      cv::Mat intensities;
      cv::Mat colors;    // grayscale from color
      cv::Mat indices;

      if (!lidar_points->has_colors()) {
        std::tie(intensities, indices) = vlcal::generate_lidar_image(lidar_proj, lidar_image_size, T_lidar_camera.inverse(), lidar_points);
      } else {
        std::tie(intensities, colors, indices) = vlcal::generate_lidar_image_has_color(lidar_proj, lidar_image_size, T_lidar_camera.inverse(), lidar_points);
      }
    

      cv::Mat indices_8uc4(indices.rows, indices.cols, CV_8UC4, reinterpret_cast<cv::Vec4b*>(indices.data));

      intensities.clone().convertTo(intensities, CV_8UC1, 255.0);
      cv::imwrite(dst_path + "/" + basename + "_lidar_intensities.png", intensities);
      cv::imwrite(dst_path + "/" + basename + "_lidar_indices.png", indices_8uc4);
      if (!colors.empty()) {
        cv::Mat colors_u8;
        colors.convertTo(colors_u8, CV_8UC1, 255.0);
        cv::imwrite(dst_path + "/" + basename + "_lidar_color_gray.png", colors_u8);
      }

      return true;
    }

  private:
    std::vector<pcl::PointCloud<pcl::PointXYZIRGB>::Ptr> pointclouds;
  public:
    const std::vector<pcl::PointCloud<pcl::PointXYZIRGB>::Ptr>& get_pointclouds() const { return pointclouds; }
    std::vector<pcl::PointCloud<pcl::PointXYZIRGB>::Ptr>& get_pointclouds() { return pointclouds; }
  };

  class InitialGuessAuto {
  public:
    InitialGuessAuto(const std::string& data_path, std::vector<pcl::PointCloud<pcl::PointXYZIRGB>::Ptr>* preprocessed_pointclouds_ptr = nullptr)
      : data_path(data_path), preprocessed_pointclouds(preprocessed_pointclouds_ptr) {
      std::ifstream ifs(data_path + "/meta_data.json");
      if (!ifs) {
        std::cerr << vlcal::console::bold_red << "error: failed to open " << data_path << "/meta_data.json" << vlcal::console::reset << std::endl;
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

      // If preprocessed pointclouds provided, use them for coarse/fine alignment and transform
      if (preprocessed_pointclouds) {
        auto &pcs = *preprocessed_pointclouds;
        for (int i = static_cast<int>(pcs.size()) - 1; i > 0; --i) {
          std::string target_name = file_names[i - 1];
          std::string source_name = file_names[i];

          std::cout << "estimating pose between " << target_name << " and " << source_name << std::endl;

          auto target_data = std::make_shared<VisualLiDARData>(data_path, target_name);
          cv::Mat target_point_indices_8uc4 = cv::imread(data_path + "/" + target_name + "/" + target_name + "_lidar_indices.png", -1);
          cv::Mat target_point_indices = cv::Mat(target_point_indices_8uc4.rows, target_point_indices_8uc4.cols, CV_32SC1, reinterpret_cast<int*>(target_point_indices_8uc4.data));

          auto source_data = std::make_shared<VisualLiDARData>(data_path, source_name);

          auto corrs = read_correspondences(data_path, target_name, source_name, target_point_indices, target_data->points, source_data->points);

          std::vector<bool> inliers;
          Eigen::Isometry3d T_target_source = pose_estimation.estimate(corrs, &inliers);

          const Eigen::Vector3d trans = T_target_source.translation();
          const Eigen::Quaterniond quat = Eigen::Quaterniond(T_target_source.linear()).normalized();
          const std::vector<double> values = {trans.x(), trans.y(), trans.z(), quat.x(), quat.y(), quat.z(), quat.w()};
          config["results"][target_name + "_" + source_name]["init_T_target_source"] = values;

          // Coarse: transform source pointcloud by T_target_source
          Eigen::Matrix4f T_coarse = T_target_source.matrix().cast<float>();
          std::cout << "Coarse Alignment between " << target_name << " and " << source_name << ":\n" << T_coarse << std::endl;

          pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::copyPointCloud(*pcs[i], *source);

          pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::copyPointCloud(*pcs[i - 1], *target);

          pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_aligned(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::transformPointCloud(*source, *coarse_aligned, T_coarse);

          pcl::VoxelGrid<pcl::PointXYZ> voxel;
          voxel.setLeafSize(0.15f, 0.15f, 0.15f);
          voxel.setInputCloud(coarse_aligned);
          voxel.filter(*coarse_aligned);

          voxel.setInputCloud(target);
          voxel.filter(*target);


          Eigen::Matrix4f T_fine = Eigen::Matrix4f::Identity();

          #ifdef USE_VGICP_CUDA
          // Fine alignment with VGICP-CUDA
          // std::cout << "--- vgicp_cuda interation " << inter << "/" << vm["fine_alignment_iterations"].as<int>() << " ---" << std::endl;
          std::cout << "--- vgicp_cuda ---" << std::endl;
          cudaDeviceSynchronize();
          fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp_cuda;
          vgicp_cuda.setResolution(0.1);
          cudaDeviceSynchronize();
          Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
          // transform = FineAlignment(vgicp_cuda, target, coarse_aligned, (vm["fine_alignment_iterations"].as<int>() - inter) * 0.5);
          transform = FineAlignment(vgicp_cuda, target, coarse_aligned);
          std::cout << "Transform from VGICP-CUDA:\n" << transform << std::endl;
          T_fine = transform * T_fine;
          // pcl::transformPointCloud(*coarse_aligned, *coarse_aligned, transform);
          cudaDeviceSynchronize();?
          #else
          // Fine alignment
          std::cout << "--- vgicp_mt ---" << std::endl;
          fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp_mt;
          #ifdef _OPENMP
          vgicp_mt.setNumThreads(omp_get_max_threads());
          #endif
          vgicp_mt.setResolution(0.1f);
          Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
          transform = FineAlignment(vgicp_mt, target, coarse_aligned);
          std::cout << "Transform from VGICP-MT:\n" << transform << std::endl;
          T_fine = transform * T_fine;
          #endif

          std::cout << "Fine Alignment between " << target_name << " and " << source_name << ":\n" << T_fine << std::endl;
          Eigen::Matrix4f T_full = T_fine * T_coarse;

          std::cout << "Full Alignment between " << target_name << " and " << source_name << ":\n" << T_full << std::endl;

          Eigen::Isometry3d T_fine_iso;
          T_fine_iso.matrix() = T_full.cast<double>();
          Eigen::Vector3d trans_fine = T_fine_iso.translation();
          Eigen::Quaterniond quat_fine = Eigen::Quaterniond(T_fine_iso.linear()).normalized();
          const std::vector<double> fine_values = {trans_fine.x(), trans_fine.y(), trans_fine.z(), quat_fine.x(), quat_fine.y(), quat_fine.z(), quat_fine.w()};
          config["results"][target_name + "_" + source_name]["fine_T_target_source"] = fine_values;

          // Apply T_full to all pointclouds j >= i
          for (int j = i; j < static_cast<int>(pcs.size()); ++j) {
            pcl::PointCloud<pcl::PointXYZIRGB>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZIRGB>);
            pcl::transformPointCloud(*pcs[j], *transformed, T_full);
            pcs[j] = transformed;
          }
        }
      } else {
        // fallback: original behavior (only estimate and save init transforms)
        for (size_t i = file_names.size() - 1; i > 0; i--) {
          std::string target_name = file_names[i - 1];
          std::string source_name = file_names[i];

          std::cout << "estimating pose between " << target_name << " and " << source_name << std::endl;

          auto target_data = std::make_shared<VisualLiDARData>(data_path, target_name);
          cv::Mat target_point_indices_8uc4 = cv::imread(data_path + "/" + target_name + "/" + target_name + "_lidar_indices.png", -1);
          cv::Mat target_point_indices = cv::Mat(target_point_indices_8uc4.rows, target_point_indices_8uc4.cols, CV_32SC1, reinterpret_cast<int*>(target_point_indices_8uc4.data));

          auto source_data = std::make_shared<VisualLiDARData>(data_path, source_name);

          auto corrs = read_correspondences(data_path, target_name, source_name, target_point_indices, target_data->points, source_data->points);

          std::vector<bool> inliers;
          Eigen::Isometry3d T_target_source = pose_estimation.estimate(corrs, &inliers);

          const Eigen::Vector3d trans = T_target_source.translation();
          const Eigen::Quaterniond quat = Eigen::Quaterniond(T_target_source.linear()).normalized();
          const std::vector<double> values = {trans.x(), trans.y(), trans.z(), quat.x(), quat.y(), quat.z(), quat.w()};
          config["results"][target_name + "_" + source_name]["init_T_target_source"] = values;
        }
      }

      pcl::PointCloud<pcl::PointXYZIRGB>::Ptr pointcloud_stitched(new pcl::PointCloud<pcl::PointXYZIRGB>);
      for (int i = 0; i < preprocessed_pointclouds->size(); ++i) {
        std::string name = file_names[i];
        pcl::io::savePCDFileBinary(data_path + "/" + name + "_transformed.pcd", *(*preprocessed_pointclouds)[i]);
        *pointcloud_stitched += *(*preprocessed_pointclouds)[i];
      }
      
      if (save_pointcloud_to_las(data_path + "/map_stitched.las", pointcloud_stitched)) {
        std::cout << "Saved stitched pointcloud to " << data_path + "/map_stitched.las" << std::endl;
      } else {
        std::cerr << "Failed to save stitched pointcloud to " << data_path + "/map_stitched.las" << std::endl;
      }
      std::ofstream ofs(data_path + "/meta_data.json");
      if (!ofs) {
        std::cerr << vlcal::console::bold_red << "error: failed to open " << data_path + "/meta_data.json" << " for writing" << vlcal::console::reset << std::endl;
        return;
      }

      ofs << config.dump(2) << std::endl;
    }

    template <typename Registration>
    Eigen::Matrix4f FineAlignment(Registration &reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source, float max_correspondence_distance = 2.5f)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

      double fitness_score = 0.0;

      auto t1 = std::chrono::high_resolution_clock::now();

      reg.clearTarget();
      reg.clearSource();
      reg.setInputTarget(target);
      reg.setInputSource(source);
      reg.setMaxCorrespondenceDistance(max_correspondence_distance);
      reg.setMaximumIterations(20000);
      reg.setTransformationEpsilon(1e-8);
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

  bool save_pointcloud_to_las(
      const std::string& las_filename,
      const pcl::PointCloud<pcl::PointXYZIRGB>::ConstPtr& cloud)
  {
    std::chrono::steady_clock::time_point t_start =
        std::chrono::steady_clock::now();
      if (!cloud || cloud->empty()) {
          std::cerr << "Empty cloud, nothing to save!" << std::endl;
          return false;
      }

      // ===============================
      // 1. Setup LAS header
      // ===============================
      LASheader header;

      header.version_major = 1;
      header.version_minor = 2;

      // Point format 2: XYZ + Intensity + RGB
      header.point_data_format = 2;
      header.point_data_record_length = 26;

      header.number_of_point_records = cloud->size();

      // -------------------------------
      // Compute bounding box
      // -------------------------------
      double min_x =  std::numeric_limits<double>::max();
      double min_y =  std::numeric_limits<double>::max();
      double min_z =  std::numeric_limits<double>::max();
      double max_x = -std::numeric_limits<double>::max();
      double max_y = -std::numeric_limits<double>::max();
      double max_z = -std::numeric_limits<double>::max();

      for (const auto& p : cloud->points) {
          min_x = std::min(min_x, static_cast<double>(p.x));
          min_y = std::min(min_y, static_cast<double>(p.y));
          min_z = std::min(min_z, static_cast<double>(p.z));
          max_x = std::max(max_x, static_cast<double>(p.x));
          max_y = std::max(max_y, static_cast<double>(p.y));
          max_z = std::max(max_z, static_cast<double>(p.z));
      }

      const double padding = 0.02;
      min_x -= padding;
      min_y -= padding;
      min_z -= padding;
      max_x += padding;
      max_y += padding;
      max_z += padding;

      header.min_x = min_x;
      header.min_y = min_y;
      header.min_z = min_z;
      header.max_x = max_x;
      header.max_y = max_y;
      header.max_z = max_z;

      // -------------------------------
      // Scale & offset
      // -------------------------------
      header.x_scale_factor = 0.001;
      header.y_scale_factor = 0.001;
      header.z_scale_factor = 0.001;

      header.x_offset = min_x;
      header.y_offset = min_y;
      header.z_offset = min_z;

      // ===============================
      // 2. Open LAS writer
      // ===============================
      LASwriteOpener writer_opener;
      writer_opener.set_file_name(las_filename.c_str());

      LASwriter* writer = writer_opener.open(&header);
      if (!writer) {
          std::cerr << "Failed to open LAS writer!" << std::endl;
          return false;
      }

      // ===============================
      // 3. Write points
      // ===============================
      LASpoint las_point;
      las_point.init(&header, header.point_data_format, header.point_data_record_length);

      for (const auto& p : cloud->points) {
          // XYZ
          las_point.set_x(p.x);
          las_point.set_y(p.y);
          las_point.set_z(p.z);

          // Intensity
          las_point.intensity = static_cast<U16>(
              std::max(0.f, std::min(65535.f, p.intensity)));

          // RGB (LAS dùng U16)
          U16 rgb[3];
          rgb[0] = static_cast<U16>(p.r) << 8;
          rgb[1] = static_cast<U16>(p.g) << 8;
          rgb[2] = static_cast<U16>(p.b) << 8;
          las_point.set_RGB(rgb);

          writer->write_point(&las_point);
          writer->update_inventory(&las_point);
      }

      // ===============================
      // 4. Finalize
      // ===============================
      writer->close();
      delete writer;

      std::chrono::steady_clock::time_point t_end = 
          std::chrono::steady_clock::now(); 
      const double total_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "[LAS saver] total time : " <<  total_ms << "ms" << std::endl;

      return true;
  }

  private:
    const std::string data_path;
    nlohmann::json config;

    std::vector<Eigen::Vector2i> pick_offsets;

    std::vector<std::string> file_names;
    std::vector<pcl::PointCloud<pcl::PointXYZIRGB>::Ptr>* preprocessed_pointclouds = nullptr;
  };
}  // namespace vlcal

class StitchNode : public rclcpp::Node
{
public:
  StitchNode() : Node("stitch_map_node"),
                 state_(0),      // 0=INIT
                 processing_(false)
  {
    // Publisher
    results_pub_ = this->create_publisher<stitch_map_msgs::msg::StitchResults>(
      "/stitch/results", 5);

    // Subscriber
    command_sub_ = this->create_subscription<stitch_map_msgs::msg::StitchCommand>(
      "/stitch/command", 5,
      std::bind(&StitchNode::commandCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Stitch node started. Waiting for /stitch/command ...");
    publishStatus();
  }

private:
  void commandCallback(const stitch_map_msgs::msg::StitchCommand::SharedPtr msg)
  {
    if (processing_.exchange(true)) {
      RCLCPP_WARN(this->get_logger(), "Already processing a stitching task → ignore new command");
      return;
    }

    std::thread([this, msg]() {
      auto start_time = std::chrono::steady_clock::now();

      stitch_map_msgs::msg::StitchResults result;
      result.state = 1; // RUNNING
      result.error_message = "";
      publishStatus(&result);

      try {
        if (msg->command != 1) {
          throw std::runtime_error("Only command=1 (START) is supported in this version");
        }

        std::string input_folder = msg->input_folder;
        std::string environment  = msg->environment;

        if (input_folder.empty()) {
          throw std::runtime_error("input_folder cannot be empty!");
        }

        if (!fs::exists(input_folder)) {
            throw std::runtime_error("The input_folder directory does not exist: " + input_folder);
        }

        if (!fs::is_directory(input_folder)) {
            throw std::runtime_error("input_folder must be a directory, not a file: " + input_folder);
        }

        if (environment.empty()) {
          environment = "indoor"; // fallback
        }

        // Create dst_path
        auto now = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&tt), "_stitched_%Y%m%d_%H%M%S");
        std::string dst_folder = input_folder + ss.str();

        if (!fs::create_directories(dst_folder)) {
          throw std::runtime_error("Failed to create output directory: " + dst_folder);
        }
        RCLCPP_INFO(this->get_logger(), "dst_path = %s", dst_folder.c_str());

        result.input_file_count = 0;
        result.output_file_count = 0;

        // 1. Preprocess
        state_ = 1; // RUNNING
        publishStatus(&result);

        vlcal::PreprocessMap prep;
        boost::program_options::variables_map vm = createDefaultVM(input_folder, dst_folder);
        int prep_ret = prep.run(vm);
        if (prep_ret != 0) {
          throw std::runtime_error("Preprocess failed with error code: " + std::to_string(prep_ret));
        }

        size_t num_files = prep.get_pointclouds().size();
        if (num_files == 0) {
            throw std::runtime_error("No valid point cloud files found in the input folder");
        }
        result.input_file_count = static_cast<uint32_t>(num_files);

        // 2. SuperPoint + SuperGlue
        std::string superglue_cmd = "ros2 run stitch_map find_matches_superglue.py " +
                                    dst_folder + " --superglue " + environment;

        RCLCPP_INFO(this->get_logger(), "Running SuperGlue: %s", superglue_cmd.c_str());
        int sg_ret = std::system(superglue_cmd.c_str());
        if (sg_ret != 0) {
          throw std::runtime_error("SuperGlue failed with exit code: " + std::to_string(sg_ret) +
                                 " (check python log or script permissions)");
        }

        // 3. Alignment & Stitching
        vlcal::InitialGuessAuto init_guess(dst_folder, &prep.get_pointclouds());
        init_guess.estimate_and_save(vm);

        std::string final_las = dst_folder + "/map_stitched.las";
        if (fs::exists(final_las)) {
          result.output_file_count = 1;
          result.output_path = final_las;
        } else {
          throw std::runtime_error("Stitched LAS file not found: " + final_las);
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration_sec = std::chrono::duration<float>(end_time - start_time).count();

        result.state = 2;           // SUCCESS
        result.processing_time = duration_sec;
      }
      catch (const std::runtime_error& e) {
        result.state = 3;
        result.error_message = std::string("Runtime error: ") + e.what();
        result.processing_time = 0.0f;
        RCLCPP_ERROR(this->get_logger(), "%s", result.error_message.c_str());
      }
      catch (const std::exception& e) {
        result.state = 3;           // FAILED
        result.error_message = std::string("Undefined error: ") + e.what();
        result.processing_time = 0.0f;
        RCLCPP_ERROR(this->get_logger(), "%s", result.error_message.c_str());
      }
      catch (...) {
        result.state = 3;
        result.error_message = "unknown exception";
        result.processing_time = 0.0f;
        RCLCPP_ERROR(this->get_logger(), "Unknown exception occurred");
      }

      publishStatus(&result);
      processing_.store(false);
      state_ = result.state;

    }).detach();      
  }

  void publishStatus(const stitch_map_msgs::msg::StitchResults* res = nullptr)
  {
    auto msg = stitch_map_msgs::msg::StitchResults();
    msg.state = state_;
    msg.processing_time = 0.0f;
    msg.error_message = "";

    if (res) {
      msg = *res;
    }

    results_pub_->publish(msg);
  }

  boost::program_options::variables_map createDefaultVM(
    const std::string& map_path,
    const std::string& dst_path)
  {
    namespace po = boost::program_options;
    po::variables_map vm;

    vm.insert(std::make_pair("map_path", po::variable_value(boost::any(map_path), false)));
    vm.insert(std::make_pair("dst_path",  po::variable_value(boost::any(dst_path), false)));
    vm.insert(std::make_pair("voxel_resolution", po::variable_value(boost::any(0.002), false)));
    vm.insert(std::make_pair("min_distance",     po::variable_value(boost::any(1.0), false)));
    vm.insert(std::make_pair("ransac_iterations", po::variable_value(boost::any(8192), false)));
    vm.insert(std::make_pair("ransac_error_thresh", po::variable_value(boost::any(10.0), false)));
    vm.insert(std::make_pair("ransac_angle_error_thresh", po::variable_value(boost::any(0.35), false)));
    vm.insert(std::make_pair("robust_kernel_delta", po::variable_value(boost::any(0.3), false)));
    vm.insert(std::make_pair("robust_kernel_width", po::variable_value(boost::any(10.0), false)));
    vm.insert(std::make_pair("fine_alignment_iterations", po::variable_value(boost::any(5), false)));

    return vm;
  }

private:
  rclcpp::Publisher<stitch_map_msgs::msg::StitchResults>::SharedPtr results_pub_;
  rclcpp::Subscription<stitch_map_msgs::msg::StitchCommand>::SharedPtr command_sub_;

  std::atomic<bool> processing_;
  uint8_t state_;   // 0=INIT, 1=RUNNING, 2=SUCCESS, 3=FAILED
};
  
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StitchNode>();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;

  return 0;
}