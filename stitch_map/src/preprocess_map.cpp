#define PCL_NO_PRECOMPILE

#include <iostream>
#include <chrono>
#include <cstdlib>


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/register_point_struct.h>
#include <pcl/io/auto_io.h>
// #include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include <nlohmann/json.hpp>

#include <camera/create_camera.hpp>
#include <vlcal/common/frame_cpu.hpp>
#include <vlcal/common/estimate_fov.hpp>
#include <vlcal/preprocess/generate_lidar_image.hpp>

#include <glk/io/ply_io.hpp>
#include <glk/texture_opencv.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <LASlib/lasreader.hpp>

#include <LASlib/laswriter.hpp>
#include <LASlib/laswriter_las.hpp>
#include <LASlib/laspoint.hpp>
#include <LASlib/lasquantizer.hpp>
#include <LASlib/lasdefinitions.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pcl {
  // --------- Struct PointXYZIRGB ---------
  struct PointXYZIRGB
  {
      PCL_ADD_POINT4D; // x, y, z + padding
      float intensity; // intensity
      PCL_ADD_RGB;     // rgb

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  } EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT(
    pcl::PointXYZIRGB,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, rgb, rgb))

struct FileInfo {
  std::string name;
  int device_id;
  long long timestamp;
};

namespace vlcal {

class PreprocessMap {
public:
  PreprocessMap() {}

  int run(int argc, char** argv) {
    using namespace boost::program_options;
    options_description description("preprocess_map");

    // clang-format off
    description.add_options()
      ("help", "produce help message")
      ("map_path", value<std::string>(), "path to input point cloud map (PCD or PLY or LAS)")
      ("dst_path", value<std::string>(), "directory to save preprocessed data")
      ("voxel_resolution", value<double>()->default_value(0.002), "voxel grid resolution")
      ("min_distance", value<double>()->default_value(1.0), "minimum point distance. Points closer than this value will be discarded")
    ;
    // clang-format on
    variables_map vm;
    store(command_line_parser(argc, argv).options(description).run(), vm);
    notify(vm);

    if (
      vm.count("help") || !vm.count("map_path") || !vm.count("dst_path")) {
      std::cout << description << std::endl;
      return 1;
    }

    const std::string map_path = vm["map_path"].as<std::string>();
    const std::string dst_path = vm["dst_path"].as<std::string>();
    boost::filesystem::create_directories(vm["dst_path"].as<std::string>());

    // Support processing a single file or all supported files inside a directory.
    std::vector<std::string> file_names;

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
        file_names.push_back(base);
      }
    }

    nlohmann::json config;
    config["meta"]["data_path"] = map_path;
    if (!file_names.empty()) {
      std::vector<FileInfo> files;
      files.reserve(file_names.size());

      for (const auto& name : file_names) {
        FileInfo info;
        if (parse_file_name(name, info)) {
          files.push_back(info);
        } else {
          std::cerr << "warning: invalid file name format: " << name << std::endl;
        }
      }

      std::sort(files.begin(), files.end(),
        [](const FileInfo& a, const FileInfo& b) {
          if (a.device_id != b.device_id)
            return a.device_id < b.device_id;        
          return a.timestamp < b.timestamp;          
        }
      );


      auto& arr = config["meta"]["file_names"];
      arr = nlohmann::json::array();

      for (size_t i = 0; i < files.size(); ++i) {
        arr.push_back(files[i].name);
      }
    } else {
      std::cerr << "warning: no valid point cloud files found in " << map_path << std::endl;
    }

    std::ofstream ofs(dst_path + "/calib.json");
    ofs << config.dump(2) << std::endl;

    return 0;
  }

  inline bool parse_file_name(const std::string& name, FileInfo& info) {
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
};

}  // namespace vlcal

int main(int argc, char** argv) {
  vlcal::PreprocessMap preprocess_map;
  return preprocess_map.run(argc, argv);
}