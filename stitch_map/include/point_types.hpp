#ifndef POINT_TYPES_HPP
#define POINT_TYPES_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h>

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
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, rgb, rgb))

#endif // POINT_TYPES_HPP