#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace vlcal {

class LidarLidarPointCost {
public:
  LidarLidarPointCost(
    const Eigen::Vector3d& p_tgt,
    const Eigen::Vector3d& p_src)
  : p_tgt(p_tgt), p_src(p_src) {}

  ~LidarLidarPointCost() {}

  template <typename T>
  bool operator()(const T* const T_params, T* residual) const {
    Eigen::Map<Sophus::SE3<T> const> T_tgt_src(T_params);

    Eigen::Matrix<T, 3, 1> p_est = T_tgt_src * p_src.cast<T>();
    Eigen::Matrix<T, 3, 1> err = p_est - p_tgt.cast<T>();

    residual[0] = err[0];
    residual[1] = err[1];
    residual[2] = err[2];
    return true;
  }

private:
  const Eigen::Vector3d p_src;
  const Eigen::Vector3d p_tgt;
};
}  // namespace vlcal
