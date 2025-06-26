#include "kalman_filter.hpp"
#include <cmath>

namespace tracking {

// Chi-square 0.95 quantiles for gating
const std::map<int, float> KalmanFilter::chi2inv95 = {
    {1, 3.8415f}, {2, 5.9915f}, {3, 7.8147f}, {4, 9.4877f},
    {5, 11.070f}, {6, 12.592f}, {7, 14.067f}, {8, 15.507f}, {9, 16.919f}
};

KalmanFilter::KalmanFilter() {
    const int ndim = 4;
    const float dt = 1.0f;

    // Create Kalman filter matrices
    motion_mat_ = Eigen::MatrixXf::Identity(2 * ndim, 2 * ndim);
    for (int i = 0; i < ndim; ++i) {
        motion_mat_(i, ndim + i) = dt;
    }

    update_mat_ = Eigen::MatrixXf::Identity(ndim, 2 * ndim);

    // Motion and observation uncertainty parameters
    std_weight_position_ = 1.0f / 20.0f;
    std_weight_velocity_ = 1.0f / 160.0f;
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::initiate(const Eigen::Vector4f& measurement) {
    Eigen::VectorXf mean = Eigen::VectorXf::Zero(8);
    mean.head(4) = measurement;

    Eigen::VectorXf std(8);
    std << 
        2 * std_weight_position_ * measurement(3),
        2 * std_weight_position_ * measurement(3),
        1e-2f,
        2 * std_weight_position_ * measurement(3),
        10 * std_weight_velocity_ * measurement(3),
        10 * std_weight_velocity_ * measurement(3),
        1e-5f,
        10 * std_weight_velocity_ * measurement(3);

    Eigen::MatrixXf covariance = std.array().square().matrix().asDiagonal();
    return {mean, covariance};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::predict(
    const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance) {
    
    Eigen::VectorXf std_pos(4);
    std_pos << 
        std_weight_position_ * mean(3),
        std_weight_position_ * mean(3),
        1e-2f,
        std_weight_position_ * mean(3);

    Eigen::VectorXf std_vel(4);
    std_vel << 
        std_weight_velocity_ * mean(3),
        std_weight_velocity_ * mean(3),
        1e-5f,
        std_weight_velocity_ * mean(3);

    Eigen::VectorXf std_concat(8);
    std_concat << std_pos, std_vel;
    
    Eigen::MatrixXf motion_cov = std_concat.array().square().matrix().asDiagonal();
    Eigen::VectorXf new_mean = motion_mat_ * mean;
    Eigen::MatrixXf new_covariance = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;

    return {new_mean, new_covariance};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::project(
    const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance, float confidence) const {
    
    Eigen::VectorXf std(4);
    std << 
        std_weight_position_ * mean(3),
        std_weight_position_ * mean(3),
        1e-1f,
        std_weight_position_ * mean(3);

    if (confidence > 0.0f) {
        std *= (1.0f - confidence);
    }

    Eigen::MatrixXf innovation_cov = std.array().square().matrix().asDiagonal();
    Eigen::VectorXf mean_proj = update_mat_ * mean;
    Eigen::MatrixXf covariance_proj = update_mat_ * covariance * update_mat_.transpose() + innovation_cov;

    return {mean_proj, covariance_proj};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::update(
    const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance,
    const Eigen::Vector4f& measurement, float confidence) {
    
    auto [projected_mean, projected_cov] = project(mean, covariance, confidence);

    // Compute Kalman gain using Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXf> llt(projected_cov);
    Eigen::MatrixXf kalman_gain = (llt.solve(update_mat_ * covariance.transpose())).transpose();

    Eigen::VectorXf innovation = measurement - projected_mean;
    Eigen::VectorXf new_mean = mean + kalman_gain * innovation;
    Eigen::MatrixXf new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();

    return {new_mean, new_covariance};
}

Eigen::VectorXf KalmanFilter::gatingDistance(
    const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance,
    const Eigen::MatrixXf& measurements, bool only_position) const {
    
    auto [mean_proj, covariance_proj] = project(mean, covariance);

    if (only_position) {
        mean_proj = mean_proj.head(2);
        covariance_proj = covariance_proj.block(0, 0, 2, 2);
    }

    Eigen::LLT<Eigen::MatrixXf> llt(covariance_proj);
    Eigen::MatrixXf L = llt.matrixL();

    int num_measurements = measurements.rows();
    Eigen::VectorXf squared_maha(num_measurements);

    for (int i = 0; i < num_measurements; ++i) {
        Eigen::VectorXf diff = measurements.row(i).transpose() - mean_proj;
        Eigen::VectorXf z = L.triangularView<Eigen::Lower>().solve(diff);
        squared_maha(i) = z.squaredNorm();
    }

    return squared_maha;
}

} // namespace tracking 