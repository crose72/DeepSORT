#pragma once

#include <Eigen/Dense>
#include <map>

namespace tracking {

class KalmanFilter {
public:
    static const std::map<int, float> chi2inv95;

    /**
     * @brief Construct a new Kalman Filter object
     * State space is 8D: (x, y, a, h, vx, vy, va, vh) where (x,y) is the center position,
     * a is the aspect ratio, h is height, and v* are their respective velocities
     */
    KalmanFilter();

    /**
     * @brief Initialize track from unassociated measurement
     * 
     * @param measurement Bounding box coordinates (x, y, a, h)
     * @return std::pair<Eigen::VectorXf, Eigen::MatrixXf> Mean and covariance of the state distribution
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> initiate(const Eigen::Vector4f& measurement);

    /**
     * @brief Run Kalman filter prediction step
     * 
     * @param mean Mean vector of the object state
     * @param covariance Covariance matrix of the object state
     * @return std::pair<Eigen::VectorXf, Eigen::MatrixXf> Predicted mean and covariance
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> predict(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance);

    /**
     * @brief Project state distribution to measurement space
     * 
     * @param mean State's mean vector
     * @param covariance State's covariance matrix
     * @param confidence Detection confidence
     * @return std::pair<Eigen::VectorXf, Eigen::MatrixXf> Projected mean and covariance
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> project(const Eigen::VectorXf& mean, 
                                                        const Eigen::MatrixXf& covariance,
                                                        float confidence = 0.0f) const;

    /**
     * @brief Run Kalman filter correction step
     * 
     * @param mean Predicted state's mean vector
     * @param covariance State's covariance matrix
     * @param measurement Measurement vector
     * @param confidence Detection confidence
     * @return std::pair<Eigen::VectorXf, Eigen::MatrixXf> Corrected mean and covariance
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> update(const Eigen::VectorXf& mean,
                                                       const Eigen::MatrixXf& covariance,
                                                       const Eigen::Vector4f& measurement,
                                                       float confidence = 0.0f);

    /**
     * @brief Compute gating distance between state distribution and measurements
     * 
     * @param mean Mean vector over the state distribution
     * @param covariance Covariance of the state distribution
     * @param measurements Matrix of measurements
     * @param only_position If true, only consider position for distance computation
     * @return Eigen::VectorXf Vector of squared Mahalanobis distances
     */
    Eigen::VectorXf gatingDistance(const Eigen::VectorXf& mean,
                                  const Eigen::MatrixXf& covariance,
                                  const Eigen::MatrixXf& measurements,
                                  bool only_position = false) const;

private:
    Eigen::MatrixXf motion_mat_;     // State transition matrix
    Eigen::MatrixXf update_mat_;     // Measurement matrix
    float std_weight_position_;      // Standard deviation multiplier for position
    float std_weight_velocity_;      // Standard deviation multiplier for velocity
};

} // namespace tracking 