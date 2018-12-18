#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

  Hj_ = MatrixXd(3, 4);
  Hj_ << 0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises - P - check
   */
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  // Initialize process noise covariance matrix
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<	0, 0, 0, 0,
             	0, 0, 0, 0,
             	0, 0, 0, 0,
             	0, 0, 0, 0;

  // Initialize state covariance matrix P - tuneable
  // RMSE(0.0973178	0.0854597	0.451267	0.439935)
  // ekf_.P_ = MatrixXd(4, 4);
  // ekf_.P_ << 1, 0, 0, 0,
  //            0, 1, 0, 0,
  //            0, 0, 1000, 0,
  //            0, 0, 0, 1000;

  // RMSE(0.0973144 0.0854264 0.456078 0.436408) -- seem like the smaller the better in (0,0) (1,1)
  // ekf_.P_ = MatrixXd(4, 4);
  // ekf_.P_ << 0.5, 0, 0, 0,
  //            0, 0.5, 0, 0,
  //            0, 0, 1000, 0,
  //            0, 0, 0, 1000;

  // RMSE(0.0973184 0.085472 0.448755 0.442862)
  // ekf_.P_ = MatrixXd(4, 4);
  // ekf_.P_ << 1.5, 0, 0, 0,
  //            0, 1.5, 0, 0,
  //            0, 0, 1000, 0,
  //            0, 0, 0, 1000;

  //RMSE(0.0973067 0.0853886 0.458591 0.434599)
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 0.3, 0, 0, 0,
             0, 0.3, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  // Initialize ekf state
  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;

  noise_ax = 9;
  noise_ay = 9;


}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.- check
     * TODO: Create the covariance matrix. - check
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "first measurement" << endl;



    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates - check
      //         and initialize state.

      // Turn off RADAR
      float ro = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float ro_dot = measurement_pack.raw_measurements_[2];


      //RMSE(0.0973178	0.0854597	0.451267	0.439935) - tuneable
      ekf_.x_ << ro * cos(phi),
                 ro * sin(phi),
                 ro_dot * cos(phi),
                 ro_dot * sin(phi);

      // have no difference between above init RMSE(0.0973178	0.0854597	0.451267	0.439935)
      // ekf_.x_ << ro * cos(phi),
      //            ro * sin(phi),
      //            0,
      //            0;
      // return;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state. - check

      // Turn off LASER
      ekf_.x_ << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1],
                 0,
                 0;
      // return;
    }

    // done initializing, no need to predict or update

    // Update last measurement
	  previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time. - check
   * Time is measured in seconds.
  */
  // previous_timestamp_ = measurement_pack.timestamp_;
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  /**
   * TODO: Update the process noise covariance matrix. - check
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // set the process covariance matrix Q
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
              0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
              dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
              0, dt_3/2*noise_ay, 0, dt_2*noise_ay;


  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO: - check
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates - check
    // Turn off RADAR - only LADAR RMSE(0.147356 0.115235 0.647613 0.532393)
    cout << "RADAR Data" << endl;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // TODO: Laser updates - check
    // Turn off LADAR - only RADAR RMSE(0.224536 0.346004 0.556564 0.781506)
    cout << "LADAR Data" << endl;
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
