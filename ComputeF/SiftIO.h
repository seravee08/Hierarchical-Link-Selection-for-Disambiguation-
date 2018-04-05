#pragma once

#ifndef  SIFTIO_H
#define  SIFTIO_H

#include "Eigen/Core"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

class Image_info {

public:
	// Constructor
	Image_info(
		const std::string img_name_
		);

	// Destructor
	~Image_info() {}

	// Read in .aux file
	void readAuxililiary_BINARY();

	// Read in .sift file
	void readASift_BINARY();

	// Write .sift file in binary form
	void writeSift_BINARY(bool VSFM_compatible_ = true);

	// Compute plain Sift with OpenCV functions
	void compute_Sift();

	// Return height of the image
	int getHeight();

	// Return width of the image
	int getWidth();

	// Return image name
	std::string getImageName();

	// Return the image
	cv::Mat getImage();

	// Return the K matrix
	cv::Mat getK();

	// Display keypoints
	void display_keypoints();

	// Display keypoints locally computed stored in cv::KeyPoint format
	void display_keypoints_locally_computed();

	// Return the sift status of a specific image
	bool getSiftStatus();

	// Return the auxiliary information status of a specific image (affine simulation info)
	bool getAuxStatus();

	// Return the locally computed Sift keypoints
	std::vector<cv::KeyPoint> getKeypoints_locally_computed();

	// Return the locally computed Sift descriptors
	cv::Mat getDescriptors_locally_computed();

	// Manually free the space of this object
	void freeSpace();

	// Return keypoints coordinates in a 2 x N matrix
	Eigen::Matrix<float, 2, Eigen::Dynamic> get_coordinates();

	// Return locally computed keypoints coordinates in a 2 x N matrix
	Eigen::Matrix<float, 2, Eigen::Dynamic> get_coordinates_locally_computed();

	// Calculate the difference between this image and the warped image with mask
	float compute_difference(
		const cv::Mat& warped,
		const cv::Mat& Mask,
		const bool greyScale
	);

	// Calculate the gist difference bewteen this image and the warped image with mask
	float compute_difference_gist_color(
		const cv::Mat& warped_,
		const cv::Mat& Mask_
	);

	// Static functions
	static std::string extract_SIFT_name(const std::string img_name_);
	static std::string extract_AUX_name(const std::string img_name_);
	static void splitFilename(const std::string target_, std::string& path_, std::string& name_);
	static cv::Mat blendImages(const cv::Mat& A, const cv::Mat& B);
	static std::string type2str(int type_);

	// Gist functions
	static std::vector<float> compute_gist(cv::Mat target_);
	static float compute_gist_dist(std::vector<float> desc1_, std::vector<float> desc2_);

private:
	cv::Mat image;
	cv::Mat K;

	bool aux_exist;
	bool sift_exist;

	std::string img_name;
	std::string sift_name;
	std::string aux_name;

	int width;
	int height;
	int feat_num;

	Eigen::Matrix<float, 2,		Eigen::Dynamic> auxInfo_mat;
	Eigen::Matrix<float, 6,		Eigen::Dynamic> keypoints_mat;
	Eigen::Matrix<float, 128,	Eigen::Dynamic> descriptor_mat;

	// ===== For local sift generation and matching =====
	std::vector<cv::KeyPoint>	keypoints;
	cv::Mat						descriptors;
};

#endif // ! SIFTIO_H
