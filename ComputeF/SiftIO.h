#pragma once

#ifndef  SIFTIO_H
#define  SIFTIO_H

#include "Eigen/Core"

class Image_info {

public:
	// Constructor
	Image_info(
		const std::string img_name_
		);

	// Destructor
	~Image_info() {}

	// Read in .aux file
	void read_Auxililiary();

	// Read in .sift file
	void read_Sift();

	// Return keypoints coordinates in a 2 x N matrix
	Eigen::Matrix<float, 2, Eigen::Dynamic> get_coordinates();

	// Static functions
	static std::string extract_SIFT_name(const std::string img_name_);
	static std::string extract_AUX_name(const std::string img_name_);
	static void splitFilename(const std::string target_, std::string& path_, std::string& name_);

private:
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
};

#endif // ! SIFTIO_H
