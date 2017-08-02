#pragma once

#ifndef MATCHING_H
#define MATCHING_H

#include <vector>

#include "Eigen/Core"

#include "SiftIO.h"
#include "Graph_Disamb.h"

class Matching {
public:
	// Constructor
	Matching(const std::string image_list_path_);

	// Destructor
	~Matching();

	// Read in matchings 
	void read_matchings();

	// Write matches out into .txt file
	void write_matches(std::vector<Graph_disamb>& graphs_);

	// Write graph layout into .txt file
	void write_layout(std::vector<Graph_disamb>& graphs_);

	// Return an outlier mask for the specified pair
	cv::Mat get_outlier_mask(
		Image_info& image_left_,
		Image_info& image_right_
	);

	// Return a mask that masks out the dense area on right image
	cv::Mat get_denseArea_mask(
		Image_info& image_left_,
		Image_info& image_right_
	);

	// Display matchings by index
	void display_matchings(
		Image_info& image_left_,
		Image_info& image_right_,
		cv::Mat&	mask_ = cv::Mat(1, 1, CV_8U)
	);

	// Compute fundamental matrix and related homographies for two images
	void compute_fundamental(
		Image_info& image_left_,
		Image_info& image_right_,
		cv::Mat&	F_,
		cv::Mat&	H1_,
		cv::Mat&	H2_,
		cv::Mat&	left_warped_
	);

	// Compute homography
	void compute_homography(
		Image_info& image_left_,
		Image_info& image_right_,
		cv::Mat&	H_,
		cv::Mat&	left_warped_
	);

	// Get matching matrix by index
	int get_matchings(
		Image_info& image_left_,
		Image_info& image_right_,
		Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings
	);

	// Get the number of matchings between the two images
	int get_matching_number(
		Image_info& image_left_,
		Image_info& image_right_
	);

	// Set the warped image difference matrix
	void setWarped_diff(
		const int row_,
		const int col_,
		const float value
	);

	// Delete bad matchings
	void delete_bad_matchings(
		Image_info& image_left_,
		Image_info& image_right_
	);

	// Get the warped image difference matrix
	Eigen::MatrixXf getWarped_diff();

	// Get the matching number matrix
	Eigen::MatrixXi getMatching_number();

	// Rectify matchings according to homography mask
	void rectify_matchings_homoMask();

private:
	int image_num;
	int pair_num;

	// Path to the input txt files
	std::string					image_list_path;
	std::string					matching_path;

	// Vector containing all the image names
	std::vector<std::string>	img_names;

	// Note: all information are stored only in the upper triangle!!
	Eigen::MatrixXi				matching_number_mat;
	Eigen::MatrixXf				warped_diff_mat;

	// Stores all the matchings matrix and outlier mask in a 2-dimensional vector in upper tirangle
	std::vector<std::vector<Eigen::Matrix<int, 2, Eigen::Dynamic>>> matching_mat;
	std::vector<std::vector<cv::Mat>>								outlier_mask_mat;

	// Signals, also stored in upper tirangle!!
	Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> homography_existence_indicator;
};

#endif // !MATCHING_H
