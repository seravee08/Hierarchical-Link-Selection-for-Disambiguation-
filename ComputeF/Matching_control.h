#pragma once

#ifndef MATCHING_CONTROL_H
#define MATCHING_CONTROL_H

#include "Graph_Disamb.h"
#include "Image_control.h"
#include "Matching.h"

class Matching_control {
public:
	Matching_control(const std::string image_list_path_);

	~Matching_control() {}

	// Read in sift and affine information
	void readIn_Keypoints();

	// Read in matchings
	void readIn_Matchings();

	// Write out matchings
	void writeOut_Matchings(std::vector<Graph_disamb>& graphs_);

	// Write out graph layout
	void writeOut_Layout(std::vector<Graph_disamb>& graphs_);

	// Delete machings where the coordinates are out of bound
	void delete_bad_matchings();

	// Compute homography between all pairs
	void compute_Homography();

	// Rectify matchings according to homography mask
	void rectify_matching_homoMask();

	// Compute gist distance accross all pairs
	Eigen::MatrixXf compute_gist_dist_all();

	// Return number of images
	int getImage_num();

	// Get image
	cv::Mat get_image(int index_);

	// Return the matrix of warped difference
	Eigen::MatrixXf getWarped_diff_mat();

	// Return the matrix of matching number between all pairs
	Eigen::MatrixXi getMatching_number_mat();

	// Display keypoints for an image
	void displayKeypoints(int index_);
		
	// Return the number of matches between the image pair
	int getMatch_number(
		const int left_index_,
		const int right_index_
	);

	// Return the warped difference
	float getWarped_diff_value(
		const int	left_index_,
		const int	right_index_
	);

	// Display matchings according to the indices
	void displayMatchings(
		const int	left_index_,
		const int	right_index_,
		const bool	use_outlier_mask_ = false
	);

	// Construct graph using the provided scores
	std::vector<Graph_disamb> constructGraph(
		const bool minmum_guided,
		const Eigen::MatrixXf& scores
	);

	// Construct graph using the provided scores and filtered by homography difference scores
	std::vector<Graph_disamb> constructGraph_with_homography_validate(
		const bool minimum_guided,
		const Eigen::MatrixXf& scores
	);

private:
	Image_control	img_ctrl;
	Matching		match;
};


#endif // !MATCHING_CONTROL_H
