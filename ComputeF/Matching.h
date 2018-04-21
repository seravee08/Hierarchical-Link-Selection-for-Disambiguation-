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

	// Compute matchings
	void compute_Matchings_1v1(
		Image_info& image_left_,
		Image_info& image_right_,
		int left_index_,
		int right_index_
	);

	// Write matches out into .txt file
	void write_matches(std::vector<Graph_disamb>& graphs_);

	// Write matches out to .txt filef
	void write_matches(Graph_disamb& graph_);

	// Write matches between a pair of images to .txt file
	void write_matches_1v1(int l_, int r_);

	// Write matches between one image and a subset of images
	void write_matches_1vN(int l_, const std::vector<int>& subset = std::vector<int>());

	// Write matches for all pairs
	void write_matches_all();

	// Write matches between the designated pairs
	std::string write_matches_designated(const std::string file_name_, const std::vector<cv::Point2i>& linkages);

	// Write graph layout into .txt file
	void write_layout(std::vector<Graph_disamb>& graphs_);

	// Retrive the name of the corresponding mat file created by VisualSFMf
	std::string get_MAT_name(int index_);

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
		cv::Mat&	mask_ = cv::Mat(1, 1, CV_8U),
		bool		locally_computed = false
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

	// Triangulate points and check reprojection error
	bool Matching::triangulateAndCheckReproj(
		const cv::Mat&				P1,
		const cv::Mat&				P2,
		std::vector<cv::Point2f>&	points1,
		std::vector<cv::Point2f>&	points2,
		const cv::Mat&				points1_mat,
		const cv::Mat&				points2_mat,
		const cv::Mat&				K1,
		const cv::Mat&				K2,
		cv::Mat&					pts_3d
	);

	// Calculate camera pose
	bool cameraPoseAndTriangulationFromFundamental(
		Image_info& image_left_,
		Image_info& image_right_,
		bool		locally_computed_ = false
	);

	// Decompose Essential matrix to rotation and translation matrices
	bool decomposeEtoRandT(
		cv::Mat_<double>& E,
		cv::Mat_<double>& R1,
		cv::Mat_<double>& R2,
		cv::Mat_<double>& t1,
		cv::Mat_<double>& t2
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

	// Get the number of matchings between the two images by index number
	int get_matching_number(
		int index_left_,
		int index_right_
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
	Eigen::MatrixXi getMatching_number_mat();

	// Get the matching number matrix in float
	Eigen::MatrixXf getMatching_number_mat_float();

	// Rectify matchings according to homography mask
	void rectify_matchings_homoMask();

	// Generate float matches number matrix
	void generate_float_matching_number_mat();

	// Output the 3D points cloud as .ply file
	static void points_to_ply(
		const std::string file_path,
		const cv::Mat& pts_3d
	);

private:
	int																	image_num;
	int																	pair_num;

	// Path to the input txt files
	std::string															image_list_path;
	std::string															matching_path;

	// Vector containing all the image names
	std::vector<std::string>											img_names;

	// Note: all information are stored only in the upper triangle!!
	Eigen::MatrixXi														matching_number_mat;
	Eigen::MatrixXf														matching_number_mat_float;
	Eigen::MatrixXf														warped_diff_mat;

	// Stores all the matchings matrix and outlier mask in a 2-dimensional vector in upper tirangle
	std::vector<std::vector<Eigen::Matrix<int, 2, Eigen::Dynamic>>>		matching_mat;
	std::vector<std::vector<cv::Mat>>									outlier_mask_mat;

	// Signals, also stored in upper tirangle!!
	Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>					homography_existence_indicator;
};

#endif // !MATCHING_H