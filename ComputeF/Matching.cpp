#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <stdlib.h>

#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "Matching.h"
#include "Parameters.h"

Matching::Matching(const std::string image_list_path_) :
	image_list_path	(image_list_path_),
	image_num		(0),
	pair_num		(0)
{
	// Compose the path to the matchings.txt
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path_, path, name);
	matching_path = path + "/matchings.txt";

	// Read in the image list
	std::ifstream list_in(image_list_path.c_str(), std::ios::in);
	assert(list_in.is_open());

	while (!list_in.eof()) {
		list_in >> name;

		// Check for duplicate image names
		if (std::find(img_names.begin(), img_names.end(), name) != img_names.end()) {
			continue;
		}
		img_names.push_back(name);
	}

	// Initialize for the number of images
	image_num = img_names.size();

	// Initialize for the matching data strucutres
	matching_number_mat				= Eigen::MatrixXi::Zero(image_num, image_num);
	warped_diff_mat					= Eigen::MatrixXf::Ones(image_num, image_num) * -1;
	homography_existence_indicator	= Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Zero(image_num, image_num);

	matching_mat.resize(image_num);
	outlier_mask_mat.resize(image_num);
	for (int i = 0; i < image_num; i++) {
		matching_mat[i].resize(image_num);
		outlier_mask_mat[i].resize(image_num);
	}
}

Matching::~Matching()
{
	img_names.clear();

	for (int i = 0; i < image_num; i++) {
		matching_mat[i].clear();
	}
	matching_mat.clear();
}

void Matching::read_matchings()
{
	// ===== Open input file stream =====
	std::ifstream match_in(matching_path.c_str(), std::ios::in);
	assert(match_in.is_open());

	// ===== Read in matchings =====
	int matching_number;
	std::string left_name;
	std::string right_name;

	int left_index;
	int right_index;

	while (!match_in.eof()) {
		match_in >> left_name;
		match_in >> right_name;
		match_in >> matching_number;

		// Find the index for the left and right image
		left_index  = std::find(img_names.begin(), img_names.end(), left_name)  - img_names.begin();
		right_index = std::find(img_names.begin(), img_names.end(), right_name) - img_names.begin();

		if (left_index >= img_names.size() || right_index >= img_names.size()) {
			std::cout << "Image names from the list do not match the names in the matchings.txt ..." << std::endl;
			exit(1);
		}
		assert(left_index < right_index);
		
		// Handle .txt file readin error
		if (matching_number_mat(left_index, right_index) > 0) {
			break;
		}
		matching_number_mat(left_index, right_index) = matching_number;

		// Read in the matchings
		std::vector<int> upper_row(matching_number);
		std::vector<int> lower_row(matching_number);
		for (int i = 0; i < matching_number; i++) {
			match_in >> upper_row[i];
		}
		for (int i = 0; i < matching_number; i++) {
			match_in >> lower_row[i];
		}

		// Load the buffers into matrix
		Eigen::Matrix<int, 1, Eigen::Dynamic> upper_row_mat(1, matching_number);
		Eigen::Matrix<int, 1, Eigen::Dynamic> lower_row_mat(1, matching_number);
		Eigen::Matrix<int, 2, Eigen::Dynamic> matchings_mat(2, matching_number);

		upper_row_mat = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic>>(upper_row.data(), 1, matching_number);
		lower_row_mat = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic>>(lower_row.data(), 1, matching_number);
		matchings_mat << upper_row_mat, lower_row_mat;
		matching_mat[left_index][right_index] = matchings_mat;

		// Increase the pair of matchings counter by 1
		pair_num++;
	}
}

void Matching::display_matchings(
	Image_info& image_left_,
	Image_info& image_right_,
	cv::Mat&	mask_
)
{
	// Define color for the keypoint
	cv::Scalar color(0, 255, 0);
	cv::Scalar line_color(0, 0, 255);

	// Get indices for the left and right images
	int left_index  = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	// Get matching number for the pair
	const int matching_number = matching_number_mat(left_index, right_index);
	assert(matching_number > 0);

	// Check if mask is passed in
	if (mask_.rows == 1) {
		mask_ = cv::Mat(matching_number, 1, CV_8U, cv::Scalar(1));
	}

	// Get images
	cv::Mat left_img  = image_left_.getImage();
	cv::Mat right_img = image_right_.getImage();

	const int lwidth  = image_left_.getWidth();
	const int lheight = image_left_.getHeight();
	const int rwidth  = image_right_.getWidth();
	const int rheight = image_right_.getHeight();

	// Get keypoints coordinates
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords1 = image_left_.get_coordinates();
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords2 = image_right_.get_coordinates();

	// Declare a new container and fill it with the two images
	cv::Mat concatenated(std::max(lheight, rheight), lwidth + rwidth, CV_8UC3, cv::Scalar(0));
	cv::Mat concat_left(concatenated,  cv::Rect(0, 0, lwidth, lheight));
	cv::Mat concat_right(concatenated, cv::Rect(lwidth, 0, rwidth, rheight));
	left_img.copyTo(concat_left);
	right_img.copyTo(concat_right);

	// Retrieve the matchings between the two images
	const Eigen::Matrix<int, 2, Eigen::Dynamic> matchings = matching_mat[left_index][right_index];

	for (int i = 0; i < matching_number; i++) {

		// Display only a portion of the keypoints
		//if (i % 13 != 0 || i % 15 != 0) {
		//	continue;
		//}

		const int upper_index = matchings(0, i);
		const int lower_index = matchings(1, i);

		const float x_upper = coords1(0, upper_index);
		const float y_upper = coords1(1, upper_index);
		const float x_lower = coords2(0, lower_index);
		const float y_lower = coords2(1, lower_index);

		if (x_upper < lwidth && y_upper < lheight && x_lower < rwidth && y_lower < rheight &&
			mask_.at<bool>(i, 0) == true && x_upper >= 0 && y_upper >= 0 && x_lower >= 0 && y_lower >= 0) {
			cv::Point left_pt(x_upper, y_upper);
			cv::Point right_pt(x_lower + lwidth, y_lower);

			cv::circle(concatenated, left_pt, 1, color, 3);
			cv::line(concatenated, left_pt, right_pt, line_color, 1, CV_AA);
		}
	}

	cv::imshow("Matchings", concatenated);
	cv::waitKey();
}

cv::Mat Matching::get_denseArea_mask(
	Image_info& image_left_,
	Image_info& image_right_
)
{
	// ===== Note that this only applies to the image in the right side of the matching =====

	// Get indices for the left and right images
	int left_index  = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	// Retrieve matching number
	const int matching_number = matching_number_mat(left_index, right_index);
	assert(matching_number > 0);

	// Retrieve matching matrix
	const Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings = matching_mat[left_index][right_index];

	// Get keypoints coordinates
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords = image_right_.get_coordinates();

	// Declare the mask container
	const int img_width  = image_right_.getWidth();
	const int img_height = image_right_.getHeight();
	cv::Mat mask(img_height, img_width, CV_8UC1, cv::Scalar(0));

	for (int i = 0; i < matching_number; i++) {
		const int lower_index = matchings(1, i);
		const float x = coords(0, lower_index);
		const float y = coords(1, lower_index);

		if (x < img_width && y < img_height && x >= 0 && y >= 0) {
			mask.at<uchar>(y, x) = 255;
		}
	}

	// Dilate the mask to cover up a border region of the dense area
	int dilation_size		= 8;
	int dilation_type		= cv::MORPH_ELLIPSE;
	cv::Mat dilate_element	= cv::getStructuringElement(
							  dilation_type,
							  cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
							  cv::Point(dilation_size, dilation_size)
							  );
	dilate(mask, mask, dilate_element);

	// Return mask
	return mask;
}

void Matching::compute_fundamental(
	Image_info& image_left_,
	Image_info& image_right_,
	cv::Mat&	F_,
	cv::Mat&	H1_,
	cv::Mat&	H2_,
	cv::Mat&	left_warped_
)
{
	// Get indices for the left and right images
	int left_index  = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	// Retrieve matching number
	const int matching_number = matching_number_mat(left_index, right_index);
	assert(matching_number > 0);

	// Retrieve matching matrix
	const Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings = matching_mat[left_index][right_index];

	// Get keypoints coordinates
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords1 = image_left_.get_coordinates();
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords2 = image_right_.get_coordinates();

	// Get height and width of the two images
	const int heightl = image_left_.getHeight();
	const int widthl  = image_left_.getWidth();
	const int heightr = image_right_.getHeight();
	const int widthr  = image_right_.getWidth();

	// Load the coordinates into opencv structures
	std::vector<cv::Point2f> points1(matching_number);
	std::vector<cv::Point2f> points2(matching_number);
	points1.reserve(matching_number);
	points2.reserve(matching_number);

	for (int i = 0; i < matching_number; i++) {
		const int upper_index = matchings(0, i);
		const int lower_index = matchings(1, i);

		const float x_upper = coords1(0, upper_index);
		const float y_upper = coords1(1, upper_index);
		const float x_lower = coords2(0, lower_index);
		const float y_lower = coords2(1, lower_index);

		if (x_upper < widthl && y_upper < heightl && x_lower < widthr && y_lower < heightr &&
			x_upper >= 0 && y_upper >= 0 && x_lower >= 0 && y_lower >= 0) {
			points1.push_back(cv::Point2f(x_upper, y_upper));
			points2.push_back(cv::Point2f(x_lower, y_lower));
		}
	}

	// Compute Fundemental matrxi and rectification matrix
	F_ = findFundamentalMat(points1, points2, CV_FM_RANSAC, 0);
	cv::stereoRectifyUncalibrated(points1, points2, F_, image_right_.getImage().size(), H1_, H2_, 1);
	cv::warpPerspective(image_left_.getImage(), left_warped_, H1_, image_right_.getImage().size());
}

void Matching::compute_homography(
	Image_info& image_left_,
	Image_info& image_right_,
	cv::Mat&	H_,
	cv::Mat&	left_warped_
)
{
	// Get indices for the left and right images
	int left_index = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	// Retrieve matching number
	const int matching_number = matching_number_mat(left_index, right_index);
	assert(matching_number > 0);

	// Retrieve matching matrix
	const Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings = matching_mat[left_index][right_index];

	// Get keypoints coordinates
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords1 = image_left_.get_coordinates();
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords2 = image_right_.get_coordinates();

	// Get height and width of the two images
	const int heightl = image_left_.getHeight();
	const int widthl  = image_left_.getWidth();
	const int heightr = image_right_.getHeight();
	const int widthr  = image_right_.getWidth();

	// Load the coordinates into opencv structures
	std::vector<cv::Point2f> points1;
	std::vector<cv::Point2f> points2;
	points1.reserve(matching_number);
	points2.reserve(matching_number);

	for (int i = 0; i < matching_number; i++) {
		const int upper_index = matchings(0, i);
		const int lower_index = matchings(1, i);
		
		const float x_upper = coords1(0, upper_index);
		const float y_upper = coords1(1, upper_index);
		const float x_lower = coords2(0, lower_index);
		const float y_lower = coords2(1, lower_index);

		if (x_upper < widthl && y_upper < heightl && x_lower < widthr && y_lower < heightr &&
			x_upper >= 0 && y_upper >= 0 && x_lower >= 0 && y_lower >= 0) {
			points1.push_back(cv::Point2f(x_upper, y_upper));
			points2.push_back(cv::Point2f(x_lower, y_lower));
		}
	}

	// Compute homography
	cv::Mat outlier_mask;
	H_ = findHomography(points1, points2, HOMOGRAPHY_METHOD, 1.5, outlier_mask);
	warpPerspective(image_left_.getImage(), left_warped_, H_, image_right_.getImage().size());

	// Store the outlier_mask and set the indicator to 1
	outlier_mask_mat[left_index][right_index] = outlier_mask;
	homography_existence_indicator(left_index, right_index) = true;
}

int Matching::get_matchings(
	Image_info& image_left_,
	Image_info& image_right_,
	Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings
)
{
	// Get indices for the left and right images
	int left_index = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	// Return the desired matchings
	const int matching_number = matching_number_mat(left_index, right_index);
	
	// Decide if the two images have match
	if (matching_number > 0) {
		matchings = matching_mat[left_index][right_index];
	}
	else {
		matchings = Eigen::Matrix<int, 2, Eigen::Dynamic>::Zero(2, 1);
	}

	return matching_number;
}

void Matching::delete_bad_matchings(
	Image_info& image_left_,
	Image_info& image_right_
)
{
	// Get indices for the left and right images
	int left_index  = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	// Return the desired matchings
	const int matching_number = matching_number_mat(left_index, right_index);
	assert(matching_number > 0);

	// Retrieve matching matrix
	const Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings = matching_mat[left_index][right_index];

	// Get keypoints coordinates
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords1 = image_left_.get_coordinates();
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords2 = image_right_.get_coordinates();

	// Get heights and widths of the images
	const int widthl  = image_left_.getWidth();
	const int heightl = image_left_.getHeight();
	const int widthr  = image_right_.getWidth();
	const int heightr = image_right_.getHeight();

	std::vector<int> filtered_matches;
	filtered_matches.reserve(2 * matching_number);

	for (int i = 0; i < matching_number; i++) {
		const int upper_index = matchings(0, i);
		const int lower_index = matchings(1, i);

		const float x_upper = coords1(0, upper_index);
		const float y_upper = coords1(1, upper_index);
		const float x_lower = coords2(0, lower_index);
		const float y_lower = coords2(1, lower_index);

		if (x_upper < widthl && y_upper < heightl && x_lower < widthr && y_lower < heightr &&
			x_upper >= 0 && y_upper >= 0 && x_lower >= 0 && y_lower >= 0) {
			filtered_matches.push_back(upper_index);
			filtered_matches.push_back(lower_index);
		}
	}

	Eigen::Matrix<int, 2, Eigen::Dynamic> pruned_match(2, filtered_matches.size() / 2);
	pruned_match = Eigen::Map<Eigen::Matrix<int, 2, Eigen::Dynamic>>(filtered_matches.data(), 2, filtered_matches.size() / 2);

	// Update the matching number and matching matrix
	matching_number_mat(left_index, right_index) = pruned_match.cols();
	matching_mat[left_index][right_index] = pruned_match;
}

int Matching::get_matching_number(
	Image_info& image_left_,
	Image_info& image_right_
)
{
	// Get indices for the left and right images
	int left_index = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	return matching_number_mat(left_index, right_index);
}

cv::Mat Matching::get_outlier_mask(
	Image_info& image_left_,
	Image_info& image_right_
)
{
	// Get indices for the left and right images
	int left_index = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	if (homography_existence_indicator(left_index, right_index) == true) {
		return outlier_mask_mat[left_index][right_index];
	}
	else {
		return cv::Mat(1, 1, CV_8U);
	}
}

void Matching::rectify_matchings_homoMask()
{
	for (int i = 0; i < image_num - 1; i++) {

		for (int j = i + 1; j < image_num; j++) {

			if (homography_existence_indicator(i, j) == true) {

				// Get matching information for processing
				const int matching_number								= matching_number_mat(i, j);
				const cv::Mat& outlier_mask								= outlier_mask_mat[i][j];
				const Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings	= matching_mat[i][j];
				assert(matching_number == outlier_mask.rows);

				std::vector<int> filtered_matches;
				filtered_matches.reserve(2 * matching_number);

				// Push back all valid matches
				for (int k = 0; k < matching_number; k++) {
					if (outlier_mask.at<bool>(k, 0) == true) {
						filtered_matches.push_back(matchings(0, k));
						filtered_matches.push_back(matchings(1, k));
					}
				}

				// Load the valid matches into a Eigen matrix
				Eigen::Matrix<int, 2, Eigen::Dynamic> pruned_match(2, filtered_matches.size() / 2);
				pruned_match = Eigen::Map<Eigen::Matrix<int, 2, Eigen::Dynamic>>(filtered_matches.data(), 2, filtered_matches.size() / 2);

				// Update the matching number and matching matrix
				matching_number_mat(i, j) = pruned_match.cols();
				matching_mat[i][j] = pruned_match;
			}
		}
	}
}

void Matching::write_matches(std::vector<Graph_disamb>& graphs_)
{
	// Compose the path to the correspoinding matchings.txt
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path, path, name);

	const int graph_number = graphs_.size();
	// Begin writing matches
	for (int i = 0; i < graph_number; i++) {

		// Compose output path for current graph
		std::string path_matches = path + "/matchings_" + std::to_string(i) + ".txt";
		std::ofstream match_out(path_matches.c_str(), std::ios::out);
		assert(match_out.is_open());

		// Retrieve the current graph layout
		const Eigen::MatrixXi layout = graphs_[i].getLayout();
		assert(image_num == layout.rows());
		assert(image_num == layout.cols());

		for (int j = 0; j < image_num - 1; j++) {
			for (int k = j + 1; k < image_num; k++) {
				const int matching_number = matching_number_mat(j, k);

				if (matching_number > 0 && layout(j, k) == 1) {
					match_out << img_names[j] << std::endl;
					match_out << img_names[k] << std::endl;
					match_out << matching_number << std::endl;

					// Retrieve the matching matrix
					const Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings = matching_mat[j][k];
					// Write out upper indices
					for (int l = 0; l < matching_number; l++) {
						match_out << matchings(0, l) << " ";
					}
					match_out << std::endl;
					// Write out lower indices
					for (int l = 0; l < matching_number; l++) {
						match_out << matchings(1, l) << " ";
					}
					match_out << std::endl << std::endl;
				}
			}
		}

		// Close the output stream
		match_out.close();
	}
}

void Matching::write_layout(std::vector<Graph_disamb>& graphs_)
{
	// Compose the path to the correspoinding matchings.txt
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path, path, name);

	const int graph_number = graphs_.size();
	// Begin writing layouts
	for (int i = 0; i < graph_number; i++) {
		// Compose output path for current graph
		std::string path_layout = path + "/GraphLog_" + std::to_string(i) + ".txt";
		std::ofstream layout_out(path_layout.c_str(), std::ios::out);
		assert(layout_out.is_open());

		// Retrieve the current graph layout
		const Eigen::MatrixXi layout = graphs_[i].getLayout();
		assert(image_num == layout.rows());
		assert(image_num == layout.cols());

		for (int j = 0; j < image_num - 1; j++) {

			layout_out << "Node(" << std::to_string(j) << ") : ";
			for (int k = j + 1; k < image_num; k++) {

				if (layout(j, k) == 1) {
					layout_out << std::to_string(k) << "(" << matching_number_mat(j, k) << ") ";
				}
			}
			layout_out << std::endl;
		}

		// Close the output stream
		layout_out.close();
	}
}

void Matching::setWarped_diff(const int row_, const int col_, const float value)
{
	// The information is stored only in upper triangle
	assert(row_ < col_);
	warped_diff_mat(row_, col_) = value;
}

Eigen::MatrixXf Matching::getWarped_diff()
{
	return warped_diff_mat;
}

Eigen::MatrixXi Matching::getMatching_number()
{
	return matching_number_mat;
}