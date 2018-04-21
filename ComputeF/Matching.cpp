#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <stdlib.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "utility.h"
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
		// Check for empty names
		if (name.length() <= 0) {
			continue;
		}
		img_names.push_back(name);
	}

	// Initialize for the number of images
	image_num = img_names.size();

	// Initialize for the matching data strucutres
	matching_number_mat				= Eigen::MatrixXi::Zero(image_num, image_num);
	matching_number_mat_float		= Eigen::MatrixXf::Zero(image_num, image_num);
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
	if (!match_in.is_open()) {
		std::cout << "Matchings file does not exist ..." << std::endl;
		exit(1);
	}

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

	generate_float_matching_number_mat();
}

void Matching::generate_float_matching_number_mat()
{
	// TODO: Naive implementaion
	matching_number_mat_float = matching_number_mat.cast<float>();
}

void Matching::compute_Matchings_1v1(
	Image_info& image_left_,
	Image_info& image_right_,
	int left_index_,
	int right_index_
) 
{
	if (image_left_.getSiftStatus() == true && image_right_.getSiftStatus() == true) {
		// Create matcher
		cv::BFMatcher matcher(cv::NORM_L2);
		std::vector<std::vector<cv::DMatch>> matches;

		matcher.knnMatch(
			image_left_.getDescriptors_locally_computed(),
			image_right_.getDescriptors_locally_computed(),
			matches,
			2);

		// Find good matches
		std::vector<int> upper_row;
		std::vector<int> lower_row;
		upper_row.reserve(matches.size());
		lower_row.reserve(matches.size());

		for (int i = 0; i < matches.size(); i++) {
			assert(matches[i].size() == 2);
			if (matches[i][0].distance < SIFT_GOOD_THRESHOLD * matches[i][1].distance) {
				upper_row.push_back(matches[i][0].queryIdx);
				lower_row.push_back(matches[i][0].trainIdx);
			}
		}

		// Load the buffers into Eigen matrix structure
		assert(upper_row.size() == lower_row.size());
		const int matching_number = upper_row.size();
		Eigen::Matrix<int, 1, Eigen::Dynamic> upper_row_mat(1, matching_number);
		Eigen::Matrix<int, 1, Eigen::Dynamic> lower_row_mat(1, matching_number);
		Eigen::Matrix<int, 2, Eigen::Dynamic> matchings_mat(2, matching_number);

		upper_row_mat = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic>>(upper_row.data(), 1, matching_number);
		lower_row_mat = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic>>(lower_row.data(), 1, matching_number);
		matchings_mat << upper_row_mat, lower_row_mat;

		assert(left_index_ < right_index_);
		matching_mat[left_index_][right_index_] = matchings_mat;
		matching_number_mat(left_index_, right_index_) = matching_number;

		// Increase the pair of matchings counter by 1
		pair_num++;
	}
	else {
		std::cout << "One or more of the required Sift keypoints does/do not exist ..." << std::endl;
		exit(-1);
	}
}

void Matching::display_matchings(
	Image_info& image_left_,
	Image_info& image_right_,
	cv::Mat&	mask_,
	bool		locally_computed
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
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords1 = (locally_computed)? image_left_.get_coordinates_locally_computed() : image_left_.get_coordinates();
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords2 = (locally_computed)? image_right_.get_coordinates_locally_computed(): image_right_.get_coordinates();

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
		if (i % 13 != 0 || i % 15 != 0) {
			continue;
		}

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

bool Matching::cameraPoseAndTriangulationFromFundamental(
	Image_info& image_left_,
	Image_info& image_right_,
	bool		locally_computed_
)
{
	// Get indices for the left and right images
	int left_index  = std::find(img_names.begin(), img_names.end(), image_left_.getImageName()) - img_names.begin();
	int right_index = std::find(img_names.begin(), img_names.end(), image_right_.getImageName()) - img_names.begin();

	if (left_index >= image_num || right_index >= image_num) {
		std::cout << "Cannot find image names in the storage ..." << std::endl;
		exit(1);
	}
	assert(left_index < right_index);

	// Retrieve matching number
	const int matching_number = matching_number_mat(left_index, right_index);
	assert(matching_number > 0);

	// Retrieve matching matrix
	const Eigen::Matrix<int, 2, Eigen::Dynamic>& matchings = matching_mat[left_index][right_index];

	// Get keypoints coordinates
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords1 = (locally_computed_)? image_left_.get_coordinates_locally_computed() : image_left_.get_coordinates();
	const Eigen::Matrix<float, 2, Eigen::Dynamic>& coords2 = (locally_computed_)? image_right_.get_coordinates_locally_computed() : image_right_.get_coordinates();

	// Get height and width of the two images
	const int heightl	= image_left_.getHeight();
	const int widthl	= image_left_.getWidth();
	const int heightr	= image_right_.getHeight();
	const int widthr	= image_right_.getWidth();

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

	// Load the valid coordinates into the opencv mat strucutres
	assert(points1.size() == points2.size());
	const int valid_points = points1.size();

	cv::Mat points1_mat(1, valid_points, CV_32FC2);
	cv::Mat points2_mat(1, valid_points, CV_32FC2);

	for (int i = 0; i < valid_points; i++) {
		points1_mat.at<cv::Point2f>(i) = points1[i];
		points2_mat.at<cv::Point2f>(i) = points2[i];
	}

	// ===== Compute Fundamental Matrix =====
	double minVal;
	double maxVal;
	std::vector<uchar> points_status;
	cv::minMaxIdx(points1, &minVal, &maxVal);

	cv::Mat F = cv::findFundamentalMat(points1, points2, CV_FM_RANSAC, 0.006 * maxVal, 0.99, points_status);
	const int inliers_num = cv::countNonZero(points_status);
	std::cout << "Image pair (" << left_index << "," << right_index << ") inliers percentage: " << inliers_num * 1.0f / points_status.size() * 100 << "%" << std::endl;

	// Retrieve the K matrices for the two cameras
	const cv::Mat K1 = image_left_.getK();
	const cv::Mat K2 = image_right_.getK();

	if (inliers_num > MIN_INLIERS) {
		// Calculate essential matrix from fundamental matrix
		cv::Mat_<double> E = K2.t() * F * K1;

		if (std::fabsf(cv::determinant(E)) > 1e-07) {
			std::cout << "Determinant of E not equal to 0: " << cv::determinant(E) << std::endl;
			return false;
		}

		cv::Mat_<double> R1(3, 3);
		cv::Mat_<double> R2(3, 3);
		cv::Mat_<double> t1(1, 3);
		cv::Mat_<double> t2(1, 3);

		if (!decomposeEtoRandT(E, R1, R2, t1, t2)) {
			return false;
		}

		if (cv::determinant(R1) + 1.0 < 1e-09) {
			std::cout << "Determinant of R equals to -1, flipping sign" << std::endl;
			E = -E;

			if (!decomposeEtoRandT(E, R1, R2, t1, t2)) {
				return false;
			}
		}

		if (std::fabsf(cv::determinant(R1)) - 1.0 > 1e-07) {
			std::cout << "Determinant of R not equal to +-1.0, this is not a rotation matrix" << std::endl;
			return false;
		}

		// Projection matrix of left camera
		cv::Mat P1 = cv::Mat::eye(3, 4, CV_64FC1);
		cv::Mat P2 = (cv::Mat_<double>(3, 4) <<
			R1(0, 0), R1(0, 1), R1(0, 2), t1(0),
			R1(1, 0), R1(1, 1), R1(1, 2), t1(1),
			R1(2, 0), R1(2, 1), R1(2, 2), t1(2)
			);

		cv::Mat pts_3d;
		bool triangulationSucceeded = true;
		if (!triangulateAndCheckReproj(P1, P2, points1, points2, points1_mat, points2_mat, K1, K2, pts_3d)) {
			P2 = (cv::Mat_<double>(3, 4) <<
				R1(0, 0), R1(0, 1), R1(0, 2), t2(0),
				R1(1, 0), R1(1, 1), R1(1, 2), t2(1),
				R1(2, 0), R1(2, 1), R1(2, 2), t2(2)
				);

			if (!triangulateAndCheckReproj(P1, P2, points1, points2, points1_mat, points2_mat, K1, K2, pts_3d)) {
				P2 = (cv::Mat_<double>(3, 4) <<
					R2(0, 0), R2(0, 1), R2(0, 2), t2(0),
					R2(1, 0), R2(1, 1), R2(1, 2), t2(1),
					R2(2, 0), R2(2, 1), R2(2, 2), t2(2)
					);

				if (!triangulateAndCheckReproj(P1, P2, points1, points2, points1_mat, points2_mat, K1, K2, pts_3d)) {
					P2 = (cv::Mat_<double>(3, 4) <<
						R2(0, 0), R2(0, 1), R2(0, 2), t1(0),
						R2(1, 0), R2(1, 1), R2(1, 2), t1(1),
						R2(2, 0), R2(2, 1), R2(2, 2), t1(2)
						);

					if (!triangulateAndCheckReproj(P1, P2, points1, points2, points1_mat, points2_mat, K1, K2, pts_3d)) {
						std::cout << "Cannot find te right projection matrix ..." << std::endl;
						triangulationSucceeded = false;
					}
				}
			}
		}


		// Output the 3D points as .ply file to be viewed in meshViewer
		if (triangulationSucceeded) {

			std::string path;
			std::string name;
			Image_info::splitFilename(image_list_path, path, name);
			std::string composed_name = path + "/" + std::to_string(left_index) + "_" + std::to_string(right_index) + ".ply";
			points_to_ply(composed_name, pts_3d);
		}

		return triangulationSucceeded;
	}

	return false;
}

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
)
{
	const int valid_points = points1_mat.cols;
	assert(points2_mat.cols == valid_points);
	assert(points1.size() == points2.size());
	assert(points1.size() == valid_points);

	// Undistort points
	cv::Mat normalized_points1;
	cv::Mat normalized_points2;

	cv::undistortPoints(points1_mat, normalized_points1, K1, cv::Mat());
	cv::undistortPoints(points2_mat, normalized_points2, K2, cv::Mat());

	// Triangulate points
	cv::Mat pts_3d_homo(4, valid_points, CV_32FC1);
	cv::triangulatePoints(P1, P2, normalized_points1, normalized_points2, pts_3d_homo);
	cv::convertPointsFromHomogeneous(cv::Mat(pts_3d_homo.t()).reshape(4, 1), pts_3d);
	assert(pts_3d.rows == valid_points);

	// Compute how many points are in front of the camera
	std::vector<uchar> points_status(valid_points, 0);
	for (int i = 0; i < valid_points; i++) {
		points_status[i] = (pts_3d.at<cv::Point3f>(i).z > 0) ? 1 : 0;
	}
	const int num_inFrontOfCamera = cv::countNonZero(points_status);

	float points_inFrontOfCamera_ratio = num_inFrontOfCamera * 1.0f / valid_points;
	std::cout << points_inFrontOfCamera_ratio * 100 << "% points are in front of the camera ..." << std::endl;
	if (points_inFrontOfCamera_ratio < 0.75) {
		return false;
	}

	// Calculate reprojection
	cv::Vec3f rvec(0, 0, 0);
	cv::Vec3f tvec(0, 0, 0);
	std::vector<cv::Point2f> reprojected_pt_set;
	cv::projectPoints(pts_3d, rvec, tvec, K1, cv::Mat(), reprojected_pt_set);
	float reprojectionErr = cv::norm(cv::Mat(reprojected_pt_set), cv::Mat(points1), cv::NORM_L2) * 1.0f / valid_points;
	std::cout << "Reprojection error: " << reprojectionErr << std::endl;

	if (reprojectionErr < 5) {
		return true;
	}

	return false;
}

bool Matching::decomposeEtoRandT(
	cv::Mat_<double>& E,
	cv::Mat_<double>& R1,
	cv::Mat_<double>& R2,
	cv::Mat_<double>& t1,
	cv::Mat_<double>& t2
)
{
	cv::SVD svd(E, cv::SVD::MODIFY_A);

	// Check if first and second singular values are the same (should be the same)
	double singular_values_ratio = std::fabsf(svd.w.at<double>(0) / svd.w.at<double>(1));
	
	if (singular_values_ratio > 1.0) {
		singular_values_ratio = 1.0 / singular_values_ratio;
	}

	if (singular_values_ratio < 0.7) {
		std::cout << "Singular values of essential matrix are too far apart" << std::endl;
		return false;
	}

	cv::Mat W  = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	cv::Mat Wt = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);

	R1 = svd.u * W * svd.vt;
	R2 = svd.u * Wt * svd.vt;
	t1 = svd.u.col(2);
	t2 = -svd.u.col(2);

	return true;
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

	if (left_index >= image_num || right_index >= image_num || left_index < 0 || right_index < 0) {
		std::cout << "Cannot find image names in the vector ..." << std::endl;
		exit(1);
	}
	if (left_index == right_index) {
		std::cout << "get_matching_number: invalid requests for matching number ..." << std::endl;
		return -1;
	}

	if (left_index > right_index) {
		SELF_DEFINE_SWAP(left_index, right_index);
	}

	return matching_number_mat(left_index, right_index);
}

int Matching::get_matching_number(
	int index_left_,
	int index_right_
)
{
	if (index_left_ >= image_num || index_left_ < 0 ||
		index_right_ >= image_num || index_right_ < 0 || index_left_ == index_right_) {
		std::cout << "get_matching_numer: invalid requests for matching number ..." << std::endl;
		return -1;
	}
	if (index_left_ > index_right_) {
		SELF_DEFINE_SWAP(index_left_, index_right_);
	}
	
	return matching_number_mat(index_left_, index_right_);
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

void Matching::write_matches(Graph_disamb& graph_)
{
	// Compose the path to the correspoinding matchings.txt
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path, path, name);

	// Compose output path for current graph
	std::string path_matches = path + "/matchings_global.txt";
	std::ofstream match_out(path_matches.c_str(), std::ios::out);
	assert(match_out.is_open());

	// Retrieve the current graph layout
	const Eigen::MatrixXi layout = graph_.getLayout();
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

void Matching::write_matches_1v1(int l_, int r_)
{
	if (l_ == r_ || l_ >= image_num || r_ >= image_num) {
		std::cout << "Invalid writing request, check index ..." << std::endl;
		return;
	}

	const int pair_match_number = matching_number_mat(l_, r_);
	if (pair_match_number <= 0) {
		std::cout << "Matchings do not exist between the required pair ..." << std::endl;
		return;
	}

	// Make sure l has lower index
	if (l_ > r_) {
		SELF_DEFINE_SWAP(l_, r_);
	}

	// Derive the .txt name to save out the matches
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path, path, name);
	std::string save_name = path + "/" + std::to_string(l_) + "_" + std::to_string(r_) + "_matches.txt";
	std::ofstream match_out(save_name.c_str(), std::ios::out);

	if (!match_out.is_open()) {
		std::cout << "File open failed ..." << std::endl;
		return;
	}

	match_out << img_names[l_] << std::endl;
	match_out << img_names[r_] << std::endl;
	match_out << pair_match_number << std::endl;

	const Eigen::Matrix<int, 2, Eigen::Dynamic>& pair_match_mat = matching_mat[l_][r_];

	for (int i = 0; i < pair_match_number; i++) {
		match_out << pair_match_mat(0, i) << " ";
	}
	match_out << std::endl;
	for (int i = 0; i < pair_match_number; i++) {
		match_out << pair_match_mat(1, i) << " ";
	}

	// Close the output stream
	match_out.close();
}

void Matching::write_matches_1vN(int l_, const std::vector<int>& subset)
{
	if (l_ < 0 || l_ >= image_num) {
		std::cout << "write_matches_1vN: Invalid request for matching computation, check the index ..." << std::endl;
		return;
	}

	// Resolve the output path
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path, path, name);
	std::string save_name = path + "/" + std::to_string(l_) + "_N_matches.txt";
	std::ofstream match_out(save_name.c_str(), std::ios::out);
	const int req_num = subset.size();

	if (!match_out.is_open()) {
		std::cout << "File open failed ..." << std::endl;
		return;
	}

	int output_cntr = 0;
	// Compute matchings between left_index_ and all the other images (N - 1)
	if (req_num == 0) {
		for (int i = 0; i < image_num; i++) {
			if (i == l_) {
				continue;
			}

			const int mach_num = matching_number_mat(std::min(l_, i), std::max(l_, i));
			if (mach_num > 0) {
				output_cntr++;
				match_out << img_names[std::min(l_, i)] << std::endl;
				match_out << img_names[std::max(l_, i)] << std::endl;
				match_out << mach_num << std::endl;

				const Eigen::Matrix<int, 2, Eigen::Dynamic>& pair_match_mat = matching_mat[std::min(l_, i)][std::max(l_, i)];
				for (int j = 0; j < mach_num; j++) {
					match_out << pair_match_mat(0, j) << " ";
				}
				match_out << std::endl;
				for (int j = 0; j < mach_num; j++) {
					match_out << pair_match_mat(1, j) << " ";
				}
				match_out << std::endl << std::endl;
			}
		}
	}
	else {
		for (int i = 0; i < req_num; i++) {
			const int mach_num = matching_number_mat(std::min(l_, subset[i]), std::max(l_, subset[i]));
			if (mach_num > 0) {
				output_cntr++;
				match_out << img_names[std::min(l_, subset[i])] << std::endl;
				match_out << img_names[std::max(l_, subset[i])] << std::endl;
				match_out << mach_num << std::endl;

				const Eigen::Matrix<int, 2, Eigen::Dynamic>& pair_match_mat = matching_mat[std::min(l_, subset[i])][std::max(l_, subset[i])];
				for (int j = 0; j < mach_num; j++) {
					match_out << pair_match_mat(0, j) << " ";
				}
				match_out << std::endl;
				for (int j = 0; j < mach_num; j++) {
					match_out << pair_match_mat(1, j) << " ";
				}
				match_out << std::endl << std::endl;
			}
		}
	}

	// Close the output stream
	match_out.close();

	if (output_cntr <= 0) {
		std::cout << "write_matches_1vN: no matchings found between request image and the rest ..." << std::endl;
		return;
	}
}

void Matching::write_matches_all()
{
	// Resolve the output path
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path, path, name);
	std::string save_name = path + "/matchings.txt";
	std::ofstream match_out(save_name.c_str(), std::ios::out);

	if (!match_out.is_open()) {
		std::cout << "File open failed ..." << std::endl;
		return;
	}

	int output_cntr = 0;
	// Compute matchings between left_index_ and all the other images (N - 1)
	for (int i = 0; i < image_num - 1; i++) {
		for (int j = i + 1; j < image_num; j++) {
			const int mach_num = matching_number_mat(i, j);
			if (mach_num > 0) {
				output_cntr++;
				match_out << img_names[i] << std::endl;
				match_out << img_names[j] << std::endl;
				match_out << mach_num << std::endl;

				const Eigen::Matrix<int, 2, Eigen::Dynamic>& pair_match_mat = matching_mat[i][j];
				for (int k = 0; k < mach_num; k++) {
					match_out << pair_match_mat(0, k) << " ";
				}
				match_out << std::endl;
				for (int k = 0; k < mach_num; k++) {
					match_out << pair_match_mat(1, k) << " ";
				}
				match_out << std::endl << std::endl;
			}
		}
	}

	// Close the output stream
	match_out.close();

	if (output_cntr <= 0) {
		std::cout << "write_matches_1vN: no matchings found between request image and the rest ..." << std::endl;
		return;
	}
}

std::string Matching::write_matches_designated(const std::string file_name_, const std::vector<cv::Point2i>& linkages)
{
	// Validation process
	const int link_nums = linkages.size();
	if (link_nums <= 0) {
		std::cout << "write_matches_designated: the linkages set is empty ..." << std::endl;
		return std::string("");
	}

	// Resolve the output path
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path, path, name);
	std::string save_name;
	(file_name_ == std::string("")) ? save_name = path + "/tmp_matches.txt" : save_name = path + "/" + file_name_;
	std::ofstream match_out(save_name.c_str(), std::ios::out | std::ios::app);

	if (!match_out.is_open()) {
		std::cout << "File open failed ..." << std::endl;
		return std::string("");
	}
	
	int output_cntr = 0;
	for (int i = 0; i < link_nums; i++) {
		const int l = std::min(linkages[i].x, linkages[i].y);
		const int r = std::max(linkages[i].x, linkages[i].y);
		const int mach_num = matching_number_mat(l, r);

		if (mach_num > 0) {
			output_cntr++;
			match_out << img_names[l] << std::endl;
			match_out << img_names[r] << std::endl;
			match_out << mach_num << std::endl;

			const Eigen::Matrix<int, 2, Eigen::Dynamic>& pair_match_mat = matching_mat[l][r];
			for (int j = 0; j < mach_num; j++) {
				match_out << pair_match_mat(0, j) << " ";
			}
			match_out << std::endl;
			for (int j = 0; j < mach_num; j++) {
				match_out << pair_match_mat(1, j) << " ";
			}
			match_out << std::endl << std::endl;
		}
	}
	
	// Close the output stream
	match_out.close();

	if (output_cntr <= 0) {
		std::cout << "write_matches_1vN: no matchings found between request image and the rest ..." << std::endl;
		return std::string("");
	}

	return save_name;
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

void Matching::points_to_ply(
	const std::string file_path,
	const cv::Mat& pts_3d
)
{
	// Create output file stream
	std::ofstream points_out(file_path.c_str(), std::ios::out);
	assert(points_out.is_open());

	// Validate the 3D points
	assert(pts_3d.cols == 1);
	assert(pts_3d.rows >= 1);
	assert(pts_3d.channels() == 3);
	const int points_num = pts_3d.rows;

	// Compose the header for ply file
	points_out << "ply" << std::endl;
	points_out << "format ascii 1.0" << std::endl;
	points_out << "element vertex " << points_num << std::endl;
	points_out << "property float32 x" << std::endl;
	points_out << "property float32 y" << std::endl;
	points_out << "property float32 z" << std::endl;
	points_out << "end_header" << std::endl;

	// Write out the 3D points coordinates
	for (int i = 0; i < points_num; i++) {
		const cv::Point3f& pt = pts_3d.at<cv::Point3f>(i);
		points_out << pt.x << " " << pt.y << " " << pt.z << std::endl;
	}

	// Close the output stream
	points_out.close();
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

Eigen::MatrixXi Matching::getMatching_number_mat()
{
	return matching_number_mat;
}

Eigen::MatrixXf Matching::getMatching_number_mat_float()
{
	return matching_number_mat_float;
}

std::string Matching::get_MAT_name(int index_)
{
	return Image_info::extract_MAT_name(img_names[index_]);
}