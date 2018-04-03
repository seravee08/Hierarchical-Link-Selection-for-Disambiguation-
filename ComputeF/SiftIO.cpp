#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <stdlib.h>

#include <opencv2/xfeatures2d.hpp>

#include "SiftIO.h"
#include "Parameters.h"
#include "standalone_image.h"
#include "gist.h"

#define _SIZE_INT				sizeof(int)
#define _SIZE_FLOAT				sizeof(float)
#define _SIZE_UNSIGNED_CHAR		sizeof(unsigned char)
#define _PI						3.14159

const cls::GISTParams DEFAULT_GIST_PARAMS{ true, 32, 32, 4, 3,{ 8, 8, 4 } };

Image_info::Image_info(const std::string img_name_):
	img_name	(img_name_),
	sift_name	(extract_SIFT_name(img_name_)),
	aux_name	(extract_AUX_name(img_name_)),
	aux_exist	(false),
	sift_exist  (false)
{
	image = cv::imread(img_name_.c_str(), CV_LOAD_IMAGE_COLOR);
	assert(image.channels() == 3);

	// Get the height and width of the image
	width  = image.cols;
	height = image.rows;

	// Initialize the camera matrix for the current camera
	K = (cv::Mat_<double>(3, 3)
		<<	35,	0,	width / 2.0,
			0,	35,	height / 2.0,
			0,	0,	1
		);
}

void Image_info::splitFilename(
	const std::string target_,
	std::string& path_,
	std::string& name_
)
{
	// Get the directory of the target
	unsigned found			= target_.find_last_of("/\\");
	path_					= target_.substr(0, found);

	// Get the object name itself without post fix (e.g. .jpg, .png etc.)
	std::string full_name	= target_.substr(found + 1);
	found					= full_name.find_last_of(".");
	name_					= full_name.substr(0, found);
}

std::string Image_info::extract_SIFT_name(const std::string img_name_)
{
	std::string des_path;
	std::string des_name;
	std::string post_fix = ".sift";
	std::string sift_name;

	// Split the directory into path and object name
	splitFilename(img_name_, des_path, des_name);
	sift_name = des_path + "/" + des_name + post_fix;

	// Return the composited name
	return sift_name;
} 

std::string Image_info::extract_AUX_name(const std::string img_name_)
{
	std::string des_path;
	std::string des_name;
	std::string post_fix = ".aff";
	std::string aux_name;

	// Split the directory into path and object name
	splitFilename(img_name_, des_path, des_name);
	aux_name = des_path + "/" + des_name + post_fix;

	// Return the composited name
	return aux_name;
}

std::vector<float> Image_info::compute_gist(cv::Mat target_)
{
	std::vector<float> result;
	cls::GIST gist_ext(DEFAULT_GIST_PARAMS);
	gist_ext.extract(target_, result);

	return result;
}

float Image_info::compute_gist_dist(
	std::vector<float> desc1_,
	std::vector<float> desc2_
)
{
	// Validate two descriptors
	assert(desc1_.size() == desc2_.size());

	float dist = 0.0;
	for (int i = 0; i < desc1_.size(); i++) {
		dist += (desc1_[i] - desc2_[i]) * (desc1_[i] - desc2_[i]);
	}
	
	return std::sqrt(dist);
}

void Image_info::read_Auxililiary()
{
	// Detect if auxiliary information has already been read
	if (aux_exist) {
		return;
	}

	// ===== Open input file stream =====
	std::ifstream aux_in(aux_name.c_str(), std::ios::in | std::ios::binary);
	assert(aux_in.is_open());

	// ===== Read in affine information =====

	// Read in Basic information
	int tilts;
	int aux_width;
	int aux_height;

	aux_in.read((char*)&aux_width,	_SIZE_INT);
	aux_in.read((char*)&aux_height,	_SIZE_INT);
	aux_in.read((char*)&feat_num,	_SIZE_INT);
	aux_in.read((char*)&tilts,		_SIZE_INT);

	assert(aux_width  == width);
	assert(aux_height == height);

	// Read in rotation information
	int* rots = new int[tilts];
	aux_in.read((char*)rots, tilts * _SIZE_INT);

	// Read in keypoints number under each tilt and rotation combination
	int** keypoints_num = new int*[tilts];
	std::vector<std::vector<int>> asift_struct(tilts);

	for (int i = 0; i < tilts; i++) {

		// Read in number keypoints
		keypoints_num[i] = new int[rots[i]];
		aux_in.read((char*)keypoints_num[i], rots[i] * _SIZE_INT);

		// Store the number of keypoints into the structure for later use
		asift_struct[i].resize(rots[i]);

		for (int j = 0; j < rots[i]; j++) {
			asift_struct[i][j] = keypoints_num[i][j];
		}
	}

	// Read in auxiliary information
	float* aux_info = new float[feat_num * 2];
	aux_in.read((char*)aux_info, feat_num * 2 * _SIZE_FLOAT);

	// Load the information into Matrix member
	auxInfo_mat = Eigen::Matrix<float, 2, Eigen::Dynamic>(2, feat_num);
	auxInfo_mat = Eigen::Map<Eigen::Matrix<float, 2, Eigen::Dynamic>>(aux_info, 2, feat_num);

	// Close the input stream
	aux_in.close();

	// Set auxiliary information existence to true
	aux_exist = true;

	// Buffer clean up
	delete[] rots;
	delete[] aux_info;

	for (int i = 0; i < tilts; i++) {
		delete[] keypoints_num[i];
	}
	delete[] keypoints_num;
}

void Image_info::read_Sift()
{
	// Detect if the Sift information has already been read
	if (sift_exist) {
		return;
	}

	// Have to read in auxiliary information before reading in sift
	if (!aux_exist) {
		std::cout << "Auxiliary iniformation does not exists ..." << std::endl;
		exit(1);
	}

	// ===== Open input file stream =====
	std::ifstream sift_in(sift_name.c_str(), std::ios::in | std::ios::binary);
	assert(sift_in.is_open());

	// ===== Read in Sift information =====

	// Read in Sift headers
	int headers[5];
	sift_in.read((char*)headers, 5 * _SIZE_INT);
	assert(feat_num == headers[2]);

	// Read in coordinates and descriptors
	float*			coordinates	= new float[feat_num * 5];
	unsigned char*	descriptor	= new unsigned char[feat_num * 128];
	sift_in.read((char*)coordinates, feat_num * 5 * _SIZE_FLOAT);
	sift_in.read((char*)descriptor,	 feat_num * 128 * _SIZE_UNSIGNED_CHAR);

	// Load buffer into matrix
	Eigen::Matrix<float,		 5,		Eigen::Dynamic> coordinates_mat(5, feat_num);
	Eigen::Matrix<unsigned char, 128,	Eigen::Dynamic> desc_mat(128, feat_num);
	coordinates_mat = Eigen::Map<Eigen::Matrix<float, 5, Eigen::Dynamic>>(coordinates, 5, feat_num);
	desc_mat = Eigen::Map<Eigen::Matrix<unsigned char, 128, Eigen::Dynamic>>(descriptor, 128, feat_num);

	// Convert descriptor matrix from unsigned char to float
	descriptor_mat = Eigen::Matrix<float, 128, Eigen::Dynamic>(128, feat_num);
	descriptor_mat = desc_mat.cast<float>();

	// Adjust keypoints matrix
	keypoints_mat = Eigen::Matrix<float, 6, Eigen::Dynamic>(6, feat_num);
	keypoints_mat << coordinates_mat.topRows(2), coordinates_mat.bottomRows(2), auxInfo_mat;
	keypoints_mat.row(2) = keypoints_mat.row(2).cwiseProduct(keypoints_mat.row(4));
	keypoints_mat.row(5) *= _PI / 180.0;
	keypoints_mat.row(3) *= -1;

	// Read in sift end of file marker
	int eof_marker;
	sift_in.read((char*)&eof_marker, _SIZE_INT);

	// Close input file stream
	sift_in.close();

	// Set the sift existence flag to true
	sift_exist = true;

	// Clean up buffer
	delete[] coordinates;
	delete[] descriptor;
}

void Image_info::compute_Sift() {

	// Initiate SIFT detector
	cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();

	if (image.channels() == 3) {
		cv::Mat gray_img;
		cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
		f2d->detect(gray_img, keypoints);
		f2d->compute(gray_img, keypoints, descriptors);
	}
	else {
		assert(image.channels() == 1);
		f2d->detect(image, keypoints);
		f2d->compute(image, keypoints, descriptors);
	}

	sift_exist	= true;
	feat_num	= keypoints.size();
}

float Image_info::compute_difference(
	const cv::Mat& warped_,
	const cv::Mat& mask_,
	const bool greyScale
)
{
	// Validate height and width of the image
	assert(width	== warped_.cols);
	assert(height	== warped_.rows);
	assert(width	== mask_.cols);
	assert(height	== mask_.rows);

	int diff = 0;
	int pixel_cntr = 0;
	int blank_pixels_cntr = 0;

	if (greyScale) {
		// Convert RGB image into grey image
		cv::Mat warped_grey, equalized_warped_grey;
		cv::Mat origin_grey, equalized_origin_grey;
		cv::cvtColor(warped_, warped_grey, CV_RGB2GRAY);
		cv::cvtColor(image, origin_grey, CV_RGB2GRAY);
		cv::equalizeHist(warped_grey, equalized_warped_grey);
		cv::equalizeHist(origin_grey, equalized_origin_grey);

		// Compute the difference and calculate the number of blank pixels in the warped image	
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (mask_.at<uchar>(i, j) == 0 && equalized_warped_grey.at<uchar>(i, j) != 0) {
					diff += abs(equalized_origin_grey.at<uchar>(i, j) - equalized_warped_grey.at<uchar>(i, j));
					pixel_cntr++;
				}
				else if (equalized_warped_grey.at<uchar>(i, j) == 0) {
					blank_pixels_cntr++;
				}
			}
		}
	}
	else {
		// Compute the difference and calculate the number of blank pixels in the warped image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (mask_.at<uchar>(i, j) == 0 && warped_.at<cv::Vec3b>(i, j)[0] != 0
					&& warped_.at<cv::Vec3b>(i, j)[1] != 0 && warped_.at<cv::Vec3b>(i, j)[2] != 0) {
					diff += abs(image.at<cv::Vec3b>(i, j)[0] - warped_.at<cv::Vec3b>(i, j)[0]);
					diff += abs(image.at<cv::Vec3b>(i, j)[1] - warped_.at<cv::Vec3b>(i, j)[1]);
					diff += abs(image.at<cv::Vec3b>(i, j)[2] - warped_.at<cv::Vec3b>(i, j)[2]);
					pixel_cntr++;
				}
				else if (warped_.at<cv::Vec3b>(i, j)[0] == 0 && warped_.at<cv::Vec3b>(i, j)[1] == 0 && warped_.at<cv::Vec3b>(i, j)[2] == 0) {
					blank_pixels_cntr++;
				}
			}
		}
	}
	
	// If there is less than a certain percentage of pixels overlap, the pair is rejected
	return (1.0 - (blank_pixels_cntr * 1.0f / (width * height)) > OVERLAP_THRESHOLD) ? (diff * 1.0f) / pixel_cntr : -2;
}

float Image_info::compute_difference_gist_color(
	const cv::Mat& warped_,
	const cv::Mat& mask_
)
{
	// Validate height and width of the image
	assert(width				== warped_.cols);
	assert(height				== warped_.rows);
	assert(width				== mask_.cols);
	assert(height				== mask_.rows);
	assert(image.channels()		== 3);
	assert(warped_.channels()	== 3);

	cv::Mat image_cpy  = image.clone();
	cv::Mat warped_cpy = warped_.clone();

	// Mask out the two images
	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width; j++) { 
	//		if (mask_.at<uchar>(i, j) == 0) {
	//			// Set original image blank
	//			image_cpy.at<cv::Vec3b>(i, j)[0] = 0;
	//			image_cpy.at<cv::Vec3b>(i, j)[1] = 0;
	//			image_cpy.at<cv::Vec3b>(i, j)[2] = 0;

	//			// Set warped image blank
	//			warped_cpy.at<cv::Vec3b>(i, j)[0] = 0;
	//			warped_cpy.at<cv::Vec3b>(i, j)[1] = 0;
	//			warped_cpy.at<cv::Vec3b>(i, j)[2] = 0;
	//		}
	//		else if (warped_cpy.at<cv::Vec3b>(i, j)[0] == 0 &&
	//			warped_cpy.at<cv::Vec3b>(i, j)[1] == 0 &&
	//			warped_cpy.at<cv::Vec3b>(i, j)[2] == 0) {

	//			// Set original image blank
	//			image_cpy.at<cv::Vec3b>(i, j)[0] = 0;
	//			image_cpy.at<cv::Vec3b>(i, j)[1] = 0;
	//			image_cpy.at<cv::Vec3b>(i, j)[2] = 0;
	//		}
	//	}
	//}

	std::vector<float> desc1 = Image_info::compute_gist(image_cpy);
	std::vector<float> desc2 = Image_info::compute_gist(warped_cpy);
	float dist = Image_info::compute_gist_dist(desc1, desc2);

	return dist;
}

void Image_info::display_keypoints()
{
	// Define dot color
	cv::Scalar color(0, 255, 0);

	// Draw points on the canvas
	cv::Mat image_disp = image.clone();
	for (int i = 0; i < feat_num; i++) {
		cv::Point pt(keypoints_mat(0, i), keypoints_mat(1, i));
		cv::circle(image_disp, pt, 1, color, 3);
	}

	cv::imshow("Keypoints", image_disp);
	cv::waitKey();
}

void Image_info::display_keypoints_locally_computed()
{
	// Define dot color
	cv::Scalar color(0, 255, 0);

	// Draw points on the canvas
	cv::Mat image_disp = image.clone();
	for (int i = 0; i < feat_num; i++) {
		cv::circle(image_disp, keypoints[i].pt, 1, color, 3);
	}

	cv::imshow("Keypoints", image_disp);
	cv::waitKey();
}

cv::Mat Image_info::blendImages(const cv::Mat& A, const cv::Mat& B)
{
	cv::Mat blended_img;
	cv::addWeighted(A, 0.5, B, 0.5, 0.0, blended_img);

	return blended_img;
}

void Image_info::freeSpace()
{
	// Manually free the space of the Eigen structure
	auxInfo_mat.resize(2, 0);
	keypoints_mat.resize(6, 0);
	descriptor_mat.resize(128, 0);
}

std::vector<cv::KeyPoint> Image_info::getKeypoints_locally_computed()
{
	return keypoints;
}

cv::Mat Image_info::getDescriptors_locally_computed()
{
	return descriptors;
}

int Image_info::getHeight()
{
	return height;
}

int Image_info::getWidth()
{
	return width;
}

std::string Image_info::getImageName()
{
	return img_name;
}

bool Image_info::getSiftStatus()
{
	return sift_exist;
}

bool Image_info::getAuxStatus()
{
	return aux_exist;
}

cv::Mat Image_info::getImage()
{
	return image;
}

cv::Mat Image_info::getK()
{
	return K;
}

Eigen::Matrix<float, 2, Eigen::Dynamic> Image_info::get_coordinates()
{
	return keypoints_mat.topRows(2);
}

Eigen::Matrix<float, 2, Eigen::Dynamic> Image_info::get_coordinates_locally_computed()
{
	const int local_feat_num = keypoints.size();
	if (local_feat_num <= 0) {
		std::cout << "No locally computed Sift points detected, exiting ..." << std::endl;
	}

	std::vector<float> coordinates(2 * local_feat_num);
	for (int i = 0; i < local_feat_num; i++) {
		coordinates[2 * i]		= keypoints[i].pt.x;
		coordinates[2 * i + 1]	= keypoints[i].pt.y;
	}

	Eigen::Matrix<float, 2, Eigen::Dynamic> coordinates_mat(2, local_feat_num);
	coordinates_mat = Eigen::Map<Eigen::Matrix<float, 2, Eigen::Dynamic>>(coordinates.data(), 2, local_feat_num);
	coordinates.clear();

	return coordinates_mat;
}