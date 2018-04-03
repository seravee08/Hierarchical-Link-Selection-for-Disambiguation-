#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <stdlib.h>

#include "Image_control.h"

Image_control::Image_control(const std::string image_list_path_) :
	image_list_path	(image_list_path_)
{
	// Read in the image list
	std::ifstream list_in(image_list_path.c_str(), std::ios::in);
	assert(list_in.is_open());

	std::string name;
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

		// Instantiate for class Image_info
		Image_info img(name);
		image_list.push_back(img);
	}

	// Initialize for the number of images
	image_num = img_names.size();

	list_in.close();
}

Image_control::Image_control(const std::vector<std::string>& image_names) {

	const int num = image_names.size();
	img_names = image_names;
	image_num = num;
	image_list.reserve(num);

	for (int i = 0; i < num; i++) {
		Image_info img(img_names[i]);
		image_list.push_back(img);
	}
}

Image_control::~Image_control()
{
	img_names.clear();
	image_list.clear();
}

void Image_control::read_Auxiliary()
{
	for (int i = 0; i < image_num; i++) {
		image_list[i].read_Auxililiary();
	}
}

void Image_control::read_Sift()
{
	for (int i = 0; i < image_num; i++) {
		image_list[i].read_Sift();
	}
}

void Image_control::compute_Sift(int index)
{
	if (index < 0) {
		for (int i = 0; i < image_num; i++) {
			image_list[i].compute_Sift();
		}
	}
	else {
		image_list[index].compute_Sift();
	}
}

void Image_control::read_single_Auxiliary(int index_)
{
	if (!image_list[index_].getAuxStatus()) {
		image_list[index_].read_Auxililiary();
	}
}

void Image_control::read_single_Sift(int index_)
{
	if (!image_list[index_].getSiftStatus()) {
		image_list[index_].read_Auxililiary();
		image_list[index_].read_Sift();
	}
}

void Image_control::displayImage(int index_)
{
	const cv::Mat& target = getImageInfo(index_).getImage();
	const std::string name = "Image" + std::to_string(index_);

	cv::imshow(name.c_str(), target);
	cv::waitKey();
}

void Image_control::display_group(std::vector<int> group) 
{
	// Retrieve the number of images to be displayed
	const int display_num = group.size();
	assert(display_num <= image_num);

	// Decide the layout of the output window
	int grid_width  = ceil(sqrt(display_num));
	int grid_height = (grid_width * (grid_width - 1) >= display_num) ? grid_width - 1 : grid_width;
	
	// Calculate the summed width and height
	int grand_width  = 0;
	int grand_height = 0;

	// Create vector to store the maximum height of each row
	std::vector<int> max_height_each_row(grid_height);

	// Compute the max width of the grand image
	for (int i = 0; i < grid_height; i++) {

		int max_height = 0;
		int temp_width = 0;
		for (int j = 0; j < grid_width; j++) {

			const int image_index = i * grid_width + j;
			if (image_index >= display_num) {
				break;
			}

			max_height = (image_list[group[image_index]].getHeight() > max_height) ? image_list[group[image_index]].getHeight() : max_height;
			temp_width += image_list[group[image_index]].getWidth();
		}

		max_height_each_row[i] = max_height;
		grand_width = (temp_width > grand_width) ? temp_width : grand_width;
	}

	// Compute the max height of the grand image
	for (int i = 0; i < grid_width; i++) {

		int temp_height = 0;
		for (int j = 0; j < grid_height; j++) {

			const int image_index = j * grid_width + i;
			if (image_index >= display_num) {
				break;
			}

			temp_height += image_list[group[image_index]].getHeight();
		}

		grand_height = (temp_height > grand_height) ? temp_height : grand_height;
	}

	// Post-process maximum height each row
	for (int i = 1; i < grid_height; i++) { 
		max_height_each_row[i] += max_height_each_row[i - 1];
	}

	// Create the grand image to be displayed
	cv::Mat grand(grand_height, grand_width, CV_8UC3, cv::Scalar(0));

	// Copy image into the grand image
	int anchor_x = 0;
	int anchor_y = 0;

	for (int i = 0; i < grid_height; i++) {
		for (int j = 0; j < grid_width; j++) {

			const int image_index = i * grid_width + j;
			if (image_index >= display_num) {
				break;
			}

			const cv::Mat& img = image_list[group[image_index]].getImage();
			img.copyTo(grand(cv::Rect(anchor_x, anchor_y, img.cols, img.rows)));

			// Update the anchor_x
			anchor_x += img.cols;
		}

		// Update the anchors
		anchor_x = 0;
		anchor_y = max_height_each_row[i];
	}

	// Resize the grand for display purpose
	int resized_grand_width;
	int resized_grand_height;
	float width_over_height = grand_width * 1.0f / grand_height;

	if (display_num == 2) {
		if (grand_width >= grand_height) {
			resized_grand_width = 640;
			resized_grand_height = resized_grand_width / width_over_height;
		}
		else {
			resized_grand_height = 640;
			resized_grand_width = resized_grand_height * width_over_height;
		}
	}
	else {
		if (grand_width >= grand_height) {
			resized_grand_width = 960;
			resized_grand_height = resized_grand_width / width_over_height;
		}
		else {
			resized_grand_height = 960;
			resized_grand_width = resized_grand_height * width_over_height;
		}
	}
	cv::resize(grand, grand, cv::Size(resized_grand_width, resized_grand_height));

	// Display image
	std::string display_name = "Group: ";
	for (int i = 0; i < display_num - 1; i++) {
		display_name += std::to_string(group[i]) + " - ";
	}
	display_name += std::to_string(group[display_num - 1]);
	cv::imshow(display_name.c_str(), grand);
	cv::waitKey();
}

bool Image_control::return_max_width_height(int& maxWidth_, int& maxHeight_) 
{
	if (image_num <= 0) {
		std::cout << "The images are not yet loaded ..." << std::endl;
		maxWidth_ = -1;
		maxHeight_ = -1;

		return false;
	}

	std::vector<int> widths(image_num);
	std::vector<int> heights(image_num);

	for (int i = 0; i < image_num; i++) {
		widths[i]  = getImage(i).cols;
		heights[i] = getImage(i).rows;
	}

	maxWidth_  = *std::max_element(widths.data(), widths.data() + image_num);
	maxHeight_ = *std::max_element(heights.data(), heights.data() + image_num);

	return true;
}

bool Image_control::return_min_width_height(int& minWidth_, int& minHeight_)
{
	if (image_num <= 0) {
		std::cout << "The images are not yet loaded ..." << std::endl;
		minWidth_ = -1;
		minHeight_ = -1;

		return false;
	}

	std::vector<int> widths(image_num);
	std::vector<int> heights(image_num);

	for (int i = 0; i < image_num; i++) {
		widths[i]  = getImage(i).cols;
		heights[i] = getImage(i).rows;
	}

	minWidth_ = *std::min_element(widths.data(), widths.data() + image_num);
	minHeight_ = *std::min_element(heights.data(), heights.data() + image_num);

	return true;
}

cv::Mat Image_control::getImage(int index_)
{
	return getImageInfo(index_).getImage();
}

Image_info Image_control::getImageInfo(int index_)
{
	return image_list[index_];
}

int Image_control::getImageNum()
{
	return image_num;
}