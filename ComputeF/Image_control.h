#pragma once

#ifndef IMAGE_CONTROL_H
#define IMAGE_CONTROL_H

#include "SiftIO.h"

class Image_control {
public:
	Image_control(const std::string image_list_path_);
	Image_control(const std::vector<std::string>& image_names);

	~Image_control();

	// Read in auxiliary information for all images, have to be called once before calling read_Sift()
	void read_Auxiliary();

	// Read in Sift information for all images
	void read_Sift();

	// Compute Sift feature using OpenCV functions locally
	void compute_Sift(int index = -1);

	// Read in auxiliary information for a specific image
	void read_single_Auxiliary(int index_);

	// Read in Sift information for a specific image
	void read_single_Sift(int index_);

	// Display specified image
	void displayImage(int index_);

	// Get image
	cv::Mat getImage(int index_);

	// Get number of images
	int getImageNum();

	// Get image information by index
	Image_info getImageInfo(int index_);

	// Display group
	void display_group(std::vector<int> group);

	// Return the maximum width and height of the image list
	bool return_max_width_height(int& maxWidth_, int& maxHeight_);

	// Return the minimum width and height of the image list
	bool return_min_width_height(int& minWidth_, int& minHeight_);

private:
	int image_num;

	std::string image_list_path;

	std::vector<std::string> img_names;

	std::vector<Image_info> image_list;
};

#endif // !IMAGE_CONTROL_H