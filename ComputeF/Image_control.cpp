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
		img_names.push_back(name);

		// Instantiate for class Image_info
		Image_info img(name);
		image_list.push_back(img);
	}

	// Initialize for the number of images
	image_num = img_names.size();
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

Image_info Image_control::getImageInfo(int index_)
{
	return image_list[index_];
}

int Image_control::getImageNum()
{
	return image_num;
}