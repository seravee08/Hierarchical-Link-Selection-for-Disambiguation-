#include "SiftIO.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <stdlib.h>

#define _SIZE_INT				sizeof(int)
#define _SIZE_FLOAT				sizeof(float)
#define _SIZE_UNSIGNED_CHAR		sizeof(unsigned char)
#define _PI						3.14159

Image_info::Image_info(const std::string img_name_):
	img_name	(img_name_),
	sift_name	(extract_SIFT_name(img_name_)),
	aux_name	(extract_AUX_name(img_name_)),
	aux_exist	(false),
	sift_exist  (false)
{

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

void Image_info::read_Auxililiary()
{
	// ===== Open input file stream =====
	std::ifstream aux_in(aux_name.c_str(), std::ios::in | std::ios::binary);
	assert(aux_in.is_open());

	// ===== Read in affine information =====

	// Read in Basic information
	int tilts;

	aux_in.read((char*)&width,		_SIZE_INT);
	aux_in.read((char*)&height,		_SIZE_INT);
	aux_in.read((char*)&feat_num,	_SIZE_INT);
	aux_in.read((char*)&tilts,		_SIZE_INT);

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

Eigen::Matrix<float, 2, Eigen::Dynamic> Image_info::get_coordinates()
{
	return keypoints_mat.topRows(2);
}