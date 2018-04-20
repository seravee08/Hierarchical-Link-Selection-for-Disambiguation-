#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <opencv2/viz.hpp>

#include "pba\DataInterface.h"

#define SELF_DEFINE_SWAP(a,b) {int temp; temp=a; a=b; b=temp;}

template <typename T>
void sort_indices(const std::vector<T>& target_, std::vector<int>& indices_, const bool increasing);

template <typename T>
void sort_indices(std::vector<T>& target_, const bool increasing);

// ===================================
// New functions added for BMVC May 2nd
// ===================================

std::string create_list(
	const std::string direc
);

int system_no_output(
	std::string command
);

class Viewer {
public:
	Viewer();

	~Viewer();

	// Read in NVM file containing models
	void readIn_NVM(
		std::string					nvm_path,
		std::vector<CameraT>&		cams,
		std::vector<int>&			cam_index,
		std::vector<Point3D>&		pt3d
	);

	// Display the point cloud
	void show_pointCloud(
		const std::vector<Point3D>&		pt3d_,
		std::vector<CameraT>&			cams_,
		const std::vector<int>&			cam_index_	= std::vector<int>(0),
		const std::vector<int>&			setA_		= std::vector<int>(0),
		const std::vector<int>&			setB_		= std::vector<int>(0)
	);

	// Resolve the old and new cameras sets
	bool resolve_cameras(
		const std::vector<CameraT>& cams_,
		const std::vector<int>&		cam_index_,
		const std::vector<int>&		setA_,
		const std::vector<int>&		setB_,
		std::vector<CameraT>&		old_cams_,
		std::vector<CameraT>&		new_cams_
	);

private:
	cv::viz::Viz3d viz;
};

class FileOperator {
public:
	FileOperator();

	~FileOperator();

	static bool deleteFile(const std::string t_);

	static bool renameFile(const std::string o_name_, const std::string n_name_);
};

#endif // !UTILITY_H