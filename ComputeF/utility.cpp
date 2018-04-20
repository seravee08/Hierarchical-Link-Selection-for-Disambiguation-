#include <iostream>
#include <string>
#include <cassert>
#include <stdlib.h>
#include <algorithm>
#include <list>
#include <numeric>
#include <random>
#include <functional>
#include <array>

// ===== Supress cmd output =====
#include <windows.h>
#include <ShellAPI.h>

// ===== Boost Library =====
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include "SiftIO.h"
#include "utility.h"
#include "pba\pba.h"
#include "pba\util.h"

template void sort_indices<int>(const std::vector<int>& target_, std::vector<int>& indices_, const bool increasing);
template void sort_indices<float>(const std::vector<float>& target_, std::vector<int>& indices_, const bool increasing);

template void sort_indices<int>(std::vector<int>& target_, const bool increasing);
template void sort_indices<float>(std::vector<float>& target_, const bool increasing);

template <typename T>
void sort_indices(const std::vector<T>& target_, std::vector<int>& indices_, const bool increasing)
{
	// Validate the inputs
	assert(target_.size() == indices_.size());

	std::vector<int> normal_indices(target_.size());
	std::iota(normal_indices.begin(), normal_indices.end(), 0);

	// Sort indices based on comparing values in target
	if (increasing) {
		std::sort(normal_indices.begin(), normal_indices.end(),
			[&target_](int i1, int i2) {return target_[i1] < target_[i2]; });
	}
	else {
		std::sort(normal_indices.begin(), normal_indices.end(),
			[&target_](int i1, int i2) {return target_[i1] > target_[i2]; });
	}
	
	// Re-arrange the indices passed in
	std::vector<int> indices_rearranged(target_.size());
	for (int i = 0; i < target_.size(); i++) {
		indices_rearranged[i] = indices_[normal_indices[i]];
	}
	indices_ = indices_rearranged;
}

template <typename T>
void sort_indices(std::vector<T>& target_, const bool increasing)
{
	if (increasing) {
		std::sort(target_.begin(), target_.end());
	}
	else {
		std::sort(target_.begin(), target_.end(), std::greater<T>());
	}
}

// ======================================================================= //
// =============== The below parts are added for BMVC May 2nd ============ //
// ======================================================================= //


std::string create_list(const std::string direc) {

	std::string list_path = direc;

	// Windows specific: replace back slash with forward slash
	if (list_path.find('\\') != std::string::npos) {
		std::replace(list_path.begin(), list_path.end(), '\\', '/');
	}

	std::string direc_rectified = list_path;

	if (list_path[list_path.length() - 1] == '/') {
		list_path = list_path + "list.txt";
	}
	else {
		list_path = list_path + "/list.txt";
	}

	// If list does not exist, create one
	if (!boost::filesystem::exists(list_path.c_str())) {

		if (boost::filesystem::is_directory(direc_rectified)) {

			std::ofstream out(list_path.c_str());
			boost::filesystem::directory_iterator itr_end;
			for (boost::filesystem::directory_iterator itr(direc_rectified); itr != itr_end; ++itr) {
				if (boost::filesystem::is_regular_file(itr->path())) {

					std::string current_file = itr->path().string();
					if (current_file.find('\\') != std::string::npos) {
						std::replace(current_file.begin(), current_file.end(), '\\', '/');
					}
					
					// Intake only images
					if (current_file.substr(current_file.find(".") + 1) == "jpg" ||
						current_file.substr(current_file.find(".") + 1) == "JPG" ||
						current_file.substr(current_file.find(".") + 1) == "png") {
						out << current_file << std::endl;
					}
				}
			}
			out.close();
		}
		else {
			std::cout << "Incorrect input directory ..." << std::endl;
		}
	}

	return list_path;
}

int system_no_output(std::string command)
{
	command.insert(0, "/C ");

	SHELLEXECUTEINFOA ShExecInfo = { 0 };
	ShExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
	ShExecInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
	ShExecInfo.hwnd = NULL;
	ShExecInfo.lpVerb = NULL;
	ShExecInfo.lpFile = "cmd.exe";
	ShExecInfo.lpParameters = command.c_str();
	ShExecInfo.lpDirectory = NULL;
	ShExecInfo.nShow = SW_HIDE;
	ShExecInfo.hInstApp = NULL;

	if (ShellExecuteExA(&ShExecInfo) == FALSE)
		return -1;

	WaitForSingleObject(ShExecInfo.hProcess, INFINITE);

	DWORD rv;
	GetExitCodeProcess(ShExecInfo.hProcess, &rv);
	CloseHandle(ShExecInfo.hProcess);

	return rv;
}

// Definations for Viewer Class
Viewer::Viewer()
{
	// Set the initial viewing point for the viewer
	cv::Matx<float, 4, 4> initial_pose(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, -20,
		0, 0, 0, 1);
	cv::Affine3d initial_pose_aff(initial_pose);

	// Initialize the viewer
	viz = cv::viz::Viz3d("Model Viewer");
	viz.setBackgroundColor(cv::viz::Color::black());
	viz.showWidget("World Frame", cv::viz::WCoordinateSystem());
	viz.setWindowPosition(cv::Point(150, 150));
	viz.showWidget("text2d", cv::viz::WText("Model Viewer", cv::Point(20, 20), 20, cv::viz::Color::green()));
	viz.setViewerPose(initial_pose_aff);
}

Viewer::~Viewer()
{
	viz.close();
}

// The viewer displays the point clouds and the cameras
// in same color or different colors if the last three
// parameters are passed in correctly

void Viewer::show_pointCloud(
	const std::vector<Point3D>& pt3d_,
	std::vector<CameraT>& cams_,
	const std::vector<int>& cam_index_,
	const std::vector<int>& setA_,
	const std::vector<int>& setB_
	) {

	if (pt3d_.size() <= 0) {
		cout << "Error input point cloud ..." << endl;
		exit(1);
	}

	// Check if the cameras are to be displayed in different colors
	const int cam_num			= cams_.size();
	const int cam_index_size	= cam_index_.size();
	const int setA_size			= setA_.size();
	const int setB_size			= setB_.size();
	if (setB_size > 0 && (cam_index_size != cam_num)) {
		std::cout << "show_pointCloud: cams_ should have the same size as cam_index_ ..." << std::endl;
		exit(1);
	}
	std::vector<cv::viz::Color> colors(cam_num, cv::viz::Color::white());
	for (int i = 0; i < cam_index_size; i++) {
		colors[i] = (std::find(setB_.begin(), setB_.end(), cam_index_[i]) != setB_.end()) ? cv::viz::Color::red() : cv::viz::Color::white();
	}

	// Transform from Point3D to Vec3f format
	const int pt3d_num = pt3d_.size();
	std::vector<cv::Vec3f> pt3d(pt3d_num);
	for (int i = 0; i < pt3d_num; i++) {
		pt3d[i] = cv::Vec3f(pt3d_[i].xyz[0], pt3d_[i].xyz[1], pt3d_[i].xyz[2]);
	}

	// Process camera positions
	std::vector<cv::Affine3d> cam_pos(cam_num);
	std::vector<cv::Matx33d> K_sequence(cam_num);
	for (int i = 0; i < cam_num; i++) {

		float T[3];
		float R[9];
		cams_[i].GetCameraCenter(T);
		cams_[i].GetMatrixRotation(R);
		cv::Mat T_mat(3, 1, CV_32FC1, T);	
		cv::Mat R_mat(3, 3, CV_32FC1, R);

		float focal = cams_[i].GetFocalLength();
		K_sequence[i] = cv::Matx33d(focal, 0.0, focal / 2.0, 0.0, focal, focal / 2.0, 0.0, 0.0, 1.0);
		cv::Mat pc = (cv::Mat_<float>(3, 1) << 0.0, 0.0, focal);
		cv::Mat fp = R_mat.t() * (pc - T_mat);

		cam_pos[i] = cv::viz::makeCameraPose(
			cv::Vec3d(T[0], T[1], T[2]),
			cv::Vec3d(fp.at<float>(0, 0), fp.at<float>(1, 0), fp.at<float>(2, 0)),
			cv::Vec3d(R_mat.at<float>(0, 1), R_mat.at<float>(1, 1), R_mat.at<float>(2, 1)));
	}
	
	viz.showWidget("Model", cv::viz::WCloud(pt3d, cv::viz::Color::bluberry()));
	for (int i = 0; i < cam_num; i++) {
		viz.showWidget(std::string("cam") + std::to_string(i), cv::viz::WCameraPosition(K_sequence[i], 1.0, colors[i]), cam_pos[i]);
	}
	viz.spin();
}

void Viewer::readIn_NVM(std::string nvm_path, std::vector<CameraT>& cams, std::vector<int>& cam_index, std::vector<Point3D>& pt3d)
{
	std::vector<int>			ptidx;	// indices for the projected features in absolute value
	std::vector<int>			camidx;	// indices for cameras on which the features are projected
	std::vector<int>			ptc;	// 3D point color

	std::vector<Point2D>		measurements;
	std::vector<std::string>	names;

	std::ifstream nvm_in(nvm_path.c_str());
	if (!nvm_in.is_open()) {
		std::cout << "readIn_NVM: nvm file open failed, check if nvm exists ..." << std::endl;
		return;
	}

	// Clear previous data
	cams.clear();
	cam_index.clear();
	pt3d.clear();

	if (!LoadNVM(nvm_in, cams, pt3d, measurements, ptidx, camidx, names, ptc)) {
		std::cout << "readIn_NVM: nvm file read failed ..." << std::endl;
		return;
	}

	// close input stream
	nvm_in.close();

	// Parse the indices for the retrived cameras
	const int cam_num = cams.size();
	cam_index.resize(cam_num);
	for (int i = 0; i < cam_num; i++) {
		std::string path;
		std::string name;
		Image_info::splitFilename(names[i], path, name);

		int j = 0;
		while (j < name.length() && !(name[j] > '0' && name[j] <= '9')) {
			j++;
		}
		(j == name.length()) ? cam_index[i] = 0 : cam_index[i] = std::atoi(name.substr(j, name.length() - j).c_str());
	}
}

// Resolve the old and new cameras sets
bool Viewer::resolve_cameras(
	const std::vector<CameraT>& cams_,
	const std::vector<int>&		cam_index_,
	const std::vector<int>&		setA_,
	const std::vector<int>&		setB_,
	std::vector<CameraT>&		old_cams_,
	std::vector<CameraT>&		new_cams_
)
{
	const int sizeA = setA_.size();
	const int sizeB = setB_.size();

	if (sizeA + sizeB != cams_.size()) {
		std::cout << "resolve_cameras: new nvm file does not contain all cams from both sets ..." << std::endl;
		return false;
	}

	old_cams_.clear();
	new_cams_.clear();
	old_cams_.resize(sizeA);
	new_cams_.resize(sizeB);

	for (int i = 0; i < sizeA; i++) {
		int ind = std::find(cam_index_.begin(), cam_index_.end(), setA_[i]) - cam_index_.begin();
		old_cams_[i] = cams_[ind];
	}

	for (int i = 0; i < sizeB; i++) {
		int ind = std::find(cam_index_.begin(), cam_index_.end(), setB_[i]) - cam_index_.begin();
		new_cams_[i] = cams_[ind];
	}

	return true;
}

// Definations for FileOperator Class
FileOperator::FileOperator() {}

FileOperator::~FileOperator() {}

bool FileOperator::deleteFile(const std::string t_)
{
	const boost::filesystem::path to_be_deleted(t_.c_str());
	bool status = boost::filesystem::remove(to_be_deleted);
	return status;
}

bool FileOperator::renameFile(const std::string o_name_, const std::string n_name_)
{
	const boost::filesystem::path original_name(o_name_.c_str());
	const boost::filesystem::path new_name(n_name_.c_str());
	boost::filesystem::rename(original_name, new_name);

	return true;
}