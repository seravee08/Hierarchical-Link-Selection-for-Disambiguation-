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
	const std::vector<Point3D>&		pt3d_,
	std::vector<CameraT>&			cams_,
	const std::vector<int>&			cam_index_,
	const std::vector<int>&			setA_,
	const std::vector<int>&			setB_,
	const bool						showPt3D
	) {

	if (cams_.size() <= 0) {
		cout << "show_pointCloud: error input point cloud ..." << endl;
		exit(1);
	}
	if (pt3d_.size() == 0 && showPt3D) {
		cout << "show_pointCloud: requesting to display empty cloud point ..." << std::endl;
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
	
	if (showPt3D) {
		viz.showWidget("Model", cv::viz::WCloud(pt3d, cv::viz::Color::bluberry()));
	}
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

bool Viewer::resolve_cameras(
	const int					end_cam_ind_,
	const std::vector<int>&		cam_ind_,
	const std::vector<CameraT>& cams_,
	CameraT&					end_cam,
	std::vector<CameraT>&		non_end_cams_
)
{
	const int num = cam_ind_.size();
	const int end_index = std::find(cam_ind_.begin(), cam_ind_.end(), end_cam_ind_) - cam_ind_.begin();

	non_end_cams_.clear();
	non_end_cams_.reserve(num - 1);
	
	for (int i = 0; i < num; i++) {
		if (i == end_index) {
			end_cam = cams_[i];
		}
		else {
			non_end_cams_.push_back(cams_[i]);
		}
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

bool FileOperator::writeGroups(
	const std::string		name_,
	const vv_int&			groups_,
	const v_int&			cam_group_,
	Graph_disamb&			graph_
)
{
	std::ofstream out(name_.c_str(), std::ios::out);
	if (!out.is_open()) {
		std::cout << "writeGroups: File open failed ..." << std::endl;
		return false;
	}

	const Eigen::MatrixXi layout	= graph_.getLayout();
	const int num					= layout.rows();
	const int gp_num				= groups_.size();
	
	// Write out graph layout
	out << num << std::endl;
	for (int i = 0; i < num; i++) {
		for (int j = 0; j < num; j++) {
			out << layout(i, j) << " ";
		}
		out << std::endl;
	}

	// Write out camera groups
	for (int i = 0; i < num; i++) {
		out << cam_group_[i] << " ";
	}
	out << std::endl;

	// Write out gropus
	out << gp_num << std::endl;
	for (int i = 0; i < gp_num; i++) {
		const int sub_group_size = groups_[i].size();
		out << sub_group_size << " ";
		for (int j = 0; j < sub_group_size; j++) {
			out << groups_[i][j] << " ";
		}
		out << std::endl;
	}
	out.close();

	return true;
}

bool FileOperator::readGroups(
	const std::string	name_,
	vv_int&				groups_,
	v_int&				cam_group_,
	Graph_disamb&		graph_
)
{
	std::ifstream in(name_.c_str(), std::ios::in);
	if (!in.is_open()) {
		std::cout << "readGroups: File open failed ..." << std::endl;
		return false;
	}

	Eigen::MatrixXi layout;
	int image_num;
	int group_num;
	int subGP_size;

	// Read in the graph layout
	in >> image_num;
	layout = Eigen::MatrixXi::Zero(image_num, image_num);
	for (int i = 0; i < image_num; i++) {
		for (int j = 0; j < image_num; j++) {
			in >> layout(i, j);
		}
	}
	graph_.setNodeNum(image_num);
	graph_.setLayout(layout);

	// Read in the cam groups
	for (int i = 0; i < image_num; i++) {
		in >> cam_group_[i];
	}

	// Read in the gropus
	in >> group_num;
	groups_.clear();
	groups_.resize(group_num);
	for (int i = 0; i < group_num; i++) {
		in >> subGP_size;
		groups_[i].resize(subGP_size);
		for (int j = 0; j < subGP_size; j++) {
			in >> groups_[i][j];
		}
	}

	in.close();
	return true;
}

// Definations for Utility Class
// This class is mainly for new functions to refine
// the iterative grouping functionality
bool Utility::computeMatches_betnGroups(
	const int				group_index_,
	const Eigen::MatrixXf&	score_mat_,
	const vvv_int&			split_res_,
	v_int&					srcNode_vec,
	v_int&					desNode_vec,
	v_int&					machNum_vec
)
{
	// Clear previous data
	srcNode_vec.clear();
	desNode_vec.clear();
	machNum_vec.clear();

	// Check if two gropus exist in the split results
	assert(split_res_[group_index_].size() == 2);
	
	// Calculate accumulated match numbers
	const int fstGroup_size = split_res_[group_index_][0].size();
	const int secGroup_size = split_res_[group_index_][1].size();

	for (int i = 0; i < fstGroup_size; i++) {
		for (int j = 0; j < secGroup_size; j++) {
			const int upper_ind  = split_res_[group_index_][0][i];
			const int lower_ind  = split_res_[group_index_][1][j];
			assert(upper_ind != lower_ind);
			const float mach_num = score_mat_(std::min(upper_ind, lower_ind), std::max(upper_ind, lower_ind));

			if (mach_num > 0) {
				srcNode_vec.push_back(std::min(upper_ind, lower_ind));
				desNode_vec.push_back(std::max(upper_ind, lower_ind));
				machNum_vec.push_back(mach_num);
			}
			else {
				continue;
			}
		}
	}

	// Sort the matching number in decreasing order
	sort_indices(machNum_vec, srcNode_vec, false);
	sort_indices(machNum_vec, desNode_vec, false);

	return true;
}

v_int Utility::mergeGroups(
	const int				group_index_,
	const vvv_int&			split_res_
)
{
	v_int fusion_;
	fusion_.reserve(split_res_[group_index_][0].size() + split_res_[group_index_][1].size());
	fusion_.insert(fusion_.end(), split_res_[group_index_][0].begin(), split_res_[group_index_][0].end());
	fusion_.insert(fusion_.end(), split_res_[group_index_][1].begin(), split_res_[group_index_][1].end());

	return fusion_;
}

vv_int Utility::initGroups(
	const int				group_num_
)
{
	vv_int groups(group_num_);
	for (int i = 0; i < group_num_; i++) {
		groups[i].push_back(i);
	}
	return groups;
}

Graph_disamb Utility::construct_singleGraph(
	const bool				minimum_guided_,
	const Eigen::MatrixXf&	score_mat_,
	v_int&					nodes_inGraph_
)
{
	v_int curr_nodes;
	v_int srcNode_vec;
	v_int desNode_vec;
	v_float score_vec;

	const int num	= nodes_inGraph_.size();
	const int start = std::find(nodes_inGraph_.begin(), nodes_inGraph_.end(), 0) - nodes_inGraph_.begin();
	Graph_disamb graph(num);

	// Mark the start node in the records
	nodes_inGraph_[start] = 1;
	curr_nodes.push_back(start);

	while (curr_nodes.size() < num) {

		// Clear the vectors
		srcNode_vec.clear();
		desNode_vec.clear();
		score_vec.clear();

		for (int i = 0; i < curr_nodes.size(); i++) {
			const int src = curr_nodes[i];
			for (int j = 0; j < num; j++) {
				const float score = score_mat_(std::min(src, j), std::max(src, j));
				if (nodes_inGraph_[j] == 0 && score > 0) {
					score_vec.push_back(score);
					srcNode_vec.push_back(src);
					desNode_vec.push_back(j);
				}
			}
		}

		if (score_vec.size() > 0) {
			sort_indices(score_vec, srcNode_vec, minimum_guided_);
			sort_indices(score_vec, desNode_vec, minimum_guided_);

			// TODO
			int bst_link_ = 0;
			nodes_inGraph_[desNode_vec[bst_link_]] = 1;
			curr_nodes.push_back(desNode_vec[bst_link_]);
			graph.addEdge(srcNode_vec[bst_link_], desNode_vec[bst_link_]);
		}
		else {
			std::cout << "The graphs break ..." << std::endl;
			break;
		}
	}

	return graph;
}

void Utility::outputCam_group(
	const std::vector<int>& cams_group_id_
)
{
	const int num = cams_group_id_.size();
	for (int i = 0; i < num; i++) {
		std::cout << "Cam " << i << " -> " << cams_group_id_[i] << std::endl;
	}
}

void  Utility::retrieveCamTR(
	CameraT&			t_,
	cv::Mat&			T_,
	cv::Mat&			R_
)
{
	float T[3];
	float R[9];

	t_.GetCameraCenter(T);
	t_.GetMatrixRotation(R);
	T_ = cv::Mat(3, 1, CV_32FC1, T);
	R_ = cv::Mat(3, 3, CV_32FC1, R);
}

void Utility::retrieveCamP(
	CameraT&			t_,
	cv::Mat&			P_
)
{
	float T[3];
	float R[9];

	t_.GetTranslation(T);
	t_.GetMatrixRotation(R);
	cv::Mat T_(3, 1, CV_32FC1, T);
	cv::Mat R_(3, 3, CV_32FC1, R);
	cv::Mat pad = (cv::Mat_<float>(1, 4) << 0, 0, 0, 1);

	cv::hconcat(T_, R_, P_);
	cv::vconcat(P_, pad, P_);
}

cv::Mat Utility::retrieveR(
	const cv::Mat&		P_
)
{
	cv::Mat R;
	cv::Rect roi(0, 0, 3, 3);
	P_(roi).copyTo(R);
	return R;
}

cv::Mat Utility::retrieveT(
	const cv::Mat&		P_
)
{
	cv::Mat T;
	cv::Rect roi(0, 3, 3, 1);
	P_(roi).copyTo(T);
	return T;
}

void Utility::constructLinks_interGP(
	const int					t_,
	const v_int&				ind_,
	const Eigen::MatrixXi&		layout_,
	const int					search_radiu_,
	std::vector<cv::Point2i>&	linkages_
)
{
	v_int candi_gp;
	const int ind_size = ind_.size();
	for (int i = 0; i < std::min(ind_size, search_radiu_); i++) {
		candi_gp.push_back(ind_[i]);
	}
	candi_gp.push_back(t_);

	const int candi_size = candi_gp.size();
	for (int i = 0; i < candi_size - 1; i++) {
		for (int j = i + 1; j < candi_size; j++) {
			const int& id1 = candi_gp[i];
			const int& id2 = candi_gp[j];
			if (layout_(std::min(id1, id2), std::max(id1, id2)) == 1) {
				linkages_.push_back(cv::Point2i(std::min(id1, id2), std::max(id1, id2)));
			}
		}
	}
}

void Utility::constructLinks_interGP_extensive(
	const int					t_,
	const v_int&				ind_,
	const Eigen::MatrixXf&		score_mat_,
	const int					search_radiu_,
	v_int&						candi_gp,
	std::vector<cv::Point2i>&	linkages_
)
{
	candi_gp.clear();
	const int ind_size = ind_.size();
	for (int i = 0; i < std::min(ind_size, search_radiu_); i++) {
		candi_gp.push_back(ind_[i]);
	}
	candi_gp.push_back(t_);

	const int candi_size = candi_gp.size();
	for (int i = 0; i < candi_size - 1; i++) {
		for (int j = i + 1; j < candi_size; j++) {
			const int& id1 = candi_gp[i];
			const int& id2 = candi_gp[j];
			if (score_mat_(std::min(id1, id2), std::max(id1, id2)) > 0) {
				linkages_.push_back(cv::Point2i(std::min(id1, id2), std::max(id1, id2)));
			}
		}
	}
}

void Utility::constructLinks_betnGP(
	const v_int&				gp1_,
	const v_int&				gp2_,
	const Eigen::MatrixXf&		score_mat_,
	std::vector<cv::Point2i>&	linkages_
)
{
	const int num1 = gp1_.size();
	const int num2 = gp2_.size();
	for (int i = 0; i < num1; i++) {
		for (int j = 0; j < num2; j++) {
			if (score_mat_(std::min(gp1_[i], gp2_[j]), std::max(gp1_[i], gp2_[j])) > 0) {
				linkages_.push_back(cv::Point2i(std::min(gp1_[i], gp2_[j]), std::max(gp1_[i], gp2_[j])));
			}
		}
	}
}

v_int Utility::oscilation_tuple_generator(
	const vv_int&			ind1_,
	const vv_int&			ind2_
)
{
	v_int anomoly;
	const int num1 = ind1_.size();
	const int num2 = ind2_.size();

	for (int i = 0; i < num1; i++) {
		if (ind1_[i].size() == 2) {
			const int pos1 = ind1_[i][0];
			const int pos2 = ind1_[i][1];
			for (int j = 0; j < num2; j++) {
				if (ind2_[j].size() != 2) {
					continue;
				}
				if (pos1 == ind2_[j][0] || pos1 == ind2_[j][1]) {
					if (pos2 == ind2_[j][0] || pos2 == ind2_[j][1]) {
						continue;
					}
					else {
						anomoly.push_back(pos1);
						anomoly.push_back(pos2);
						if (pos1 == ind2_[j][0]) {
							anomoly.push_back(ind2_[j][1]);
						}
						else if (pos1 == ind2_[j][1]) {
							anomoly.push_back(ind2_[j][0]);
						}
						break;
					}
				}
				else if (pos2 == ind2_[j][0] || pos2 == ind2_[j][1]) {
					if (pos1 == ind2_[j][0] || pos1 == ind2_[j][1]) {
						continue;
					}
					else {
						anomoly.push_back(pos2);
						anomoly.push_back(pos1);
						if (pos2 == ind2_[j][0]) {
							anomoly.push_back(ind2_[j][1]);
						}
						else if (pos2 == ind2_[j][1]) {
							anomoly.push_back(ind2_[j][0]);
						}
						break;
					}
				}
			}
			if (anomoly.size() == 3) {
				break;
			}
		}
	}

	return anomoly;
}

void Utility::mostMatches_betnGP(
	const int				gp1_,
	const int				gp2_,
	const vv_int&			group_,
	const Eigen::MatrixXf&	score_mat_,
	cv::Point3i&			src_des_score_,
	v_int&					srcNode_vec,
	v_int&					desNode_vec,
	v_int&					machNum_vec
)
{
	// Clear previous data
	srcNode_vec.clear();
	desNode_vec.clear();
	machNum_vec.clear();

	vvv_int split_expaned(1);
	split_expaned[0].resize(2);
	split_expaned[0][0] = group_[gp1_];
	split_expaned[0][1] = group_[gp2_];

	computeMatches_betnGroups(0, score_mat_, split_expaned, srcNode_vec, desNode_vec, machNum_vec);
	if (machNum_vec.size() > 0) {
		std::sort(machNum_vec.begin(), machNum_vec.end(), std::greater<int>());
		src_des_score_ = cv::Point3i(srcNode_vec[0], desNode_vec[0], machNum_vec[0]);
	}
	else {
		src_des_score_ = cv::Point3i(-1, -1, -1);
	}
}

void Utility::generateUniqueImageList(
	const std::vector<cv::Point2i>&		linkages_,
	v_int&								img_ind_
)
{
	img_ind_.clear();
	const int link_nums = linkages_.size();

	for (int i = 0; i < link_nums; i++) {
		const int l = linkages_[i].x;
		const int r = linkages_[i].y;
		if (std::find(img_ind_.begin(), img_ind_.end(), l) == img_ind_.end()) {
			img_ind_.push_back(l);
		}
		if (std::find(img_ind_.begin(), img_ind_.end(), r) == img_ind_.end()) {
			img_ind_.push_back(r);
		}
	}
	std::sort(img_ind_.begin(), img_ind_.end());
}

Eigen::MatrixXf Utility::getRelyScoreMatrix(
	const Eigen::MatrixXf& machScore
)
{
	const int imageNum = machScore.rows();
	Eigen::MatrixXf relyScore = Eigen::MatrixXf::Zero(imageNum, imageNum);
	v_float match_rec;
	v_float normalize(imageNum);

	for (int i = 0; i < imageNum; i++) {
		match_rec.clear();
		for (int j = i + 1; j < imageNum; j++) {
			if (machScore(i, j) > 0) {
				match_rec.push_back(machScore(i, j));
			}
		}
		for (int j = 0; j < i; j++) {
			if (machScore(j, i) > 0) {
				match_rec.push_back(machScore(j, i));
			}
		}
		std::sort(match_rec.begin(), match_rec.end(), std::greater<float>());
		normalize[i] = (match_rec.size() == 0) ? 1 : match_rec[match_rec.size() / 2];
	}

	for (int i = 0; i < imageNum - 1; i++) {
		for (int j = i + 1; j < imageNum; j++) {
			relyScore(i, j) = (normalize[i] >= normalize[j]) ? machScore(i, j) / normalize[i] : machScore(i, j) / normalize[j];
		}
	}
	
	return relyScore;
}

void Utility::bigGroup_bridge(
	const int				bigGP_threshold,
	const vv_int&			ind_,
	const vv_int&			group_,
	const Eigen::MatrixXf&	score_mat_,
	vvv_int&				split_results_,
	vvv_int&				split_disamb_res_
	
)
{
	// Clear previous data
	split_disamb_res_.clear();

	bool bigGP_only = false;
	int big_group_num = 0;
	const int new_gp_num = ind_.size();
	const int ori_gp_num = group_.size();
	v_int bigGP_rec;
	v_int single_rec;
	v_int big_single_rec;
	// Determine if only two big roups are gruped together
	for (int i = 0; i < new_gp_num; i++) {
		if (ind_[i].size() == 2) {
			const int l_ind = ind_[i][0];
			const int r_ind = ind_[i][1];
			if ((group_[l_ind].size() >= bigGP_threshold && group_[r_ind].size() >= bigGP_threshold)
				|| (group_[l_ind].size() >= bigGP_threshold && group_[r_ind].size() >= 0.5 * bigGP_threshold)
				|| (group_[r_ind].size() >= bigGP_threshold && group_[l_ind].size() >= 0.5 * bigGP_threshold)) {
				big_group_num++;
				bigGP_rec.push_back(i);
			}
		}
		else {
			single_rec.push_back(i);
			if (group_[ind_[i][0]].size() >= bigGP_threshold) {
				big_single_rec.push_back(i);
			}
		}
	}
	if (ori_gp_num - big_group_num == new_gp_num) {
		bigGP_only = true;
	}

	// Avoid force merging of big groups
	if (single_rec.size() == big_single_rec.size()) {
		split_disamb_res_ = split_results_;
		return;
	}
	
	// There are groups other than big groups grouping
	if (!bigGP_only) {
		vv_int expand_container;
		for (int i = 0; i < new_gp_num; i++) {
			if (ind_[i].size() == 2) {
				const int l_ind = ind_[i][0];
				const int r_ind = ind_[i][1];
				if (group_[l_ind].size() >= bigGP_threshold && group_[r_ind].size() >= bigGP_threshold
					|| (group_[l_ind].size() >= bigGP_threshold && group_[r_ind].size() >= 0.5 * bigGP_threshold)
					|| (group_[r_ind].size() >= bigGP_threshold && group_[l_ind].size() >= 0.5 * bigGP_threshold)) {
					expand_container.clear();
					expand_container.push_back(group_[l_ind]);
					split_disamb_res_.push_back(expand_container);
					expand_container.clear();
					expand_container.push_back(group_[r_ind]);
					split_disamb_res_.push_back(expand_container);
				}
				else {
					expand_container.clear();
					expand_container.push_back(group_[l_ind]);
					expand_container.push_back(group_[r_ind]);
					split_disamb_res_.push_back(expand_container);
				}
			}
			else {
				expand_container.clear();
				expand_container.push_back(group_[ind_[i][0]]);
				split_disamb_res_.push_back(expand_container);
			}
		}
	}
	// There are only big groups
	else if (single_rec.size() > 0) {
		v_int srcNode_vec;
		v_int desNode_vec;
		v_int machNum_vec;
		v_int combine_score;
		vv_int combine_rec;
		v_int expand_container;
		cv::Point3i src_des_score_;

		for (int i = 0; i < bigGP_rec.size(); i++) {
			const int l_ind = ind_[bigGP_rec[i]][0];
			const int r_ind = ind_[bigGP_rec[i]][1];
			for (int j = 0; j < single_rec.size(); j++) {
				const int s_ind = ind_[single_rec[j]][0];
				expand_container.clear();
				expand_container.push_back(l_ind);
				expand_container.push_back(s_ind);
				combine_rec.push_back(expand_container);
				mostMatches_betnGP(l_ind, s_ind, group_, score_mat_, src_des_score_, srcNode_vec, desNode_vec, machNum_vec);
				combine_score.push_back(src_des_score_.z);

				expand_container.clear();
				expand_container.push_back(r_ind);
				expand_container.push_back(s_ind);
				combine_rec.push_back(expand_container);
				mostMatches_betnGP(r_ind, s_ind, group_, score_mat_, src_des_score_, srcNode_vec, desNode_vec, machNum_vec);
				combine_score.push_back(src_des_score_.z);
			}
		}

		v_int combine_ind(combine_rec.size());
		std::iota(combine_ind.begin(), combine_ind.end(), 0);
		sort_indices<int>(combine_score, combine_ind, false);
		std::sort(combine_score.begin(), combine_score.end(), std::greater<int>());

		int min_gp_size = std::numeric_limits<int>::max();
		int best_link = 0;
		for (int i = 0; i < std::min((int)combine_ind.size(), 5); i++) {
			const int cur_gp_size = group_[combine_rec[combine_ind[i]][1]].size();
			if (cur_gp_size < min_gp_size) {
				min_gp_size = cur_gp_size;
				best_link = i;
			}
		}

		if (group_[combine_rec[combine_ind[best_link]][1]].size() >= bigGP_threshold
			|| combine_score[best_link] <= 0) {
			split_disamb_res_ = split_results_;
			return;
		}

		const int new_l = combine_rec[combine_ind[best_link]][0];
		const int new_r = combine_rec[combine_ind[best_link]][1];
		std::cout << "Big group merging delay: " << new_l << " " << new_r << std::endl;

		vv_int expander;
		expander.push_back(group_[new_l]);
		expander.push_back(group_[new_r]);
		split_disamb_res_.push_back(expander);
		for (int i = 0; i < new_gp_num; i++) {
			if (ind_[i].size() == 1) {
				if (ind_[i][0] != new_r) {
					expander.clear();
					expander.push_back(group_[ind_[i][0]]);
					split_disamb_res_.push_back(expander);
				}
			}
			else if (ind_[i].size() == 2) {
				if (ind_[i][0] != new_l) {
					expander.clear();
					expander.push_back(group_[ind_[i][0]]);
					split_disamb_res_.push_back(expander);
				}
				if (ind_[i][1] != new_l) {
					expander.clear();
					expander.push_back(group_[ind_[i][1]]);
					split_disamb_res_.push_back(expander);
				}
			}
			else {
				std::cout << "bigGroup_bridge: the graph size cannot be larger than 2 ..." << std::endl;
			}
		}
	}
	else {
		split_disamb_res_ = split_results_;
	}
}