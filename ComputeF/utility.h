#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <opencv2/viz.hpp>

#include "Eigen/Core"

#include "pba\DataInterface.h"
#include "Parameters.h"
#include "Graph_Disamb.h"

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
		const std::vector<int>&			setB_		= std::vector<int>(0),
		const bool						showPt3D	= true
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

	// Resolve the old and new cameras sets
	bool resolve_cameras(
		const int					end_cam_ind_,
		const std::vector<int>&		cam_ind_,
		const std::vector<CameraT>& cams_,
		CameraT&					end_cam,
		std::vector<CameraT>&		non_end_cams_
	);

private:
	cv::viz::Viz3d viz;
};

class FileOperator {
public:
	FileOperator();

	~FileOperator();

	static bool deleteFile(
		const std::string t_
	);

	static bool renameFile(
		const std::string o_name_,
		const std::string n_name_
	);

	static bool writeGroups(
		const std::string		name_,
		const vv_int&			groups_,
		const v_int&			cam_group_,
		Graph_disamb&			graph_
	);

	static bool readGroups(
		const std::string	name_,
		vv_int&				groups_,
		v_int&				cam_group_,
		Graph_disamb&		graph_
	);
};

class Utility {
public:
	Utility() {};
	
	~Utility() {};

	static bool computeMatches_betnGroups(
		const int				group_index_,
		const Eigen::MatrixXf&	score_mat_,
		const vvv_int&			split_res_,
		v_int&					srcNode_vec,
		v_int&					desNode_vec,
		v_int&					machNum_vec
	);

	static v_int mergeGroups(
		const int				group_index_,
		const vvv_int&			split_res_
	);
	
	static Graph_disamb construct_singleGraph(
		const bool				minimum_guided_,
		const Eigen::MatrixXf&	score_mat_,
		v_int&					nodes_inGraph_
	);

	static vv_int initGroups(
		const int				group_num_
	);

	static void outputCam_group(
		const std::vector<int>& cams_group_id_
	);

	static void retrieveCamTR(
		CameraT&			t_,
		cv::Mat&			T,
		cv::Mat&			R
	);

	static void retrieveCamP(
		CameraT&			t_,
		cv::Mat&			P_
	);

	static cv::Mat retrieveR(
		const cv::Mat&		P_
	);

	static cv::Mat retrieveT(
		const cv::Mat&		P_
	);

	static void constructLinks_interGP(
		const int					t_,
		const v_int&				ind_,
		const Eigen::MatrixXi&		layout_,
		const int					search_radiu_,
		std::vector<cv::Point2i>&	linkages_
	);

	static void constructLinks_interGP_extensive(
		const int					t_,
		const v_int&				ind_,
		const Eigen::MatrixXf&		score_mat_,
		const int					search_radiu_,
		v_int&						candi_gp,
		std::vector<cv::Point2i>&	linkages_
	);

	static void constructLinks_betnGP(
		const v_int&				gp1_,
		const v_int&				gp2_,
		const Eigen::MatrixXf&		score_mat_,
		std::vector<cv::Point2i>&	linkages_
	);

	static v_int oscilation_tuple_generator(
		const vv_int&			ind1_,
		const vv_int&			ind2_
	);

	static void mostMatches_betnGP(
		const int				gp1_,
		const int				gp2_,
		const vv_int&			group_,
		const Eigen::MatrixXf&	score_mat_,
		cv::Point3i&			src_des_score_,
		v_int&					srcNode_vec,
		v_int&					desNode_vec,
		v_int&					machNum_vec
	);

	static void generateUniqueImageList(
		const std::vector<cv::Point2i>&		linkages_,
		v_int&								img_ind_
	);

	static Eigen::MatrixXf getRelyScoreMatrix(
		const Eigen::MatrixXf& machScore
	);

	static void bigGroup_bridge(
		const int				bigGP_threshold,
		const vv_int&			ind_,
		const vv_int&			group_,
		const Eigen::MatrixXf&	score_mat_,
		vvv_int&				split_results_,
		vvv_int&				split_disamb_res_
	);
};

#endif // !UTILITY_H