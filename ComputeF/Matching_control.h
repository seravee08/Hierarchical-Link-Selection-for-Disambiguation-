#pragma once

#ifndef MATCHING_CONTROL_H
#define MATCHING_CONTROL_H

#include "Graph_Disamb.h"
#include "Image_control.h"
#include "Matching.h"
#include "utility.h"

// ===== MatLAB Library =====
#include "engine.h"

class Matching_control {
public:
	Matching_control(
		const std::string direc_,
		const std::string image_list_path_
	);

	~Matching_control();

	// Read in sift and affine information
	void readIn_Keypoints();

	// Read in matchings
	void readIn_Matchings();

	// Compute Sift feature using OpenCV functions locally
	void compute_Sift(int index = -1);

	// Write out the Sift features in binary form, default to write for all images
	void writeSift_BINARY(int index_ = -1, bool VSFM_compatible_ = true);

	// Compute matchings
	void compute_Matchings_1v1(int left_index_ = -1, int right_index_ = -1);

	// Compute matchings between one image and a list of images
	void compute_Matchings_1vN(int left_index_, const std::vector<int>& subset = std::vector<int>());

	// Write out matchings
	void writeOut_Matchings(std::vector<Graph_disamb>& graphs_);

	// Write out matchings
	void writeOut_Matchings(Graph_disamb& graph_);

	// Write out matches for a single image pair
	void write_matches_1v1(int l_ = -1, int r_ = -1);

	// Write out matches between one image and a subset of images
	void write_matches_1vN(int l_, const std::vector<int>& subset = std::vector<int>());

	// Write matches between the designated pairs
	std::string write_matches_designated(const std::string file_name_, const std::vector<cv::Point2i>& linkages);

	// Write out graph layout
	void writeOut_Layout(std::vector<Graph_disamb>& graphs_);

	// Delete machings where the coordinates are out of bound
	void delete_bad_matchings();

	// Compute homography between all pairs
	void compute_Homography();

	// Rectify matchings according to homography mask
	void rectify_matching_homoMask();

	// Compute gist distance accross all pairs
	Eigen::MatrixXf compute_gist_dist_all();

	// Return number of images
	int getImage_num();

	// Get image
	cv::Mat get_image(int index_);

	// Return the maximum width and height of the image list
	void return_max_width_height(int& maxWidth_, int& maxHeight_);

	// Return the minimum width and height of the image list
	void return_min_width_height(int& minWidth_, int& minHeight_);

	// Return the matrix of warped difference
	Eigen::MatrixXf getWarped_diff_mat();

	// Return the matrix of matching number between all pairs
	Eigen::MatrixXi getMatching_number_mat();

	// Display keypoints for an image
	void displayKeypoints(int index_);

	// Display the group status of current round
	void showGroupStatus(const vvv_int& split_results_);

	// Display the group status of current round
	void showGroupStatus(const vv_int& groups_);
		
	// Return the number of matches between the image pair
	int getMatch_number(
		const int left_index_,
		const int right_index_
	);

	// Return the warped difference
	float getWarped_diff_value(
		const int	left_index_,
		const int	right_index_
	);

	// Display matchings according to the indices
	void displayMatchings(
		const int	left_index_,
		const int	right_index_,
		const bool	use_outlier_mask_ = false,
		const bool  locally_computed  = false
	);

	// Construct a new score matrix for the grouped nodes
	Eigen::MatrixXf construct_score_grouped_nodes(
		const Eigen::MatrixXf&			origin_score_,
		std::vector<std::vector<int>>&	grouped_nodes_
	);

	// Construct a new score matrix for the grouped nodes, only top N pairs are considered
	Eigen::MatrixXf construct_score_grouped_nodes_topN(
		const bool						minimum_guided_,
		const Eigen::MatrixXf&			origin_score_,
		std::vector<std::vector<int>>&	grouped_nodes_
	);

	// Translate the relative indices into absolute indices
	void translate_indices(
		std::vector<int>	grand_indices_,
		std::vector<int>&	upper_set_,
		std::vector<int>&	lower_set_
	);

	// Construct graph using the provided scores
	std::vector<Graph_disamb> constructGraph(
		const bool minmum_guided,
		const Eigen::MatrixXf& scores
	);

	// Split the graph
	vvv_int split_graph(
		const bool						minimum_guided_,
		const Eigen::MatrixXf&			origin_score_,
		Graph_disamb&					graph_,
		std::vector<std::vector<int>>	grouped_nodes_,
		Graph_disamb*&					upper_graph_,
		Graph_disamb*&					lower_graph_,
		v_int&							upper_set,
		v_int&							lower_set
	);

	// Split the nodes based on the input graph
	vvv_int Matching_control::iterative_split(
		const bool						minimum_guided_,
		const Eigen::MatrixXf&			origin_score_,
		const vv_int&					grouped_nodes_,
		Graph_disamb&					graph_,
		vv_int&							split_indices
	);

	// Split the graph independent from the graph constructed for this round
	std::vector<std::vector<std::vector<int>>> split_graph_independent_of_graph(
		const bool						minimum_guided_,
		const Eigen::MatrixXf&			origin_score_,
		std::vector<std::vector<int>>	grouped_nodes_
	);

	// Iterative split the graph until certain condition is met
	std::vector<std::vector<std::vector<int>>> iterative_split_graph_independent(
		const bool						minimum_guided_,
		const Eigen::MatrixXf&			origin_score_,
		std::vector<std::vector<int>>	grouped_nodes_
	);

	// Analyze the groups and add eges
	std::vector<std::vector<int>> validate_add_edges_with_homography(
		Eigen::MatrixXf&							origin_score_on_fly,
		std::vector<std::vector<std::vector<int>>>& split_results_,
		std::vector<std::vector<float>>&			warped_diff,
		std::vector<std::vector<int>>&				split_indices
	);

	// Analyze the groups and validate edges with geometric cues: BMVC added
	vv_int validate_add_edges_width_SFM(
		const Eigen::MatrixXf&		score_mat_,
		vvv_int&					split_results_,
		Graph_disamb&				global_graph_
	);

	vv_int validate_last_edge_width_SFM(
		const Eigen::MatrixXf&		score_mat_,
		vv_int&						groups_,
		Graph_disamb&				global_graph_
	);

	// Iteratively construct and split the graphs
	void iterative_group_split(
		const bool				minimum_guided_
	);

	// Construct graph using the provided scores and filtered by homography difference scores
	std::vector<Graph_disamb> constructGraph_with_homography_validate(
		const bool				minimum_guided,
		const Eigen::MatrixXf&	scores
	);

	// Construct graphs for grouped nodes with same settings
	std::vector<Graph_disamb> constructGraph_with_grouped_nodes(
		const bool						minimum_guided_,
		const Eigen::MatrixXf&			origin_score_,
		std::vector<std::vector<int>>	grouped_nodes_
	);

	// Construct graphs using the scores and filtered by the size of stitched images
	std::vector<Graph_disamb> constructGraph_with_stitcher(
		const bool				minimum_guided_,
		const Eigen::MatrixXf&	scores_
	);

	// Compute camera projection matrix and triangulate
	void triangulateTwoCameras(
		const int  left_index_,
		const int  right_index_,
		const bool locally_computed_ = false
	);

	// Stitch images
	static cv::Mat stitch_images(
		const cv::Mat left_,
		const cv::Mat right_
	);

	// =========== BMVC: VSFM SfM Functions =============
	void set_vsfm_path(
		const std::string path_
	);

	void triangulate_VSFM(
		const std::vector<int>& setA,
		const std::vector<int>& setB,
		const bool interrupted = false
	);

	bool linkage_selection(
		const std::vector<int>& setA,
		const std::vector<int>& setB,
		const bool interrupted = false
	);

	int find_closestCam(
		int index_,
		int group_id_
	);

	void find_closestCam(
		CameraT&					tar_cam_,
		std::vector<CameraT>&		cams_,
		int&						closest_ind_,
		float&						closest_dis_
	);

	void Matching_control::find_closestCam(
		CameraT&					tar_cam_,
		std::vector<CameraT>&		cams_,
		std::vector<int>&			index_
	);

	float Matching_control::compute_cam_dis(
		CameraT& l_,
		CameraT& r_
	);

	void commit_cams(
		const std::vector<CameraT>& cams_,
		const std::vector<int>& cams_ind_,
		std::vector<Point3D> pt3d_,
		const int group_id_
	);

	bool call_VSFM(
		std::vector<cv::Point2i>&	linkages_,
		const std::string&			match_name_,
		const std::string&			nvm_path_,
		bool						save_matches_file	= false,
		bool						resume				= false,
		const std::string&			original_file		= std::string(""),
		const std::string&			tmp_nvm_path_		= std::string("")
	);

	float convhull_volume(
		std::vector<CameraT>& cams_
	);

	float convhull_volume(
		std::vector<Point3D>& pt3d_
	);

	void readIn_NVM(
		std::string	nvm_path
	);

	void delete_MAT(
		int index_
	);

	void delete_MAT(
		const std::vector<int>& index_
	);

	void dummy_control();

private:
	int									image_num;
	std::string							vsfm_exec;
	std::string							direc;

	std::vector<CameraT>				cams;
	std::vector<std::vector<Point3D>>	pt3d;
	std::vector<int>					cams_group_id;

	Image_control						img_ctrl;
	Matching							match;
	Viewer								viewer;
	Utility								utility;

	Engine								*ep;
};


#endif // !MATCHING_CONTROL_H