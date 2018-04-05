#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <numeric>
#include <stdlib.h>
#include <limits>

#include "Parameters.h"
#include "utility.h"
#include "Matching_control.h"
#include "graph.h"

#include "opencv2/stitching.hpp"

// ===== Boost Library =====
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

typedef Graph<float, float, float> GraphType;

Matching_control::Matching_control(const std::string image_list_path_) :
	vsfm_exec		(""),
	image_list_path	(image_list_path_),
	img_ctrl		(image_list_path_),
	match			(image_list_path_)
{

}

void Matching_control::readIn_Keypoints()
{
	img_ctrl.read_Auxiliary();
	img_ctrl.readASift_BINARY();

	std::cout << "Keypoints readin completes ..." << std::endl;
}

void Matching_control::readIn_Matchings()
{
	match.read_matchings();

	std::cout << "Matchings readin completes ..." << std::endl;
}

void Matching_control::compute_Sift(int index)
{
	img_ctrl.compute_Sift(index);
}

void Matching_control::writeSift_BINARY(int index_, bool VSFM_compatible_)
{
	img_ctrl.writeSift_BINARY(index_, VSFM_compatible_);
}

void Matching_control::compute_Matchings_1v1(int left_index_, int right_index_)
{
	if (left_index_ < 0 || right_index_ < 0) {
		// Get the number of images
		const int image_num = img_ctrl.getImageNum();

		for (int i = 0; i < image_num - 1; i++) {
			Image_info& img_l = img_ctrl.getImageInfo(i);

			for (int j = i + 1; j < image_num; j++) {
				Image_info& img_r = img_ctrl.getImageInfo(j);
				match.compute_Matchings_1v1(img_l, img_r, i, j);
			}
		}
	}
	else {
		if (left_index_ >= img_ctrl.getImageNum() ||
			right_index_ >= img_ctrl.getImageNum() ||
			left_index_ == right_index_) {
			std::cout << "compute_Matchings_1v1: invalid request for matching computation ..." << std::endl;
		}
		if (left_index_ > right_index_) {
			SELF_DEFINE_SWAP(left_index_, right_index_);
		}
		Image_info& img_l = img_ctrl.getImageInfo(left_index_);
		Image_info& img_r = img_ctrl.getImageInfo(right_index_);
		match.compute_Matchings_1v1(img_l, img_r, left_index_, right_index_);
	}
}

void Matching_control::compute_Matchings_1vN(int left_index_, const std::vector<int>& subset)
{
	if (left_index_ < 0 || left_index_ >= img_ctrl.getImageNum()) {
		std::cout << "Invalid request for matching computation, check the index ..." << std::endl;
		return;
	}

	const int img_num = img_ctrl.getImageNum();
	const int req_num = subset.size();

	// Compute matchings between left_index_ and all the other images (N - 1)
	if (req_num == 0) {
		for (int i = 0; i < img_num; i++) {
			if (left_index_ == i) {
				continue;
			}

			const int mach_num = match.get_matching_number(left_index_, i);
			if (mach_num <= 0) { 
				compute_Matchings_1v1(left_index_, i);
			}
		}
	}
	else {
		for (int i = 0; i < req_num; i++) {
			const int mach_num = match.get_matching_number(left_index_, subset[i]);
			if (mach_num <= 0) {
				compute_Matchings_1v1(left_index_, subset[i]);
			}
		}
	}
}


void Matching_control::writeOut_Matchings(std::vector<Graph_disamb>& graphs_)
{
	match.write_matches(graphs_);
}

void Matching_control::write_matches_1v1(int l_, int r_)
{
	if (l_ < 0 || r_ < 0) {
		match.write_matches_all();
	}
	else {
		if (l_ >= img_ctrl.getImageNum() ||
			r_ >= img_ctrl.getImageNum() ||
			l_ == r_) {
			std::cout << "write_matches_1v1: invalid requests ..." << std::endl;
		}

		if (l_ > r_) {
			SELF_DEFINE_SWAP(l_, r_);
		}
		match.write_matches_1v1(l_, r_);
	}
}

void Matching_control::write_matches_1vN(int l_, const std::vector<int>& subset)
{
	match.write_matches_1vN(l_, subset);
}

std::string Matching_control::write_matches_designated(const std::string file_name_, const std::vector<cv::Point2i>& linkages)
{
	return match.write_matches_designated(file_name_, linkages);
}

void Matching_control::writeOut_Layout(std::vector<Graph_disamb>& graphs_)
{
	match.write_layout(graphs_);
}

void Matching_control::rectify_matching_homoMask()
{
	match.rectify_matchings_homoMask();
}

int Matching_control::getImage_num()
{
	return img_ctrl.getImageNum();
}

Eigen::MatrixXf Matching_control::getWarped_diff_mat()
{
	return match.getWarped_diff();
}

Eigen::MatrixXi Matching_control::getMatching_number_mat()
{
	return match.getMatching_number();
}

void Matching_control::displayKeypoints(int index_)
{
	img_ctrl.getImageInfo(index_).display_keypoints();
}

int Matching_control::getMatch_number(
	const int left_index_,
	const int right_index_
)
{
	Image_info& left_image  = img_ctrl.getImageInfo(left_index_);
	Image_info& right_image = img_ctrl.getImageInfo(right_index_);

	return match.get_matching_number(left_image, right_image);
}

cv::Mat Matching_control::get_image(int index_)
{
	return img_ctrl.getImage(index_);
}

void Matching_control::return_max_width_height(int& maxWidth_, int& maxHeight_)
{
	img_ctrl.return_max_width_height(maxWidth_, maxHeight_);
}

void Matching_control::return_min_width_height(int& minWidth_, int& minHeight_)
{
	img_ctrl.return_min_width_height(minWidth_, minHeight_);
}

float Matching_control::getWarped_diff_value(
	const int	left_index_,
	const int	right_index_
)
{
	assert(left_index_ < right_index_);
	return match.getWarped_diff()(left_index_, right_index_);
}

void Matching_control::showGroupStatus(const std::vector<std::vector<std::vector<int>>>& split_results_)
{
	// Retrieve the number of groups
	const int group_number = split_results_.size();

	// Output the layout of current graph
	std::cout << std::endl;
	for (int i = 0; i < group_number; i++) {

		std::cout << "Group " << i << ": ";
		for (int j = 0; j < split_results_[i].size(); j++) {

			for (int k = 0; k < split_results_[i][j].size(); k++) {

				std::cout << split_results_[i][j][k] << " ";
			}
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}

void Matching_control::displayMatchings(
	const int	left_index_,
	const int	right_index_,
	const bool	use_outlier_mask_,
	const bool  locally_computed
)
{
	// Retrieve the corresponding image strucutres
	Image_info& image_left  = img_ctrl.getImageInfo(left_index_);
	Image_info& image_right = img_ctrl.getImageInfo(right_index_);
	const int matching_number = match.get_matching_number(image_left, image_right);

	if (matching_number > 0) {

		if (use_outlier_mask_ == true) {
			match.display_matchings(image_left, image_right, match.get_outlier_mask(image_left, image_right), locally_computed);
		}
		else {
			match.display_matchings(image_left, image_right, cv::Mat(1, 1, CV_8U), locally_computed);
		}
	}
	else {
		std::cout << "No matches found between the image pair (" << left_index_ << " , " << right_index_ << ")" << std::endl;
	}
}

void Matching_control::delete_bad_matchings()
{
	// Get the number of images
	const int image_num = img_ctrl.getImageNum();

	// Delete bad matches among all pairs
	for (int i = 0; i < image_num - 1; i++) {
		Image_info& img_l = img_ctrl.getImageInfo(i);

		for (int j = i + 1; j < image_num; j++) {
			Image_info& img_r = img_ctrl.getImageInfo(j);

			// Decide if there is any match between the image pair
			const int matching_number = match.get_matching_number(img_l, img_r);
			if (matching_number <= 0) {
				continue;
			}

			match.delete_bad_matchings(img_l, img_r);
		}
	}

	std::cout << "Invalid matchings deletion completes ..." << std::endl;
}

void Matching_control::compute_Homography()
{
	// Get the number of images
	const int image_num = img_ctrl.getImageNum();

	// Declare temporary container
	float	value_diff;
	cv::Mat homography;
	cv::Mat warped_left;
	cv::Mat denseArea_mask;
	
	for (int i = 0; i < image_num - 1; i++) {
		Image_info& img_l = img_ctrl.getImageInfo(i);

		for (int j = i + 1; j < image_num; j++) {
			Image_info& img_r = img_ctrl.getImageInfo(j);

			// Decide if there is any match between the image pair
			const int matching_number = match.get_matching_number(img_l, img_r);
			if (matching_number <= 0) {
				continue;
			}

			denseArea_mask = match.get_denseArea_mask(img_l, img_r);
			match.compute_homography(img_l, img_r, homography, warped_left);

			value_diff = img_r.compute_difference(warped_left, denseArea_mask, true);
			// value_diff = img_r.compute_difference_gist_color(warped_left, denseArea_mask);

			// Set warped difference matrix
			match.setWarped_diff(i, j, value_diff);
		}
	}
}

std::vector<Graph_disamb> Matching_control::constructGraph(
	const bool minimum_guided,
	const Eigen::MatrixXf& scores
	)
{
	std::vector<Graph_disamb> graphs;				// Declare a vector to hold the graph(s)
	graphs.reserve(1);								// Reserve space for at least one graph

	const int image_num = img_ctrl.getImageNum();	// Get the number of images in the database
	std::vector<int> nodes_inGraph(image_num, 0);	// Use a vector to hold the status of each node

	int numberNodes_inGraph = std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0);

	// Double while loop is used to construct multiple graphs in case the database is inherently not continuous
	while (numberNodes_inGraph < image_num) {

		// Find the first node that is not in some graph
		int start_node = std::find(nodes_inGraph.begin(), nodes_inGraph.end(), 0) - nodes_inGraph.begin();

		// Initialize a graph for the current round
		Graph_disamb graph(image_num);
		std::vector<int> current_nodes;

		// Add the start node into the graph
		nodes_inGraph[start_node] = 1;
		current_nodes.push_back(start_node);

		// ===== Construct current graph =====
		while (current_nodes.size() < image_num) {

			int		best_srcNode = -1;
			int		best_dstNode = -1;
			float	best_score	 = (minimum_guided) ? 1000000 : -1;

			// Searching for the best link
			for (int i = 0; i < current_nodes.size(); i++) {

				const int src_node_index = current_nodes[i];
				for (int j = 0; j < image_num; j++) {

					// All information is default to be in the upper triangle
					if (src_node_index == j) {
						continue;
					}
					else {
						const float score = (src_node_index < j) ? scores(src_node_index, j) : scores(j, src_node_index);
						if (score > -1 && nodes_inGraph[j] == 0) {
							
							if (minimum_guided && score < best_score) {
								best_score		= score;
								best_srcNode	= src_node_index;
								best_dstNode	= j;
							}
							else if (!minimum_guided && score > best_score) {
								best_score		= score;
								best_srcNode	= src_node_index;
								best_dstNode	= j;
							}
						}
					}
				}
			}
			// End of searching best link
			if (best_srcNode != -1 && best_dstNode != -1) {

				// Update the status of the new node
				nodes_inGraph[best_dstNode] = 1;
				current_nodes.push_back(best_dstNode);

				// Add the link to the current graph
				graph.addEdge(best_srcNode, best_dstNode);
			}
			else {
				// No link can be found
				break;
			}
		}

		// Push the current graph into the storage
		graphs.push_back(graph);

		// Update the status of all nodes
		numberNodes_inGraph = std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0);
	}

	return graphs;
}

std::vector<Graph_disamb> Matching_control::constructGraph_with_homography_validate(
	const bool minimum_guided,
	const Eigen::MatrixXf& scores
)
{
	std::vector<Graph_disamb> graphs;				// Declare a vector to hold the graph(s)
	graphs.reserve(1);								// Reserve space for at least one graph

	const int image_num = img_ctrl.getImageNum();	// Get the number of images in the database
	std::vector<int> nodes_inGraph(image_num, 0);	// Use a vector to hold the status of each node

	int numberNodes_inGraph = std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0);

	// Doulbe while loop is used to construct multiple graphs in case the database is inherently not continuous
	while (numberNodes_inGraph < image_num) {

		// Find the first node that is not in the graph
		int start_node = std::find(nodes_inGraph.begin(), nodes_inGraph.end(), 0) - nodes_inGraph.begin();

		// Initialize a graph for the current round
		Graph_disamb graph(image_num);
		std::vector<int> current_nodes;

		// Add the start node into the graph
		nodes_inGraph[start_node] = 1;
		current_nodes.push_back(start_node);

		std::vector<int> srcNode_vec;
		std::vector<int> desNode_vec;
		std::vector<float> score_vec;

		// ===== Construct current graph =====
		while (current_nodes.size() < image_num) {

			// Clear the vectors
			srcNode_vec.clear();
			desNode_vec.clear();
			score_vec.clear();

			// Analyze and store links into the structure
			for (int i = 0; i < current_nodes.size(); i++) {

				const int src_node_index = current_nodes[i];
				for (int j = 0; j < image_num; j++) {

					// All information is default to be in the upper triangle
					if (src_node_index == j) {
						continue;
					}
					else {
						const float score = (src_node_index < j) ? scores(src_node_index, j) : scores(j, src_node_index);
						if (score > 0 && nodes_inGraph[j] == 0) {

							score_vec.push_back(score);
							srcNode_vec.push_back(src_node_index);
							desNode_vec.push_back(j);
						}
					}
				}
			}

			// End of links scanning
			if (score_vec.size() > 0) {

				// This part searches the top N link candidates and return the link with least warp difference		
				sort_indices(score_vec, srcNode_vec, minimum_guided);
				sort_indices(score_vec, desNode_vec, minimum_guided);

				// ===== Find link with least warped difference among top N candidates =====
				bool	use_greyScale = true;
				float	value_diff;
				cv::Mat homography;
				cv::Mat warped_left;
				cv::Mat denseArea_mask;

				int		best_link_index		= -1;
				float	best_link_score		= 10000.0f;
				int search_limit = (score_vec.size() > TOP_N_CANDIDATE) ? TOP_N_CANDIDATE : score_vec.size();

				for (int i = 0; i < search_limit; i++) {

					int src_index = srcNode_vec[i];
					int dst_index = desNode_vec[i];

					// Make sure the upper triangle is accessed
					if (src_index > dst_index) {
						int swap_tmp;
						swap_tmp  = src_index;
						src_index = dst_index;
						dst_index = swap_tmp;
					}

					if (getWarped_diff_value(src_index, dst_index) == -1) {

						// If the homography is not computed between the pair
						Image_info& img_l = img_ctrl.getImageInfo(src_index);
						Image_info& img_r = img_ctrl.getImageInfo(dst_index);

						// Read in the auxiliary and Sift information if have not already done so
						img_l.readAuxililiary_BINARY();
						img_l.readASift_BINARY();
						img_r.readAuxililiary_BINARY();
						img_r.readASift_BINARY();

						denseArea_mask = match.get_denseArea_mask(img_l, img_r);
						match.compute_homography(img_l, img_r, homography, warped_left);
						value_diff = img_r.compute_difference(warped_left, denseArea_mask, use_greyScale);

						match.setWarped_diff(src_index, dst_index, value_diff);

						// Clean up the space of these two image objects
						img_l.freeSpace();
						img_r.freeSpace();
					}
					else {
						value_diff = getWarped_diff_value(src_index, dst_index);
					}
	
					if (value_diff < best_link_score && value_diff > 0) {
						best_link_score = value_diff;
						best_link_index = i;
					}
				}

				// Handle the case when all the candidates at current stage are very bad, split the graph
				if (best_link_index == -1) {
					break;
				}

				// Update the status of the new node
				nodes_inGraph[desNode_vec[best_link_index]] = 1;
				current_nodes.push_back(desNode_vec[best_link_index]);

				// Add the link to the current graph
				graph.addEdge(srcNode_vec[best_link_index], desNode_vec[best_link_index]);
			}
			else {
				// No link can be found
				break;
			}

			std::cout << std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0) * 1.0f / image_num << "% completes ..." << std::endl;
		}

		// Push the current graph into the storage
		graphs.push_back(graph);

		// Update the status of all nodes
		numberNodes_inGraph = std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0);
	}

	return graphs;
}

std::vector<Graph_disamb> Matching_control::constructGraph_with_homography_validate_grouped_nodes(
	const bool						minimum_guided_,
	const Eigen::MatrixXf&			origin_score_,
	std::vector<std::vector<int>>	grouped_nodes_
)
{
	std::vector<Graph_disamb> graphs;				// Declare a vector to hold the graph(s)
	graphs.reserve(1);								// Reserve space for at least one graph

	const int image_num = grouped_nodes_.size();	// Retrive the number of grouped nodes in the graph
	std::vector<int> nodes_inGraph(image_num, 0);	// Use a vector to hold the status of each node

	// Construct a score matrix for the current graph configuration
	Eigen::MatrixXf grouped_scores = construct_score_grouped_nodes_topN(minimum_guided_, origin_score_, grouped_nodes_);
	assert(grouped_scores.rows() == image_num);

	// Calculate how many nodes are in the current graph
	int numberNodes_inGraph = std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0);

	// Double while loop is used to construct multiple graphs in case the database is inherently not continuous
	while (numberNodes_inGraph < image_num) {

		// Find the first node that is not in the current graph
		int start_node = std::find(nodes_inGraph.begin(), nodes_inGraph.end(), 0) - nodes_inGraph.begin();

		// Initialize a graph for the current round
		Graph_disamb graph(image_num);
		std::vector<int> current_nodes;

		// Add the start node into the graph
		nodes_inGraph[start_node] = 1;
		current_nodes.push_back(start_node);

		std::vector<int> srcNode_vec;
		std::vector<int> desNode_vec;
		std::vector<float> score_vec;

		// ===== Construct current graph =====
		while (current_nodes.size() < image_num) {

			// Clear the vectors
			srcNode_vec.clear();
			desNode_vec.clear();
			score_vec.clear();

			// Analyze and store links into the structure
			for (int i = 0; i < current_nodes.size(); i++) {

				const int src_node_index = current_nodes[i];
				for (int j = 0; j < image_num; j++) {

					// All informatin is defualt to be stored in the upper triangle
					if (src_node_index == j) {
						continue;
					}
					else {
						const float score = (src_node_index < j) ? grouped_scores(src_node_index, j) : grouped_scores(j, src_node_index);
						if (score > 0 && nodes_inGraph[j] == 0) {

							score_vec.push_back(score);
							srcNode_vec.push_back(src_node_index);
							desNode_vec.push_back(j);
						}
					}
				}
			}
			// End of links scanning

			if (score_vec.size() > 0) {

				// This part searches for the top N link candidates and return the best link
				// TODO
				sort_indices(score_vec, srcNode_vec, minimum_guided_);
				sort_indices(score_vec, desNode_vec, minimum_guided_);

				// TODO
				int best_link_index = 0;

				// Update the status of the new node
				nodes_inGraph[desNode_vec[best_link_index]] = 1;
				current_nodes.push_back(desNode_vec[best_link_index]);

				// Add the link to the current graph
				graph.addEdge(srcNode_vec[best_link_index], desNode_vec[best_link_index]);
			}
			else {
				// No link can be found
				break;
			}

			std::cout << std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0) * 1.0f / image_num << "% completes ..." << std::endl;
		}

		// Push  the current graph into the storage
		graphs.push_back(graph);

		// Update the status of all nodes
		numberNodes_inGraph = std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0);
	}
	
	return graphs;
}

std::vector<Graph_disamb> Matching_control::constructGraph_with_stitcher(
	const bool minimum_guided,
	const Eigen::MatrixXf& scores
)
{
	std::vector<Graph_disamb> graphs;				// Declare a vector to hold the graph(s)
	graphs.reserve(1);								// Reserve space for at least one graph

	const int image_num = img_ctrl.getImageNum();	// Get the number of images in the database
	std::vector<int> nodes_inGraph(image_num, 0);	// Use a vector to hold the status of each node

	int numberNodes_inGraph = std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0);

	// Doulbe while loop is used to construct multiple graphs in case the database is inherently not continuous
	while (numberNodes_inGraph < image_num) {

		// Find the first node that is not in the graph
		int start_node = std::find(nodes_inGraph.begin(), nodes_inGraph.end(), 0) - nodes_inGraph.begin();

		// Initialize a graph for the current round
		Graph_disamb graph(image_num);
		std::vector<int> current_nodes;

		// Add the start node into the graph
		nodes_inGraph[start_node] = 1;
		current_nodes.push_back(start_node);

		// Initialize the panorama
		cv::Mat pano = img_ctrl.getImage(start_node);

		std::vector<int> srcNode_vec;
		std::vector<int> desNode_vec;
		std::vector<float> score_vec;

		// ===== Construct current graph =====
		while (current_nodes.size() < image_num) {

			// Clear the vectors
			srcNode_vec.clear();
			desNode_vec.clear();
			score_vec.clear();

			// Analyze and store links into the structure
			for (int i = 0; i < current_nodes.size(); i++) {

				const int src_node_index = current_nodes[i];
				for (int j = 0; j < image_num; j++) {

					// All information is default to be in the upper triangle
					if (src_node_index == j) {
						continue;
					}
					else {
						const float score = (src_node_index < j) ? scores(src_node_index, j) : scores(j, src_node_index);
						if (score > 0 && nodes_inGraph[j] == 0) {

							score_vec.push_back(score);
							srcNode_vec.push_back(src_node_index);
							desNode_vec.push_back(j);
						}
					}
				}
			}

			// End of links scanning
			if (score_vec.size() > 0) {

				// This part searches the top N link candidates and return the link with least warp difference		
				sort_indices(score_vec, srcNode_vec, minimum_guided);
				sort_indices(score_vec, desNode_vec, minimum_guided);

				// ===== Find link that will expand current panorama =====

				int	best_link_index = -1;
				int search_limit = (score_vec.size() > TOP_N_CANDIDATE) ? TOP_N_CANDIDATE : score_vec.size();

				for (int i = 0; i < search_limit; i++) {

					int src_index = srcNode_vec[i];
					int dst_index = desNode_vec[i];

					// Make sure the upper triangle is accessed
					if (src_index > dst_index) {
						int swap_tmp;
						swap_tmp = src_index;
						src_index = dst_index;
						dst_index = swap_tmp;
					}

					
					cv::Mat pano_result = stitch_images(pano, img_ctrl.getImage(dst_index));

					std::cout << srcNode_vec[i] << "  " << desNode_vec[i] << " : (" << pano.rows << "," << pano.cols << ") (" << pano_result.rows << "," << pano_result.cols << ")" << std::endl;
					cv::imshow("Test", pano_result);
					cv::waitKey();

					if (pano_result.rows * pano_result.cols > pano.rows * pano.cols) {


						std::cout << " IN "  << std::endl;

						

						pano = pano_result;
						best_link_index = i;
						break;
					}
				}

				// Handle the case when all the candidates at current stage are very bad, split the graph
				if (best_link_index == -1) {
					break;
				}

				// Update the status of the new node
				nodes_inGraph[desNode_vec[best_link_index]] = 1;
				current_nodes.push_back(desNode_vec[best_link_index]);

				// Add the link to the current graph
				graph.addEdge(srcNode_vec[best_link_index], desNode_vec[best_link_index]);
			}
			else {
				// No link can be found
				break;
			}

			// std::cout << std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0) * 1.0f / image_num << "% completes ..." << std::endl;

			//std::cout << std::endl;
			//for (int ii = 0; ii < current_nodes.size(); ii++) {
			//	std::cout << current_nodes[ii] << std::endl;
			//}
			//std::cout << std::endl;
		}

		// std::cout << "===========================================" << std::endl;

		// Push the current graph into the storage
		graphs.push_back(graph);

		// Update the status of all nodes
		numberNodes_inGraph = std::accumulate(nodes_inGraph.begin(), nodes_inGraph.end(), 0);
	}

	return graphs;
}

std::vector<std::vector<std::vector<int>>> Matching_control::split_graph(
	const bool						minimum_guided_,
	const Eigen::MatrixXf&			origin_score_,
	Graph_disamb&					graph_,
	std::vector<std::vector<int>>	grouped_nodes_,
	Graph_disamb*&					upper_graph_,
	Graph_disamb*&					lower_graph_,
	std::vector<int>&				upper_set,
	std::vector<int>&				lower_set
)
{
	// The split groups to be returned
	std::vector<std::vector<std::vector<int>>> split_groups(2);

	// Get number of nodes in the graph
	const int node_num			= grouped_nodes_.size();
	const int inGraph_nodeNum	= graph_.number_nodes_inGraph();

	// Shrink the indices of the original graph to the compact graph
	int cntr = 0;
	std::vector<int> compact_indices(node_num, -1);
	for (int i = 0; i < node_num; i++) {
		if (graph_.get_node_status(i) == 1) {
			compact_indices[i] = cntr++;
		}
	}

	// Retrieve the graph layout
	const Eigen::MatrixXi graph = graph_.getLayout();

	// Construct a score matrix for the current graph configuration
	Eigen::MatrixXf grouped_scores = construct_score_grouped_nodes_topN(minimum_guided_, origin_score_, grouped_nodes_);
	assert(node_num == grouped_scores.rows());

	// Invert the score matrix if it is minimum guided
	if (minimum_guided_) {
		const float max_score = grouped_scores.maxCoeff();

		for (int i = 0; i < node_num - 1; i++) {
			for (int j = i + 1; j < node_num; j++) {

				if (grouped_scores(i, j) > 0) {
					grouped_scores(i, j) = max_score - grouped_scores(i, j);
				}
			}
		}
	}

	// Set initial values
	int		best_source;
	int		best_sink;
	float	best_score = 100000000;

	// ===== Find source and sink node =====
	for (int i = 0; i < node_num - 1; i++) {
		for (int j = i + 1; j < node_num; j++) {

			if (graph_.get_node_status(i) == 1 && graph_.get_node_status(j) == 1 && grouped_scores(i, j) < best_score) {

				best_source = i;
				best_sink = j;
				best_score = grouped_scores(i, j);
			}
		}
	}

	// Empty graph case
	if (best_score == 100000000) {
		return split_groups;
	}
	
	// ===== Create the graphcut structure ======
	GraphType* graph_cut = new GraphType(inGraph_nodeNum, 2 * inGraph_nodeNum + inGraph_nodeNum * (inGraph_nodeNum - 1));
	graph_cut->add_node(inGraph_nodeNum);

	// Add edges to the source and sink
	for (int i = 0; i < node_num; i++) {
		if (graph_.get_node_status(i) == 1) {

			if (i == best_source) {
				graph_cut->add_tweights(compact_indices[i], std::numeric_limits<float>::infinity(), 0);
			}
			else if (i == best_sink) {
				graph_cut->add_tweights(compact_indices[i], 0, std::numeric_limits<float>::infinity());
			}
			else {
				graph_cut->add_tweights(compact_indices[i], 0, 0);
			}
		}
	}
	
	// Add edges among the nodes
	for (int i = 0; i < node_num - 1; i++) {
		for (int j = i + 1; j < node_num; j++) {

			if (graph(i, j) == 1) {
				graph_cut->add_edge(compact_indices[i], compact_indices[j], grouped_scores(i, j), grouped_scores(i, j));
			}
		}
	}

	// Calculate maximum flow
	float max_flow = graph_cut->maxflow();

	// Split the nodes
	for (int i = 0; i < node_num; i++) {
		if (graph_.get_node_status(i) == 1) {
			if (graph_cut->what_segment(compact_indices[i]) == GraphType::SINK) {
				split_groups[0].push_back(grouped_nodes_[i]);
				upper_set.push_back(i);
			}
			else {
				split_groups[1].push_back(grouped_nodes_[i]);
				lower_set.push_back(i);
			}
		}
	}

	// Construct sub-graph for subsequent rounds, all information stored in upper triangle
	const int upper_size = upper_set.size();
	const int lower_size = lower_set.size();
	upper_graph_ = new Graph_disamb(upper_size);
	lower_graph_ = new Graph_disamb(lower_size);

	for (int i = 0; i < upper_size - 1; i++) {
		for (int j = i + 1; j < upper_size; j++) {
			if (graph(upper_set[i], upper_set[j]) == 1) {
				upper_graph_->addEdge(i, j);
			}
		}
	}

	for (int i = 0; i < lower_size - 1; i++) {
		for (int j = i + 1; j < lower_size; j++) {
			if (graph(lower_set[i], lower_set[j]) == 1) {
				lower_graph_->addEdge(i, j);
			}
		}
	}

	// Clean up the graph
	delete graph_cut;

	// Return the results
	return split_groups;
}

// ===== This function has major flaw, cannot use =====
// ====================================================
std::vector<std::vector<std::vector<int>>> Matching_control::split_graph_independent_of_graph(
	const bool						minimum_guided_,
	const Eigen::MatrixXf&			origin_score_,
	std::vector<std::vector<int>>	grouped_nodes_
)
{
	// The split groups to be returned
	std::vector<std::vector<std::vector<int>>> split_groups(2);

	// Get number of nodes in the graph
	const int node_num = grouped_nodes_.size();

	// Construct a score matrix for the current graph configuration
	Eigen::MatrixXf grouped_scores = construct_score_grouped_nodes(origin_score_, grouped_nodes_);
	assert(node_num == grouped_scores.rows());

	// Invert the score matrix if it is minimum guided
	if (minimum_guided_) {
		const float max_score = grouped_scores.maxCoeff();

		for (int i = 0; i < node_num - 1; i++) {
			for (int j = i + 1; j < node_num; j++) {

				if (grouped_scores(i, j) > 0) {
					grouped_scores(i, j) = max_score - grouped_scores(i, j);
				}
			}
		}
	}

	// Set initial values
	int		best_source;
	int		best_sink;
	float	best_score = 100000000;

	// ===== Find source and sink node =====
	for (int i = 0; i < node_num - 1; i++) {
		for (int j = i + 1; j < node_num; j++) {

			if (grouped_scores(i, j) < best_score) {

				best_source = i;
				best_sink = j;
				best_score = grouped_scores(i, j);
			}
		}
	}

	// Empty graph case
	if (best_score == 100000000) {
		return split_groups;
	}

	// ===== Create the graphcut structure ======
	GraphType* graph_cut = new GraphType(node_num, node_num - 1);
	graph_cut->add_node(node_num);

	// Add edges to the source
	for (int i = 0; i < node_num; i++) {

		if (i != best_source && i != best_sink) {

			const float score = (i > best_source) ? grouped_scores(best_source, i) : grouped_scores(i, best_source);
			if (score > 0) {
				graph_cut->add_tweights(i, score, 0);
			}
			else {
				graph_cut->add_tweights(i, 0, 0);
			}
		}
	}
	graph_cut->add_tweights(best_source, 10000, 0);

	// Add edgees to the sink
	for (int i = 0; i < node_num; i++) {

		if (i != best_source && i != best_sink) {

			const float score = (i > best_sink) ? grouped_scores(best_sink, i) : grouped_scores(i, best_sink);
			if (score > 0) {
				graph_cut->add_tweights(i, 0, score);
			}
			else {
				graph_cut->add_tweights(i, 0, 0);
			}
		}
	}
	graph_cut->add_tweights(best_sink, 0, 10000);

	// Add edges between the normal nodes
	for (int i = 0; i < node_num - 1; i++) {
		for (int j = i + 1; j < node_num; j++) {

			const float score = grouped_scores(i, j);

			// std::cout << score << std::endl;

			if (score > 0) {
				graph_cut->add_edge(i, j, score, score);
			}
			else {
				graph_cut->add_edge(i, j, 0, 0);
			}
		}
	}

	// Run the graph cut
	float flow = graph_cut->maxflow();

	// Split the nodes
	for (int i = 0; i < node_num; i++) {
		if (graph_cut->what_segment(i) == GraphType::SINK) {
			split_groups[0].push_back(grouped_nodes_[i]);
		}
		else {
			split_groups[1].push_back(grouped_nodes_[i]);
		}
	}

	// Clean up the graph
	delete graph_cut;

	// Return the results
	return split_groups;
}

std::vector<std::vector<std::vector<int>>> Matching_control::iterative_split_graph_independent(
	const bool						minimum_guided_,
	const Eigen::MatrixXf&			origin_score_,
	std::vector<std::vector<int>>	grouped_nodes_
)
{
	// Get the number of nodes in the graph
	const int node_num = grouped_nodes_.size();

	// The inputs have to have at least three grouped nodes
	assert(node_num > 2);

	// The final split results to be returned
	std::vector<std::vector<std::vector<int>>> split_results;

	// Initialize the to be processed groups
	std::vector<std::vector<std::vector<int>>> to_be_processed(1);
	to_be_processed[0] = grouped_nodes_;

	// ====== Start the iterative split =====
	while (to_be_processed.size() != 0) {

		std::vector<std::vector<std::vector<int>>> results = split_graph_independent_of_graph(minimum_guided_, origin_score_, to_be_processed.back());
		to_be_processed.pop_back();

		if (results[0].size() <= SPLIT_LIMIT) {
			split_results.push_back(results[0]);
		}
		else {
			to_be_processed.push_back(results[0]);
		}

		if (results[1].size() <= SPLIT_LIMIT) {
			split_results.push_back(results[1]);
		}
		else {
			to_be_processed.push_back(results[1]);
		}
	}

	return split_results;
}

std::vector<std::vector<std::vector<int>>> Matching_control::iterative_split(
	const bool						minimum_guided_,
	const Eigen::MatrixXf&			origin_score_,
	std::vector<std::vector<int>>	grouped_nodes_,
	Graph_disamb&					graph_,
	std::vector<std::vector<int>>&  split_indices
)
{
	// Get the number of nodes in the graph
	const int node_num = grouped_nodes_.size();

	// The inputs have to have at least three grouped nodes
	assert(node_num > 2);

	// The final split results to be returned
	std::vector<std::vector<std::vector<int>>> split_results;

	// Initialize a index vector
	std::vector<int> initial_indices(node_num);
	std::iota(initial_indices.begin(), initial_indices.end(), 0);

	// Initialize the to be processed groups
	std::vector<std::vector<std::vector<int>>> to_be_processed(1);
	std::vector<Graph_disamb> graph_vec;
	std::vector<std::vector<int>> indices_vec;

	to_be_processed[0] = grouped_nodes_;
	graph_vec.push_back(graph_);
	indices_vec.push_back(initial_indices);

	// ====== Start the iterative split =====
	while (to_be_processed.size() != 0) {

		Graph_disamb* upper_graph;
		Graph_disamb* lower_graph;

		std::vector<int> upper_set;
		std::vector<int> lower_set;

		std::vector<std::vector<std::vector<int>>> results = split_graph(minimum_guided_, origin_score_,
			graph_vec.back(), to_be_processed.back(), upper_graph, lower_graph, upper_set, lower_set);
		translate_indices(indices_vec.back(), upper_set, lower_set);

		to_be_processed.pop_back();
		graph_vec.pop_back();
		indices_vec.pop_back();

		if (results[0].size() <= SPLIT_LIMIT) {
			split_results.push_back(results[0]);
			split_indices.push_back(upper_set);
		}
		else {
			to_be_processed.push_back(results[0]);
			graph_vec.push_back(*upper_graph);
			indices_vec.push_back(upper_set);
		}

		if (results[1].size() <= SPLIT_LIMIT) {
			split_results.push_back(results[1]);
			split_indices.push_back(lower_set);
		}
		else {
			to_be_processed.push_back(results[1]);
			graph_vec.push_back(*lower_graph);
			indices_vec.push_back(lower_set);
		}
	}

	return split_results;
}

void Matching_control::iterative_group_split(
	const bool				minimum_guided_,
	const Eigen::MatrixXf&	origin_score_
)
{
	// Copy the origin score
	Eigen::MatrixXf origin_score_on_fly = origin_score_;

	// Get the number of images in the database
	const int image_num = img_ctrl.getImageNum();
	assert(image_num == origin_score_.rows());

	// Initialize for the gropus
	std::vector<std::vector<int>> groups(image_num);
	for (int i = 0; i < image_num; i++) {
		groups[i].push_back(i);
	}

	// Initialize the warp difference
	std::vector<std::vector<float>> warped_diff(image_num);
	std::vector<std::vector<int>> split_indices;

	// TODO
	while (groups.size() > 2) {

		split_indices.clear();
		std::vector<Graph_disamb> graphs = constructGraph_with_homography_validate_grouped_nodes(minimum_guided_, origin_score_on_fly, groups);
		std::vector<std::vector<std::vector<int>>> split_results = iterative_split(minimum_guided_, origin_score_on_fly, groups, graphs[0], split_indices);
		showGroupStatus(split_results);
		groups = validate_add_edges(origin_score_on_fly, split_results, warped_diff, split_indices);

		//for (int i = 0; i < groups.size(); i++) {
		//	std::cout << "Group " << i << ": ";

		//	for (int j = 0; j < groups[i].size(); j++) {
		//		std::cout << groups[i][j] << "  ";
		//	}

		//	std::cout << std::endl;
		//}
		//std::cout << std::endl;

		//for (int i = 0; i < warped_diff.size(); i++) {

		//	std::cout << "Record " << i << " : ";
		//	for (int j = 0; j < warped_diff[i].size(); j++) {
		//		std::cout << warped_diff[i][j] << " ";
		//	}
		//	std::cout << std::endl;
		//}

		//system("pause");
	}
}

// ============================================================== //
// This function loop through all the groups in which one or two
// sub-groups exist. In case of two sub-groups, among all pairs,
// warped difference is computed on only top N pairs with most
// number of matches. 
// ============================================================== //

std::vector<std::vector<int>> Matching_control::validate_add_edges(
	Eigen::MatrixXf&							origin_score_on_fly,
	std::vector<std::vector<std::vector<int>>>& split_results_,
	std::vector<std::vector<float>>&			warped_diff_,
	std::vector<std::vector<int>>&				split_indices_
)
{
	std::vector<std::vector<int>> combined_groups;
	std::vector<std::vector<float>> warped_diff;
	const int group_num = split_results_.size();
	assert(group_num == split_indices_.size());

	for (int i = 0; i < group_num; i++) {

		// Only one group in the section
		if (split_results_[i].size() == 1) {

			assert(split_indices_[i].size() == 1);
			combined_groups.push_back(split_results_[i][0]);
			warped_diff.push_back(warped_diff_[split_indices_[i][0]]);
		}
		// More than one group in the section
		else {

			assert(split_results_[i].size() == 2);
			assert(split_indices_[i].size() == 2);
			
			// =====================================================
			// ====== Loop through all pairs among two groups ======
			// =====================================================

			// ====================== Phase 1 ===================== //
			// Find the top N pairs with the most number of matches //

			std::vector<int>	srcNode_vec;
			std::vector<int>	desNode_vec;
			std::vector<int>	match_num_vec;
			std::vector<float>	score_vec;

			for (int j = 0; j < split_results_[i][0].size(); j++) {
				for (int k = 0; k < split_results_[i][1].size(); k++) {

					int upper_index = split_results_[i][0][j];
					int lower_index = split_results_[i][1][k];

					// Swap the upper and lower if necessary
					if (upper_index > lower_index) {
						int swap_tmp = upper_index;
						upper_index = lower_index;
						lower_index = swap_tmp;
					}
					
					// Retrieve the matching number between the pair
					const int match_number  = origin_score_on_fly(upper_index, lower_index);
					if (match_number <= 0) {
						continue;
					}

					// Push back the indices
					match_num_vec.push_back(match_number);
					srcNode_vec.push_back(upper_index);
					desNode_vec.push_back(lower_index);
				}
			}

			// Sort the matching number in decreasing order
			sort_indices(match_num_vec, srcNode_vec, false);
			sort_indices(match_num_vec, desNode_vec, false);

			// ====================== Phase 2 ===================== //
			//  Compute the warped difference for the top N pairs   //

			bool	use_greyScale = true;
			float	value_diff;
			cv::Mat homography;
			cv::Mat warped_left;
			cv::Mat denseArea_mask;
			
			// Decide the search limit
			int search_limit = (match_num_vec.size() >= TOP_N_CANDIDATE) ? TOP_N_CANDIDATE : match_num_vec.size();
			srcNode_vec.erase(srcNode_vec.end() - (srcNode_vec.size() - search_limit), srcNode_vec.end());
			desNode_vec.erase(desNode_vec.end() - (desNode_vec.size() - search_limit), desNode_vec.end());

			// ===== Record the warp difference for this group =====
			std::vector<float> fusion_warp;
			fusion_warp.reserve(warped_diff_[split_indices_[i][0]].size() + warped_diff_[split_indices_[i][1]].size());
			fusion_warp.insert(fusion_warp.end(), warped_diff_[split_indices_[i][0]].begin(), warped_diff_[split_indices_[i][0]].end());
			fusion_warp.insert(fusion_warp.end(), warped_diff_[split_indices_[i][1]].begin(), warped_diff_[split_indices_[i][1]].end());

			// ===== Calculate the warp difference history =====
			sort_indices(fusion_warp, true);
			int history_limit = (fusion_warp.size() >= WARP_HISTORY) ? WARP_HISTORY : fusion_warp.size();
			float warp_history = (history_limit == 0) ? 0 : std::accumulate(fusion_warp.end() - history_limit, fusion_warp.end(), 0.0) / history_limit;

			for (int i = 0; i < search_limit; i++) {

				const int upper_index = srcNode_vec[i];
				const int lower_index = desNode_vec[i];

				// ===== Compute warped difference ===== //
				if (getWarped_diff_value(upper_index, lower_index) == -1) {

					// If the homography is not computed between the pair
					Image_info& img_l = img_ctrl.getImageInfo(upper_index);
					Image_info& img_r = img_ctrl.getImageInfo(lower_index);

					// Read in the auxiliary and Sift information if have not already done so
					img_l.readAuxililiary_BINARY();
					img_l.readASift_BINARY();
					img_r.readAuxililiary_BINARY();
					img_r.readASift_BINARY();

					denseArea_mask = match.get_denseArea_mask(img_l, img_r);
					match.compute_homography(img_l, img_r, homography, warped_left);
					value_diff = img_r.compute_difference(warped_left, denseArea_mask, use_greyScale);

					// Set the warp difference
					match.setWarped_diff(upper_index, lower_index, value_diff);

					// Clean up the space of the two image objects
					img_l.freeSpace();
					img_r.freeSpace();
				}
				else {

					// The warped difference is computed and stored
					value_diff = getWarped_diff_value(upper_index, lower_index);
				}

				// Push the score value into the vector for post-processing
				if (value_diff > 0) {
					score_vec.push_back(abs(value_diff - warp_history));
				}
				else {
					score_vec.push_back(-1);
				}
			}

			// Sort the matching number in decreasing order
			sort_indices(score_vec, srcNode_vec, true);
			sort_indices(score_vec, desNode_vec, true);
			sort_indices(score_vec, true);

			// ====================== Phase 3 ===================== //
			//              Validate the potential edge             //

			// TODO: Find best link
			int best_link_index = 0;
			const int best_src		= srcNode_vec[best_link_index];
			const int best_des		= desNode_vec[best_link_index];
			const float best_score	= getWarped_diff_value(best_src, best_des);

			// ===== Validate the potential link ===== //
			bool accept_link = false;
			if (best_score > 0) {
				if (best_score > warp_history && (best_score - warp_history) / warp_history <= WARP_UPPER_THRESHOLD) {
					accept_link = true;
				}
				else if (warp_history > best_score && (warp_history - best_score) / warp_history <= WARP_LOWER_THRESHOLD) {
					accept_link = true;
				}
			}

			// ===== Update the data structure for next round ===== //
			if (accept_link || history_limit == 0) {

				// TODO: if the warped difference is too small to be true
				std::cout << "Link: " << best_src << " -> " << best_des << " established (" << best_score << "," << warp_history << ")" << "..." << std::endl;

				// ===== Combine groups =====
				std::vector<int> fusion;
				fusion.reserve(split_results_[i][0].size() + split_results_[i][1].size());
				fusion.insert(fusion.end(), split_results_[i][0].begin(), split_results_[i][0].end());
				fusion.insert(fusion.end(), split_results_[i][1].begin(), split_results_[i][1].end());
				combined_groups.push_back(fusion);

				// ===== Update the warp record =====
				fusion_warp.push_back(best_score);
				warped_diff.push_back(fusion_warp);
			}
			else {

				std::cout << "Link: " << best_src << " - " << best_des << " rejected (" << best_score << "," << warp_history << ")" << "..." << std::endl;
				origin_score_on_fly(best_src, best_des) = 0;

				// Do not combine the group
				combined_groups.push_back(split_results_[i][0]);
				combined_groups.push_back(split_results_[i][1]);
				warped_diff.push_back(warped_diff_[split_indices_[i][0]]);
				warped_diff.push_back(warped_diff_[split_indices_[i][1]]);
			}

			// ===================================================== //
			// ===================================================== //

#ifdef DISPLAY
			std::vector<int> display_sequence(2);
			display_sequence[0] = srcNode_vec[best_link_index];
			display_sequence[1] = desNode_vec[best_link_index];
			img_ctrl.display_group(display_sequence); 
#endif
		}
	}

	warped_diff_ = warped_diff;
	return combined_groups;
}

Eigen::MatrixXf Matching_control::construct_score_grouped_nodes(
	const Eigen::MatrixXf&			origin_score_,
	std::vector<std::vector<int>>&	grouped_nodes_
)
{
	// Note, only upper triangle is used
	const int group_number = grouped_nodes_.size();
	Eigen::MatrixXf new_score = Eigen::MatrixXf::Ones(group_number, group_number) * -1;

	for (int i = 0; i < group_number - 1; i++) {
		for (int j = i + 1; j < group_number; j++) {

			std::vector<float> score_record;
			for (int ii = 0; ii < grouped_nodes_[i].size(); ii++) {
				for (int jj = 0; jj < grouped_nodes_[j].size(); jj++) {

					int left_index		= grouped_nodes_[i][ii];
					int right_index		= grouped_nodes_[j][jj];
					assert(left_index	!= right_index);

					if (left_index > right_index) {
						int swap_tmp	= left_index;
						left_index		= right_index;
						right_index		= swap_tmp;
					}

					score_record.push_back(origin_score_(left_index, right_index));
				}
			}

			// Metric one: average score
			float avg_score = std::accumulate(score_record.begin(), score_record.end(), 0) / score_record.size();
			
			// Set the new score matrix
			new_score(i, j) = avg_score;
		}
	}

	return new_score;
}

Eigen::MatrixXf Matching_control::construct_score_grouped_nodes_topN(
	const bool						minimum_guided_,
	const Eigen::MatrixXf&			origin_score_,
	std::vector<std::vector<int>>&	grouped_nodes_
)
{
	// Note, only upper triangle is used
	const int group_number = grouped_nodes_.size();
	Eigen::MatrixXf new_score = Eigen::MatrixXf::Ones(group_number, group_number) * -1;

	for (int i = 0; i < group_number - 1; i++) {
		for (int j = i + 1; j < group_number; j++) {

			std::vector<float> score_record;
			for (int ii = 0; ii < grouped_nodes_[i].size(); ii++) {
				for (int jj = 0; jj < grouped_nodes_[j].size(); jj++) {

					int left_index  = grouped_nodes_[i][ii];
					int right_index = grouped_nodes_[j][jj];
					assert(left_index != right_index);

					if (left_index > right_index) {
						int swap_tmp	= left_index;
						left_index		= right_index;
						right_index		= swap_tmp;
					}

					score_record.push_back(origin_score_(left_index, right_index));
				}
			}

			// Only top N strongest pairs are used for score computation
			sort_indices(score_record, minimum_guided_);

			// Metric one: average over the top N strongest scores
			int search_limit = (score_record.size() >= TOP_MAX_PAIR) ? TOP_MAX_PAIR : score_record.size();
			float avg_score = std::accumulate(score_record.begin(), score_record.begin() + search_limit, 0) / search_limit;

			// Set the new score matrix
			new_score(i, j) = avg_score;
		}
	}

	return new_score;
}

void Matching_control::translate_indices(
	std::vector<int>	grand_indicies_,
	std::vector<int>&	upper_set_,
	std::vector<int>&	lower_set_)
{
	const int upper_size = upper_set_.size();
	const int lower_size = lower_set_.size();

	// Declare the indices to be returned
	std::vector<int> upper_set(upper_size);
	std::vector<int> lower_set(lower_size);

	for (int i = 0; i < upper_size; i++) {
		upper_set[i] = grand_indicies_[upper_set_[i]];
	}
	for (int i = 0; i < lower_size; i++) {
		lower_set[i] = grand_indicies_[lower_set_[i]];
	}

	// Replace the original indices
	upper_set_ = upper_set;
	lower_set_ = lower_set;
}

Eigen::MatrixXf Matching_control::compute_gist_dist_all()
{
	// This function is for experiment purpose

	// Get the number of images in the database
	const int image_num = img_ctrl.getImageNum();
	Eigen::MatrixXf dist_mat = Eigen::MatrixXf::Ones(image_num, image_num) * -1;

	for (int i = 0; i < image_num - 1; i++) {
		for (int j = i + 1; j < image_num; j++) {
			std::vector<float> desc1 = Image_info::compute_gist(img_ctrl.getImageInfo(i).getImage());
			std::vector<float> desc2 = Image_info::compute_gist(img_ctrl.getImageInfo(j).getImage());
			dist_mat(i, j) = Image_info::compute_gist_dist(desc1, desc2);
		}
	}
	
	return dist_mat;
}

void Matching_control::triangulateTwoCameras(
	const int left_index_,
	const int right_index_, 
	const bool locally_computed_
)
{
	int left_index  = left_index_;
	int right_index = right_index_;

	if (left_index > right_index) {
		int swap_tmp = left_index;
		left_index   = right_index;
		right_index  = swap_tmp;
	}

	Image_info left_image  = img_ctrl.getImageInfo(left_index);
	Image_info right_image = img_ctrl.getImageInfo(right_index);

	if (!match.cameraPoseAndTriangulationFromFundamental(left_image, right_image, locally_computed_)) {
		std::cout << "Image pair (" << left_index << "," << right_index << ") triangulation failed ..." << std::endl;
	}
}

cv::Mat Matching_control::stitch_images(
	const cv::Mat left_,
	const cv::Mat right_)
{
	// Declare vector to hold the images
	std::vector<cv::Mat> img_vector;
	img_vector.push_back(left_);
	img_vector.push_back(right_);

	// Declare the stitcher objects
	cv::Mat panorama;
	cv::Stitcher stitcher = cv::Stitcher::createDefault();
	stitcher.stitch(img_vector, panorama);

	// Return the result
	return panorama;
}

// =============== BMVC: VSFM SfM Functions ===============
void Matching_control::set_vsfm_path(const std::string path_)
{
	vsfm_exec = path_ + "/visualSFM";
}

void Matching_control::triangulate_VSFM(const std::vector<int>& setA, const std::vector<int>& setB)
{
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// the models should be constructed for each sub-group
	// and then merged, current version supports one group
	// temporarily

	const int sizeA = setA.size();
	const int sizeB = setB.size();

	if (sizeA <= 0 || sizeB <= 0) {
		std::cout << "triangulate_VSFM: incorrect triangulate sequence ..." << std::endl;
	}

	std::string tmp_matches_name("tmp_matches.txt");
	std::vector<cv::Point2i> linkages = linkage_selection(setA, setB);
	std::string tmp_matches_path = write_matches_designated(tmp_matches_name, linkages);

	// Decide if an nvm already exists, use different command call in different cases
	std::string path;
	std::string name;
	Image_info::splitFilename(image_list_path, path, name);
	std::string nvm_path = path + "/o.nvm";
	if (!boost::filesystem::exists(nvm_path.c_str())) {
		std::string cmd = vsfm_exec + " sfm+import " + path + " " + nvm_path + " " + tmp_matches_path;
		system(cmd.c_str());
	}
	else {
		std::string cmd = vsfm_exec + " sfm+import+resume " + nvm_path + " " + path + "/o_tmp.nvm " + tmp_matches_path;
		system(cmd.c_str());

		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		// Pass the linkage validation here, then make the temporary
		// structure official
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		// Delete all original nvm file
		boost::filesystem::path delete_nvm(nvm_path.c_str());
		boost::filesystem::remove(delete_nvm);

		// Rename the tmp nvm file as nvm file
		boost::filesystem::path rename_nvm(std::string(path + "/o_tmp.nvm").c_str());
		boost::filesystem::rename(rename_nvm, delete_nvm);
	}
	
	
	boost::filesystem::path delete_matches(tmp_matches_path.c_str());
	boost::filesystem::remove(delete_matches);
}

std::vector<cv::Point2i> Matching_control::linkage_selection(const std::vector<int>& setA, const std::vector<int>& setB)
{
	const int sizeA = setA.size();
	const int sizeB = setB.size();

	if (sizeA <= 0 || sizeB <= 0) {
		std::cout << "linkage_selection: incorrect triangulate sequence ..." << std::endl;
	}

	std::vector<cv::Point2i> linkages;

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// ===== Linkages selection scheme =====
	// !!!!! Dummy implementation !!!!!
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	for (int i = 0; i < sizeA; i++) {
		for (int j = 0; j < sizeB; j++) {
			const int mach_num = match.get_matching_number(setA[i], setB[j]);
			if (mach_num > 0) {
				linkages.push_back(cv::Point2i(std::min(setA[i], setB[j]), std::max(setA[i], setB[j])));
			}
		}
	}

	return linkages;
}