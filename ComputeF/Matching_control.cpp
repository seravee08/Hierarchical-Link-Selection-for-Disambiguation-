#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <numeric>
#include <stdlib.h>

#include "Parameters.h"
#include "utility.h"
#include "Matching_control.h"

Matching_control::Matching_control(const std::string image_list_path_) :
	img_ctrl	(image_list_path_),
	match		(image_list_path_)
{

}

void Matching_control::readIn_Keypoints()
{
	img_ctrl.read_Auxiliary();
	img_ctrl.read_Sift();

	std::cout << "Keypoints readin completes ..." << std::endl;
}

void Matching_control::readIn_Matchings()
{
	match.read_matchings();

	std::cout << "Matchings readin completes ..." << std::endl;
}

void Matching_control::writeOut_Matchings(std::vector<Graph_disamb>& graphs_)
{
	match.write_matches(graphs_);
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

float Matching_control::getWarped_diff_value(
	const int	left_index_,
	const int	right_index_
)
{
	assert(left_index_ < right_index_);
	return match.getWarped_diff()(left_index_, right_index_);
}

void Matching_control::displayMatchings(
	const int	left_index_,
	const int	right_index_,
	const bool	use_outlier_mask_
)
{
	// Retrieve the corresponding image strucutres
	Image_info& image_left  = img_ctrl.getImageInfo(left_index_);
	Image_info& image_right = img_ctrl.getImageInfo(right_index_);
	const int matching_number = match.get_matching_number(image_left, image_right);

	if (matching_number > 0) {

		if (use_outlier_mask_ == true) {
			match.display_matchings(image_left, image_right, match.get_outlier_mask(image_left, image_right));
		}
		else {
			match.display_matchings(image_left, image_right);
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
						img_l.read_Auxililiary();
						img_l.read_Sift();
						img_r.read_Auxililiary();
						img_r.read_Sift();

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
	const bool minimum_guided_,
	const Eigen::MatrixXf& scores_,
	std::vector<std::vector<int>> grouped_nodes_
)
{

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