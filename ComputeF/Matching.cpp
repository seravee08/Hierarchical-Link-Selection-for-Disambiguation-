#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <stdlib.h>

#include "Matching.h"

Matching::Matching(const std::string matching_path_) :
	matching_path	(matching_path_),
	pair_num		(0)
{

}

void Matching::read_matchings()
{
	// ===== Open input file stream =====
	std::ifstream match_in(matching_path.c_str(), std::ios::in);
	assert(match_in.is_open());

	// ===== Read in matchings =====
	int matching_number;
	std::string left_name;
	std::string right_name;
	std::string dummy;

	while (!match_in.eof()) {
		match_in >> left_name;
		match_in >> right_name;
		match_in >> matching_number;

		// Push the names and matching number into the vectors
		left_img_names. push_back(left_name);
		right_img_names.push_back(right_name);
		vec_matching_number.push_back(matching_number);

		// Read in the matchings
		std::vector<int> upper_row(matching_number);
		std::vector<int> lower_row(matching_number);
		for (int i = 0; i < matching_number; i++) {
			match_in >> upper_row[i];
		}
		for (int i = 0; i < matching_number; i++) {
			match_in >> lower_row[i];
		}

		// Empty line as separator
		match_in >> dummy;

		// Load the buffers into matrix
		Eigen::Matrix<int, 1, Eigen::Dynamic> upper_row_mat(1, matching_number);
		Eigen::Matrix<int, 1, Eigen::Dynamic> lower_row_mat(1, matching_number);
		Eigen::Matrix<int, 2, Eigen::Dynamic> matchings_mat(2, matching_number);

		upper_row_mat = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic>>(upper_row.data(), 1, matching_number);
		lower_row_mat = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic>>(lower_row.data(), 1, matching_number);
		matchings_mat << upper_row_mat, lower_row_mat;
		vec_matching.push_back(matchings_mat);

		// Increase the pair of matchings counter by 1
		pair_num++;
	}
}

Eigen::Matrix<int, 2, Eigen::Dynamic> Matching::get_matchings(int index_)
{
	return vec_matching[index_];
}