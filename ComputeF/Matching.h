#pragma once

#ifndef MATCHING_H
#define MATCHING_H

#include <vector>

#include "Eigen/Core"

class Matching {
public:
	// Constructor
	Matching(const std::string matching_path_);

	// Destructor
	~Matching() {};

	// Read in matchings 
	void read_matchings();

	// Get matching matrix by index
	Eigen::Matrix<int, 2, Eigen::Dynamic> get_matchings(int index_);

private:
	int pair_num;

	std::string matching_path;

	std::vector<std::string> left_img_names;
	std::vector<std::string> right_img_names;

	std::vector<int> vec_matching_number;
	std::vector<Eigen::Matrix<int, 2, Eigen::Dynamic>> vec_matching;
};

#endif // !MATCHING_H
