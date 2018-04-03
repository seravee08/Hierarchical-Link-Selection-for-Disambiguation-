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

// ===== Boost Library =====
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include "utility.h"

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

