#include <iostream>
#include <string>
#include <cassert>
#include <stdlib.h>
#include <algorithm>
#include <list>
#include <numeric>
#include <random>

#include "utility.h"

template void sort_indices<int>(const std::vector<int>& target_, std::vector<int>& indices_, bool increasing);
template void sort_indices<float>(const std::vector<float>& target_, std::vector<int>& indices_, bool increasing);

template <typename T>
void sort_indices(const std::vector<T>& target_, std::vector<int>& indices_, bool increasing)
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