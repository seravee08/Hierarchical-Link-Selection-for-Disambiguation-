#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include <vector>

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

#define SELF_DEFINE_SWAP(a,b) {int temp; temp=a; a=b; b=temp;}

#endif // !UTILITY_H