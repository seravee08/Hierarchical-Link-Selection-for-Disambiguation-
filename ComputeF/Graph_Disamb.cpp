#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <stdlib.h>

#include "Graph_Disamb.h"

Graph_disamb::Graph_disamb(const int node_num_) :
	node_num	(node_num_)
{
	layOut = Eigen::MatrixXi::Zero(node_num, node_num);
}

void Graph_disamb::addEdge(
	int source_index_,
	int desti_index_
)
{
	// Validate the source and destination indices
	assert(source_index_ != desti_index_);
	if (source_index_ > desti_index_) {
		int swap_tmp = source_index_;
		source_index_ = desti_index_;
		desti_index_ = swap_tmp;
	}

	layOut(source_index_, desti_index_) = 1;
}

Eigen::MatrixXi Graph_disamb::getLayout()
{
	return layOut;
}