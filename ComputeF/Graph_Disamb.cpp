#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <stdlib.h>
#include <numeric>

#include "Graph_Disamb.h"

Graph_disamb::Graph_disamb(const int node_num_) :
	node_num	(node_num_)
{
	layOut = Eigen::MatrixXi::Zero(node_num, node_num);
	node_status = std::vector<int>(node_num, 0);
}

void Graph_disamb::addEdge(
	const int source_index_,
	const int desti_index_
)
{
	// Validate the source and destination indices
	assert(source_index_ != desti_index_);

	// Update the layout and node status
	layOut(std::min(source_index_, desti_index_), std::max(source_index_, desti_index_)) = 1;
	node_status[source_index_]															 = 1;
	node_status[desti_index_]															 = 1;
}

int Graph_disamb::get_node_status(int index_)
{
	return node_status[index_];
}

int Graph_disamb::number_nodes_inGraph()
{
	return std::accumulate(node_status.begin(), node_status.end(), 0);
}

Eigen::MatrixXi Graph_disamb::getLayout()
{
	return layOut;
}