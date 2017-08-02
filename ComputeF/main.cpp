#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "Matching_control.h"

using namespace std;
using namespace cv;

int main()
{
	// Declare path to the image list
	std::string image_list_path = "C:/Users/fango/OneDrive/Documents/GitHub/ComputeF/data/fc/image_list.txt";

	Matching_control match_ctrl(image_list_path);
	// match_ctrl.readIn_Keypoints();
	match_ctrl.readIn_Matchings();
	//match_ctrl.delete_bad_matchings();
	//match_ctrl.compute_Homography();
	//match_ctrl.rectify_matching_homoMask();

	//const int number_image = match_ctrl.getImage_num();
	//for (int i = 0; i < number_image - 1; i++) {
	//	for (int j = i + 1; j < number_image; j++) {
	//		cout << "Link (" << i << " , " << j << ")  " <<  match_ctrl.getWarped_diff_value(i, j) << "  " << match_ctrl.getMatch_number(i, j) << endl;
	//		// match_ctrl.displayMatchings(i, j);
	//	}
	//}

	// Eigen::MatrixXf warped_diff = match_ctrl.getWarped_diff_mat();
	Eigen::MatrixXf matching_number = match_ctrl.getMatching_number_mat().cast<float>();
	// Eigen::MatrixXf gist_dist = match_ctrl.compute_gist_dist_all();
	std::vector<Graph_disamb> graphs = match_ctrl.constructGraph_with_homography_validate(false, matching_number);
	// match_ctrl.writeOut_Matchings(graphs);
	match_ctrl.writeOut_Layout(graphs);

	for (int i = 0; i < graphs.size(); i++) {
		cout << "Graph " << i << ": " << std::endl;
		cout << graphs[i].getLayout() << std::endl;
		system("pause");
	}

	system("pause");

	return 0;
}