#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "utility.h"
#include "Matching_control.h"

using namespace std;
using namespace cv;

int return_index(vector<pair<int, int>>& queue) {
	int index;
	int height = -1;
	int num_ppl = -1;
	const int num = queue.size();
	for (int i = 0; i < num; i++) {
		if (queue[i].first > height) {
			height = queue[i].first;
			num_ppl = queue[i].second;
			index = i;
		}
		else if (queue[i].first == height) {
			if (queue[i].second < num_ppl) {
				index = i;
				num_ppl = queue[i].second;
			}
		}
	}
	queue[index].first = -1;
	return index;
}

void reconstruct(vector<pair<int, int>>& queue) {

	const int num = queue.size();	
	vector<pair<int, int>> cpy = queue;
	vector<pair<int, int>> res;

	for (int i = 0; i < num; i++) {
		int ind = return_index(cpy);
		const pair<int, int>& p_ = queue[ind];
		const int& n_ = queue[ind].second;

		// Construct true sequence
		res.insert(res.begin() + n_, p_);
	}

	// Output results
	for (int i = 0; i < num; i++) {
		cout << res[i].first << "  " << res[i].second << endl;
	}
}

int main()
{
	//// Declare path to the image list
	//std::string image_list_path = "C:/Users/fango/OneDrive/Documents/GitHub/ComputeF/data/oats/image_list.txt";

	//Matching_control match_ctrl(image_list_path);
	//match_ctrl.readIn_Keypoints();
	//match_ctrl.readIn_Matchings();
	//// match_ctrl.delete_bad_matchings();
	//// match_ctrl.compute_Homography();
	//// match_ctrl.rectify_matching_homoMask();

	////const int number_image = match_ctrl.getImage_num();
	////for (int i = 0; i < number_image - 1; i++) {
	////	for (int j = i + 1; j < number_image; j++) {
	////		cout << "Link (" << i << " , " << j << ")  " <<  match_ctrl.getWarped_diff_value(i, j) << "  " << match_ctrl.getMatch_number(i, j) << endl;
	////		// match_ctrl.displayMatchings(i, j);
	////	}
	////}

	//// Eigen::MatrixXf warped_diff = match_ctrl.getWarped_diff_mat();
	//// Eigen::MatrixXf matching_number = match_ctrl.getMatching_number_mat().cast<float>();
	//// Eigen::MatrixXf gist_dist = match_ctrl.compute_gist_dist_all();
	//// std::vector<Graph_disamb> graphs = match_ctrl.constructGraph_with_homography_validate(false, matching_number);
	//// std::vector<Graph_disamb> graphs = match_ctrl.constructGraph_with_stitcher(false, matching_number);
	//// match_ctrl.iterative_group_split(false, matching_number);
	//// match_ctrl.writeOut_Matchings(graphs);
	//// match_ctrl.writeOut_Layout(graphs);

	////for (int i = 0; i < graphs.size(); i++) {
	////	cout << "Graph " << i << ": " << std::endl;
	////	cout << graphs[i].getLayout() << std::endl;
	////	system("pause");
	////}


	//match_ctrl.triangulateTwoCameras(0, 1);


	////float* points_array1 = new float[4];
	////float* points_array2 = new float[4];

	////points_array1[0] = 0;
	////points_array1[1] = 1;
	////points_array1[2] = 2;
	////points_array1[3] = 3;

	////points_array2[0] = 0;
	////points_array2[0] = 1;
	////points_array2[0] = 2;
	////points_array2[0] = 3;


	////Mat t(1, 2, CV_32FC2, points_array1);
	////Mat s(1, 2, CV_32FC2, points_array2);

	////Mat normalized_t;
	////Mat camMat = (Mat_<float>(3, 3) << 20, 0, 20, 0, 20, 20, 0, 0, 1);
	////


	////undistortPoints(t, normalized_t, camMat, Mat());

	////cout << normalized_t << endl;



	////delete points_array1;
	////delete points_array2;



	//vector<pair<int, int>> input(6);
	//input[0].first = 7;
	//input[0].second = 0;
	//input[1].first = 4;
	//input[1].second = 4;
	//input[2].first = 7;
	//input[2].second = 1;
	//input[3].first = 5;
	//input[3].second = 0;
	//input[4].first = 6;
	//input[4].second = 1;
	//input[5].first = 5;
	//input[5].second = 2;

	//reconstruct(input);
	//[[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
		   
	

	string direc = string("C:/Users/fango/Desktop/DATA/alexander_nevsky_cathedral/O");
	string list_path = create_list(direc);
	Matching_control mach_ctrl(list_path);

	int maxWidth, maxHeight;
	int minWidth, minHeight;
	mach_ctrl.return_max_width_height(maxWidth, maxHeight);
	mach_ctrl.return_min_width_height(minWidth, minHeight);

	cout << "Max width: " << maxWidth << endl;
	cout << "Max height: " << maxHeight << endl;
	cout << "Min width: " << minWidth << endl;
	cout << "Min height: " << minHeight << endl;

	//string direc = string("C:/Users/fango/OneDrive/Documents/GitHub/ComputeF/data/Semper-Statue");
	//string list_path = create_list(direc);
	//Matching_control mach_ctrl(list_path);
	//mach_ctrl.compute_Sift(0);
	//mach_ctrl.compute_Sift(1);
	//mach_ctrl.compute_Matchings(0, 1);
	////mach_ctrl.displayMatchings(0, 1, false, true);
	//mach_ctrl.triangulateTwoCameras(0, 1, true);

	system("pause");

	return 0;
}