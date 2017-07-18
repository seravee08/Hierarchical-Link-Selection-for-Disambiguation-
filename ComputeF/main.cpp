#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "SiftIO.h"
#include "Matching.h"

using namespace std;
using namespace cv;

int main()
{
	//int point_count = 100;
	//vector<Point2f> points1(point_count);
	//vector<Point2f> points2(point_count);

	//for (int i = 0; i < point_count; i++) {
	//	points1[i] = Point2f(float(rand() % 640), float(rand() % 480));
	//	points2[i] = Point2f(float(rand() % 640), float(rand() % 480));
	//}

	//Mat F = findFundamentalMat(points1, points2, CV_FM_RANSAC, 2, 0.50);

	//cout << F << endl;

	// ===== Read in sift and auxiliary information =====
	std::string img_name1 = "C:/Users/fango/OneDrive/Documents/GitHub/ComputeF/data/view_0000.jpg";
	std::string img_name2 = "C:/Users/fango/OneDrive/Documents/GitHub/ComputeF/data/view_0001.jpg";
	Image_info img1(img_name1);
	Image_info img2(img_name2);
	img1.read_Auxililiary();
	img2.read_Auxililiary();
	img1.read_Sift();
	img2.read_Sift();

	Eigen::Matrix<float, 2, Eigen::Dynamic> coords1 = img1.get_coordinates();
	Eigen::Matrix<float, 2, Eigen::Dynamic> coords2 = img2.get_coordinates();

	// ===== Read in matchings =====
	std::string matchings_name = "C:/Users/fango/OneDrive/Documents/GitHub/ComputeF/data/matchings.txt";
	Matching match(matchings_name);
	match.read_matchings();

	Eigen::Matrix<int, 2, Eigen::Dynamic> matchings = match.get_matchings(0);

	// ===== TODO: Lib-igl =====
	const int matching_number = matchings.cols();
	vector<Point2f> points1(matching_number);
	vector<Point2f> points2(matching_number);

	for (int i = 0; i < matching_number; i++) {
		int upper_index = matchings(0, i);
		int lower_index = matchings(1, i);
		points1[i] = Point2f(coords1(0, upper_index), coords1(1, upper_index));
		points2[i] = Point2f(coords2(0, lower_index), coords2(1, lower_index));
	}

	Mat F = findFundamentalMat(points1, points2, CV_FM_RANSAC, 2, 0.99);

	cout << F << endl;

	system("pause");

	return 0;
}