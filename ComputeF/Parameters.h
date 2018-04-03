#pragma once

#ifndef PARAMETERS_H
#define PARAMETERS_H

#define OVERLAP_THRESHOLD		0.3			// The image pair is rejected if the overlap is lower than this value

#define HOMOGRAPHY_METHOD		CV_LMEDS	// Method to compute homography, options: 0 ; CV_RANSAC ; CV_LMEDS

#define GIST_SIZE				128			// Gist parameters

#define	TOP_N_CANDIDATE			5			// Top N candidate linkes to compute homographies when deciding link

#define SPLIT_LIMIT				2			// Stop splitting when number of nodes in graph is no greater than this threshold

#define WARP_HISTORY			5			// The last N warp difference records affect the link decision

#define WARP_UPPER_THRESHOLD	1.0			// Maximum allowed percentage over the current warp difference history

#define WARP_LOWER_THRESHOLD	0.25		// Maximum allowed percentage below the current warp difference history

#define TOP_MAX_PAIR			5			// Use the top N pairs as the score between two groups

#define MIN_INLIERS				50			// Minimum number of inliers required to pass the validation test

#define SIFT_GOOD_THRESHOLD		0.7			// Determine the threshold between first matching pair and the second, used to filter matches

// =============================================================================================//

#define DISPLAY								// Enable the display function of the program

#endif // !PARAMETERS_H