#pragma once

#ifndef PARAMETERS_H
#define PARAMETERS_H

#define OVERLAP_THRESHOLD		0.3			// The image pair is rejected if the overlap is lower than this value

#define HOMOGRAPHY_METHOD		CV_LMEDS	// Method to compute homography, options: 0 ; CV_RANSAC ; CV_LMEDS

#define GIST_SIZE				128			// Gist parameters

#define	TOP_N_CANDIDATE			5			// Top N candidate linkes to compute homographies when deciding link

#endif // !PARAMETERS_H
