/**
 * @file Program3.cpp
 * 
 * This program demonstrates the basic functionality of OpenCV methods for image processing tasks.
 * The program applies the chroma key effect on a foreground image, replacing a selected color
 * with a background image. The user can interactively control the threshold using a trackbar.
 * 
 * Comments posted throughout, some general thoughts at the bottom.
 * 
 * @author Josiah Zacharias
 * Contact: josiahz@uw.edu
 * UW 587 with Clark Olsen
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    // Read the images
    Mat img1 = imread("kittens1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("kittens2.jpg", IMREAD_GRAYSCALE);

    // Create a SIFT detector/descriptor
    Ptr<SIFT> detector = SIFT::create();

    // Detect keypoints and compute descriptors for both images
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // Draw the keypoints on the images
    Mat img_keypoints1, img_keypoints2;
    drawKeypoints(img1, keypoints1, img_keypoints1);
    drawKeypoints(img2, keypoints2, img_keypoints2);

    // Create a brute-force matcher
    Ptr<BFMatcher> matcher = BFMatcher::create();

    // Compute matches between keypoints/descriptors in the two images
    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Draw the matches
    Mat output;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, output);

    // Output the drawn matches to a file
    imwrite("output.jpg", output);

    // Display the matches
    int SCALE = 8;
    namedWindow("output", WINDOW_NORMAL);
    resizeWindow("output", output.cols / SCALE, output.rows / SCALE);
    imshow("output", output);

    waitKey(0);

    return 0;
}
