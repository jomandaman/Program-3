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

const int SCALE = 8;

struct MatchData {
    Mat img1, img2, output;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    int numMatches;
};

void on_trackbar(int, void* userdata) {
    MatchData* data = static_cast<MatchData*>(userdata);

    // Sort the matches based on the distance
    sort(data->matches.begin(), data->matches.end());

    // Draw only the top 'numMatches' matches
    vector<DMatch> topMatches(data->matches.begin(), data->matches.begin() + min(data->numMatches, static_cast<int>(data->matches.size())));

    // Draw the top matches
    drawMatches(data->img1, data->keypoints1, data->img2, data->keypoints2, topMatches, data->output, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Display the output image
    imshow("output", data->output);
}

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

    // Create an empty output image
    Mat output;

    // Initialize match data
    MatchData data;
    data.numMatches = 50;
    data.matches = matches;
    data.output = output;
    data.img1 = img1;
    data.keypoints1 = keypoints1;
    data.img2 = img2;
    data.keypoints2 = keypoints2;
    
    // Create a window to display the output
    namedWindow("output", WINDOW_NORMAL);

    // Trackbar to control the number of matches displayed
    createTrackbar("Number of matches", "output", &data.numMatches, matches.size(), on_trackbar, &data);

    // Call the trackbar callback once to display the initial state
    on_trackbar(0, &data);

    // Resize the window
    resizeWindow("output", data.output.cols / SCALE, data.output.rows / SCALE);

    // Wait for the user to press a key
    waitKey(0);
    
    return 0;
}
