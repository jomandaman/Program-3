/**
 * @file Program3.cpp
 * 
 * This program demonstrates the use of various feature detection and matching algorithms in OpenCV.
 * It allows the user to interactively select a feature detection algorithm (SIFT, ORB, BRISK, SURF) 
 * and control the number of displayed matches between two input images. The program provides a GUI using 
 * trackbars to adjust the algorithm and the number of matches, while visualizing the results in real-time.
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
const int INITIAL_NUM_MATCHES = 50;

// Structure to store match data, including images, keypoints, descriptors, and matches
struct MatchData {
    Mat img1, img2, output, descriptors1, descriptors2; // Input images, output image, and descriptors for each image
    vector<KeyPoint> keypoints1, keypoints2;            // Keypoints for each input image
    vector<DMatch> matches;                             // Matches between keypoints of the two input images
    Ptr<Feature2D> detector;                            // Pointer to the feature detector and descriptor
    Ptr<DescriptorMatcher> matcher;                     // Pointer to the descriptor matcher
    int numMatches;                                     // Number of matches to display
    int selectedDetector;                               // Index of the selected feature detector (0: SIFT, 1: ORB, 2: BRISK, 3: SURF)
};

/**
 * Updates the detector and descriptor in the MatchData structure.
 * Supported detectors are SIFT, ORB, BRISK, and SURF.
 * 
 * @param data Pointer to the MatchData structure containing the current detector and other match-related data.
 * preconditions: data should be a valid pointer to a MatchData structure and data->selectedDetector should be within the range [0, 3].
 * postconditions: data->detector is updated to the selected feature detector and descriptor.
 */
void update_detector(MatchData* data) {
    switch (data->selectedDetector) {
        case 0:
            data->detector = SIFT::create();
            break;
        case 1:
            data->detector = ORB::create();
            break;
        case 2:
            data->detector = BRISK::create();
            break;
        case 3:
            data->detector = xfeatures2d::SURF::create();
            break;
        default:
            data->detector = SIFT::create();
            break;
    }
}

/**
 * Convert detector index value to a string representation.
 * Supported detectors are SIFT, ORB, BRISK, and SURF.
 * 
 * @param value Integer value representing the detector index (0: SIFT, 1: ORB, 2: BRISK, 3: SURF).
 * @return String representation of the detector corresponding to the given value.
 * 
 * preconditions: value should be within the range [0, 3].
 * postconditions: returns a string representation of the selected detector or "Unknown" for invalid values.
 */
string trackbar_label(int value) {
    switch (value) {
        case 0: return "SIFT";
        case 1: return "ORB";
        case 2: return "BRISK";
        case 3: return "SURF";
        default: return "Unknown";
    }
}

/**
 * Callback function when trackbar values are changed. Updates the detector,
 * recomputes keypoints and matches, and redraws the output image with the top matches.
 * 
 * @param userdata Pointer to user data, which should be a pointer to a MatchData structure.
 * 
 * preconditions: userdata should be a valid pointer to a MatchData structure
 * postconditions: keypoints and matches are recomputed if the detector has changed.
 * - Output image is updated with the top matches and displayed in the OpenCV window.
 */
void on_trackbar(int, void* userdata) {
    MatchData* data = static_cast<MatchData*>(userdata);

    // Fetch trackbar values
    data->numMatches = getTrackbarPos("Number of matches", "output");
    data->selectedDetector = getTrackbarPos("Detector (0:SIFT, 1:ORB, 2:BRISK, 3:SURF)", "output");

    // Update the detector and recompute keypoints and matches if the detector has changed
    static int prevSelectedDetector = data->selectedDetector;
    if (prevSelectedDetector != data->selectedDetector) {
        prevSelectedDetector = data->selectedDetector;
        update_detector(data);
        data->detector->detectAndCompute(data->img1, noArray(), data->keypoints1, data->descriptors1);
        data->detector->detectAndCompute(data->img2, noArray(), data->keypoints2, data->descriptors2);
        data->matcher->match(data->descriptors1, data->descriptors2, data->matches);
    }

    // Sort the matches based on the distance
    sort(data->matches.begin(), data->matches.end());

    // Draw only the top 'numMatches' matches
    vector<DMatch> topMatches(data->matches.begin(), data->matches.begin() + min(data->numMatches, static_cast<int>(data->matches.size())));

    // Draw the top matches
    drawMatches(data->img1, data->keypoints1, data->img2, data->keypoints2, topMatches, data->output, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Display the detector name and the number of matches on the output image
    stringstream ss;
    ss << trackbar_label(data->selectedDetector) << ", Matches: " << topMatches.size();
    putText(data->output, ss.str(), Point(30, 150), FONT_HERSHEY_SIMPLEX, 6, Scalar(0, 255, 0), 8);


    // Display the output image
    imshow("output", data->output);
}

/**
 * Main function of the program.
 * 
 * preconditions: The input images "kittens1.jpg" and "kittens2.jpg" should be available in the working directory.
 * postconditions: - An OpenCV window is displayed with the top matches between the two images.
 * - Trackbars are created to control the number of matches displayed and the detector used.
 * - The program waits for the user to press a key before exiting.
 * 
 * @return 0 if the program exits successfully
 */
int main() {
    // Read the images
    Mat img1 = imread("kittens1.jpg", IMREAD_COLOR);
    Mat img2 = imread("kittens2.jpg", IMREAD_COLOR);

    // Initialize match data
    MatchData data;
    data.numMatches = INITIAL_NUM_MATCHES;
    data.img1 = img1;
    data.img2 = img2;
    data.selectedDetector = 0;
    update_detector(&data);
    data.matcher = BFMatcher::create();

    // Detect keypoints and compute descriptors for both images
    data.detector->detectAndCompute(img1, noArray(), data.keypoints1, data.descriptors1);
    data.detector->detectAndCompute(img2, noArray(), data.keypoints2, data.descriptors2);

    // Compute matches between keypoints/descriptors in the two images
    data.matcher->match(data.descriptors1, data.descriptors2, data.matches);

    // Create a window to display the output
    namedWindow("output", WINDOW_NORMAL);

    // Trackbar to control the number of matches displayed
    createTrackbar("Number of matches", "output", NULL, data.matches.size(), on_trackbar, &data);
    setTrackbarPos("Number of matches", "output", INITIAL_NUM_MATCHES);

    // Trackbar to control the detector
    createTrackbar("Detector (0:SIFT, 1:ORB, 2:BRISK, 3:SURF)", "output", NULL, 3, on_trackbar, &data);

    /**
     * Some thoughts about the detectors used:
     * SIFT: Fairly accurate and robust, but seemingly slower and computationally more expensive than the other detectors.
     * ORB: Faster and more efficient than SIFT, while still providing good matching performance in this case.
     * BRISK: Very fast compared to the others with good matching performance.
     * SURF: Aims to be faster than SIFT, but might not be as accurate in some cases like this.
     */

    // Call the trackbar callback once to display the initial state
    on_trackbar(0, &data);

    // Resize the window
    resizeWindow("output", data.output.cols / SCALE, data.output.rows / SCALE);

    // Wait for the user to press a key
    waitKey(0);
    
    return 0;
}

/**
 * Some thoughts on the various detectors and the program:
 * * SIFT, ORB, BRISK, and SURF all have their strengths and weaknesses in terms of accuracy, speed, and robustness.
 * * This program allows for a quick comparison of the detectors' matching performance by using trackbars and real-time feedback, atlhough 
 *    In practice, matching performance varies depending on the images and the application.
 * * As discussed in class, likely the best result would be achieved by combining multiple detectors, such as SIFT and ORB.
 *
 * Overall, this project demonstrates an interactive way to explore and compare feature detectors using OpenCV. 
 * The real-time visualization and easy switching between detectors provide valuable insights into their performance 
 * characteristics and practical implications. Going forward, I would like to explore the use of multiple detectors and 
 * the use of other feature descriptors to improve matching performance.
 */
