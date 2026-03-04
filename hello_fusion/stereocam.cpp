// stereocam_depth.cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

static cv::Point g_lastClick(-1, -1);

static void onMouseDisparity(int event, int x, int y, int, void*)
{
    if (event == EVENT_LBUTTONDOWN) g_lastClick = Point(x, y);
}

static bool loadStereoCalibration(
    const string& path,
    Mat& K1, Mat& D1, Mat& K2, Mat& D2, Mat& R, Mat& T, Size& imageSize)
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open calibration file: " << path << "\n";
        return false;
    }

    fs["K1"] >> K1; fs["D1"] >> D1;
    fs["K2"] >> K2; fs["D2"] >> D2;
    fs["R"]  >> R;  fs["T"]  >> T;

    int w = 0, h = 0;
    fs["image_width"]  >> w;
    fs["image_height"] >> h;
    if (w <= 0 || h <= 0) {
        cerr << "Missing/invalid image_width or image_height in calibration file.\n";
        return false;
    }
    imageSize = Size(w, h);

    if (K1.empty() || D1.empty() || K2.empty() || D2.empty() || R.empty() || T.empty()) {
        cerr << "Calibration file missing required matrices (K1,D1,K2,D2,R,T).\n";
        return false;
    }
    return true;
}

static Mat colorizeDisparity(const Mat& disp16S, int numDisparities)
{
    Mat disp8U;
    // StereoSGBM outputs CV_16S disparity scaled by 16
    disp16S.convertTo(disp8U, CV_8U, 255.0 / (numDisparities * 16.0));
    Mat colored;
    applyColorMap(disp8U, colored, COLORMAP_JET);
    return colored;
}

int main(int argc, char** argv)
{
    // Usage:
    //   ./stereocam_depth calib.yml [leftIndex] [rightIndex]
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " calib.yml [camLeftIndex] [camRightIndex]\n";
        return 1;
    }

    const string calibPath = argv[1];
    const int camL = (argc >= 3) ? atoi(argv[2]) : 0;
    const int camR = (argc >= 4) ? atoi(argv[3]) : 1;

    Mat K1, D1, K2, D2, R, T;
    Size imageSize;
    if (!loadStereoCalibration(calibPath, K1, D1, K2, D2, R, T, imageSize)) {
        return 1;
    }

    VideoCapture capL(camL, CAP_ANY);
    VideoCapture capR(camR, CAP_ANY);
    if (!capL.isOpened() || !capR.isOpened()) {
        cerr << "Could not open cameras " << camL << " and/or " << camR << "\n";
        return 1;
    }

    // Match capture resolution to calibration resolution (strongly recommended)
    capL.set(CAP_PROP_FRAME_WIDTH,  imageSize.width);
    capL.set(CAP_PROP_FRAME_HEIGHT, imageSize.height);
    capR.set(CAP_PROP_FRAME_WIDTH,  imageSize.width);
    capR.set(CAP_PROP_FRAME_HEIGHT, imageSize.height);

    // Rectification outputs (renamed to avoid collision with SGBM P1/P2 ints)
    Mat R1, R2, P1mat, P2mat, Q;
    Rect roi1, roi2;

    double alpha = 0.0; // 0=crop to valid, 1=keep all with black borders
    stereoRectify(
        K1, D1, K2, D2, imageSize,
        R, T,
        R1, R2, P1mat, P2mat, Q,
        CALIB_ZERO_DISPARITY,
        alpha,
        imageSize,
        &roi1, &roi2
    );

    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(K1, D1, R1, P1mat, imageSize, CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(K2, D2, R2, P2mat, imageSize, CV_32FC1, map2x, map2y);

    // StereoSGBM parameters (tune as needed)
    int minDisparity = 0;
    int numDisparities = 16 * 8; // must be divisible by 16
    int blockSize = 5;           // odd, 3..11 typical

    // Penalties for disparity changes (must be ints; do NOT name them P1/P2)
    int sgbmP1 = 8 * blockSize * blockSize;
    int sgbmP2 = 32 * blockSize * blockSize;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        sgbmP1,
        sgbmP2,
        1,    // disp12MaxDiff
        10,   // preFilterCap
        100,  // uniquenessRatio
        32,   // speckleWindowSize
        2,    // speckleRange
        StereoSGBM::MODE_SGBM_3WAY
    );

    namedWindow("Rectified (L|R)", WINDOW_NORMAL);
    namedWindow("Disparity", WINDOW_NORMAL);
    setMouseCallback("Disparity", onMouseDisparity);

    cout << "Press q/ESC to quit. Click on Disparity window to print 3D at that pixel.\n";

    Mat frameL, frameR, rectL, rectR, grayL, grayR;
    Mat disp16S, disp32F, dispColor;
    Mat points3D; // CV_32FC3

    while (true)
    {
        capL >> frameL;
        capR >> frameR;
        if (frameL.empty() || frameR.empty()) break;

        // Rectify
        remap(frameL, rectL, map1x, map1y, INTER_LINEAR);
        remap(frameR, rectR, map2x, map2y, INTER_LINEAR);

        cvtColor(rectL, grayL, COLOR_BGR2GRAY);
        cvtColor(rectR, grayR, COLOR_BGR2GRAY);

        // Disparity (CV_16S scaled by 16)
        sgbm->compute(grayL, grayR, disp16S);

        // Convert to float disparity in pixels
        disp16S.convertTo(disp32F, CV_32F, 1.0 / 16.0);

        // 3D reprojection: points3D(x,y) = (X,Y,Z) in units of your calibration (often meters)
        reprojectImageTo3D(disp32F, points3D, Q, true);

        // Visualize disparity
        dispColor = colorizeDisparity(disp16S, numDisparities);

        // Visual sanity check: horizontal epipolar lines should align features
        Mat visRect;
        hconcat(rectL, rectR, visRect);
        for (int y = 0; y < visRect.rows; y += 40)
            line(visRect, Point(0, y), Point(visRect.cols, y), Scalar(0, 255, 0), 1);

        imshow("Rectified (L|R)", visRect);
        imshow("Disparity", dispColor);

        // Query depth on click
        if (g_lastClick.x >= 0 && g_lastClick.y >= 0 &&
            g_lastClick.x < points3D.cols && g_lastClick.y < points3D.rows)
        {
            Vec3f p = points3D.at<Vec3f>(g_lastClick.y, g_lastClick.x);
            // p[2] is depth Z along camera axis
            cout << "Pixel (" << g_lastClick.x << ", " << g_lastClick.y << ") -> "
                 << "X=" << p[0] << " Y=" << p[1] << " Z=" << p[2] << "\n";
            g_lastClick = Point(-1, -1);
        }

        int key = waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
    }

    capL.release();
    capR.release();
    destroyAllWindows();
    return 0;
}
