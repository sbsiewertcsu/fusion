// fuse_two_usb_cams.cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

static bool fuseTwoFrames(const Mat& imgA, const Mat& imgB, Mat& fusedOut)
{
    if (imgA.empty() || imgB.empty()) return false;

    // ORB features (fast and widely available)
    Ptr<ORB> orb = ORB::create(3000);

    Mat grayA, grayB;
    cvtColor(imgA, grayA, COLOR_BGR2GRAY);
    cvtColor(imgB, grayB, COLOR_BGR2GRAY);

    vector<KeyPoint> kpsA, kpsB;
    Mat desA, desB;
    orb->detectAndCompute(grayA, noArray(), kpsA, desA);
    orb->detectAndCompute(grayB, noArray(), kpsB, desB);

    if (desA.empty() || desB.empty() || kpsA.size() < 10 || kpsB.size() < 10)
        return false;

    // KNN matches + Lowe ratio test
    BFMatcher matcher(NORM_HAMMING, false);
    vector<vector<DMatch>> knn;
    matcher.knnMatch(desA, desB, knn, 2);

    const float ratio = 0.75f;
    vector<DMatch> good;
    good.reserve(knn.size());
    for (const auto& pair : knn)
    {
        if (pair.size() < 2) continue;
        if (pair[0].distance < ratio * pair[1].distance)
            good.push_back(pair[0]);
    }

    if (good.size() < 12) return false;

    vector<Point2f> ptsA, ptsB;
    ptsA.reserve(good.size());
    ptsB.reserve(good.size());
    for (const auto& m : good)
    {
        ptsA.push_back(kpsA[m.queryIdx].pt); // in A
        ptsB.push_back(kpsB[m.trainIdx].pt); // in B
    }

    // Estimate homography mapping B -> A
    Mat inlierMask;
    Mat H = findHomography(ptsB, ptsA, RANSAC, 4.0, inlierMask);
    if (H.empty()) return false;

    // Compute output canvas that fits both A and warped B
    int hA = imgA.rows, wA = imgA.cols;
    int hB = imgB.rows, wB = imgB.cols;

    vector<Point2f> cornersB = { {0,0}, {(float)wB,0}, {(float)wB,(float)hB}, {0,(float)hB} };
    vector<Point2f> warpedCornersB;
    perspectiveTransform(cornersB, warpedCornersB, H);

    vector<Point2f> cornersA = { {0,0}, {(float)wA,0}, {(float)wA,(float)hA}, {0,(float)hA} };

    float xmin = cornersA[0].x, ymin = cornersA[0].y, xmax = cornersA[0].x, ymax = cornersA[0].y;
    auto absorb = [&](const Point2f& p){
        xmin = min(xmin, p.x); ymin = min(ymin, p.y);
        xmax = max(xmax, p.x); ymax = max(ymax, p.y);
    };
    for (auto& p : cornersA) absorb(p);
    for (auto& p : warpedCornersB) absorb(p);

    int tx = (int)floor(-xmin + 0.5f);
    int ty = (int)floor(-ymin + 0.5f);
    int outW = (int)ceil(xmax - xmin + 0.5f);
    int outH = (int)ceil(ymax - ymin + 0.5f);

    Mat T = (Mat_<double>(3,3) << 1,0,tx,  0,1,ty,  0,0,1);
    Mat Ht = T * H;

    Mat warpedB(outH, outW, imgB.type(), Scalar::all(0));
    warpPerspective(imgB, warpedB, Ht, Size(outW, outH), INTER_LINEAR, BORDER_CONSTANT);

    Mat canvasA(outH, outW, imgA.type(), Scalar::all(0));
    imgA.copyTo(canvasA(Rect(tx, ty, wA, hA)));

    // Build masks (where pixels exist)
    Mat maskA, maskB;
    cvtColor(canvasA, maskA, COLOR_BGR2GRAY);
    cvtColor(warpedB, maskB, COLOR_BGR2GRAY);
    threshold(maskA, maskA, 0, 255, THRESH_BINARY);
    threshold(maskB, maskB, 0, 255, THRESH_BINARY);

    // Feather blending using distance transform
    Mat invA, invB, distA, distB;
    bitwise_not(maskA, invA);
    bitwise_not(maskB, invB);

    // distanceTransform wants 8-bit, single-channel, with zeros as "objects"
    distanceTransform(invA, distA, DIST_L2, 3);
    distanceTransform(invB, distB, DIST_L2, 3);

    Mat wAfloat, wBfloat;
    Mat denom = distA + distB + 1e-6f;
    divide(distA, denom, wAfloat);
    divide(distB, denom, wBfloat);

    // Where only one image exists, force weight to 1 there
    Mat onlyA, onlyB;
    compare(maskA, 0, onlyA, CMP_GT);
    compare(maskB, 0, onlyB, CMP_GT);

    Mat overlap;
    bitwise_and(onlyA, onlyB, overlap);

    // onlyA = A exists, B doesn't
    Mat notB;
    bitwise_not(onlyB, notB);
    Mat aOnly;
    bitwise_and(onlyA, notB, aOnly);

    // onlyB = B exists, A doesn't
    Mat notA;
    bitwise_not(onlyA, notA);
    Mat bOnly;
    bitwise_and(onlyB, notA, bOnly);

    wAfloat.setTo(1.0f, aOnly);
    wBfloat.setTo(1.0f, bOnly);

    // Convert weights to 3-channel
    Mat wA3, wB3;
    vector<Mat> chA(3, wAfloat), chB(3, wBfloat);
    merge(chA, wA3);
    merge(chB, wB3);

    // Blend in float
    Mat fA, fB, fusedF;
    canvasA.convertTo(fA, CV_32FC3);
    warpedB.convertTo(fB, CV_32FC3);

    fusedF = fA.mul(wA3) + fB.mul(wB3);
    fusedF.convertTo(fusedOut, CV_8UC3);

    return true;
}

int main(int argc, char** argv)
{
    // Camera indices can vary. Typical: 0 and 1.
    int camAIndex = 0;
    int camBIndex = 2;
    if (argc >= 3)
    {
        camAIndex = atoi(argv[1]);
        camBIndex = atoi(argv[2]);
    }

    VideoCapture capA(camAIndex, CAP_ANY);
    VideoCapture capB(camBIndex, CAP_ANY);

    if (!capA.isOpened() || !capB.isOpened())
    {
        cerr << "ERROR: Could not open cameras " << camAIndex << " and/or " << camBIndex << "\n";
        return 1;
    }

    // Try to set same resolution for both (not guaranteed)
    capA.set(CAP_PROP_FRAME_WIDTH,  640);
    capA.set(CAP_PROP_FRAME_HEIGHT, 480);
    capB.set(CAP_PROP_FRAME_WIDTH,  640);
    capB.set(CAP_PROP_FRAME_HEIGHT, 480);

    cout << "Press 'q' or ESC to quit.\n";
    cout << "Tip: If fusion fails, point both cameras at a textured scene with overlap.\n";

    Mat frameA, frameB, fused;
    while (true)
    {
        capA >> frameA;
        capB >> frameB;

        if (frameA.empty() || frameB.empty())
            break;

        bool ok = fuseTwoFrames(frameA, frameB, fused);

        imshow("Cam A", frameA);
        imshow("Cam B", frameB);

        if (ok)
            imshow("Fused mosaic", fused);
        else
        {
            // If homography fails, show side-by-side as a fallback
            Mat side(max(frameA.rows, frameB.rows), frameA.cols + frameB.cols, frameA.type(), Scalar::all(0));
            frameA.copyTo(side(Rect(0, 0, frameA.cols, frameA.rows)));
            frameB.copyTo(side(Rect(frameA.cols, 0, frameB.cols, frameB.rows)));
            imshow("Fused mosaic", side);

            putText(side, "Fusion failed (not enough matches / bad overlap).",
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,0,255), 2);
        }

        int key = waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
    }

    capA.release();
    capB.release();
    destroyAllWindows();
    return 0;
}
