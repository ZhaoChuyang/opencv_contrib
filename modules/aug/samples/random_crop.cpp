#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aug.hpp>
#include <iostream>
#include <vector>

//
//void test_pad_class(cv::Mat src, cv::Mat& dst) {
//    cv::Pad pad(cv::Vec4i(100, 100, 100, 100), 255);
//    pad.call(src, dst);
//}
//
//
//void test_randomFlip(cv::Mat src, cv::Mat& dst){
//    cv::randomFlip(src, dst);
//}

int main(int argv, char** argc) {
    std::string filename = "/Users/chuyang/Development/opencv/samples/data/lena.jpg";
    cv::Mat src = cv::imread(filename);
    cv::Mat dst;

    cv::randomAffine(src, dst, cv::Vec2f(10, 90), cv::Vec2f(0.3, 0.3), cv::Vec2f(1, 1), cv::Vec4f(5, 10, 0, 0));
    cv::imshow("lena.png", dst);
    cv::waitKey(0);

    return 0;
}
