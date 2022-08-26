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
    for(int i=0; i<5; i++) {
        std::string filename = "/Users/bytedance/Workspace/opencv/samples/data/lena.jpg";
        cv::Mat src = cv::imread(filename);
        cv::Mat dst;
//        cv::imgaug::RandomRotation aug(cv::Vec2d(-10, 10));

//        aug.call(src, dst);
        uint64 seed = 15;
        cv::imgaug::setSeed(seed);
        cv::imgaug::randomRotation(src, dst, cv::Vec2d(-20, 20));
        cv::imshow("lena_dst.png", dst);
        cv::waitKey(0);
//        cv::det::RandomFlip aug(-1, 0.5);
//
//        std::vector<cv::Rect> target{
//                cv::Rect{100, 200, 100, 200},
//        };
//        cv::Point pt1{target[0].x, target[0].y};
//        cv::Point pt2{target[0].x + target[0].width, target[0].y + target[0].height};
//        cv::Mat src_copy;
//        src.copyTo(src_copy);
//        cv::rectangle(src_copy, pt1, pt2, cv::Scalar(), 2);
//        cv::imshow("lena_src.png", src_copy);
//
//        aug.call(src, dst, target);
//
//        cv::Point pt3{target[0].x, target[0].y};
//        cv::Point pt4{target[0].x + target[0].width, target[0].y + target[0].height};
//        cv::rectangle(dst, pt3, pt4, cv::Scalar(), 2);
//        cv::imshow("lena_dst.png", dst);
//        cv::waitKey(0);
    }
    return 0;
}
