#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgaug.hpp>
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

void det_test() {
    for(int i=0; i<1; i++) {
        std::string filename = "/Users/bytedance/Workspace/opencv/samples/data/lena.jpg";
        cv::Mat src = cv::imread(filename);
        cv::Mat dst;
//        cv::imgaug::RandomRotation imgaug(cv::Vec2d(-10, 10));

//        imgaug.call(src, dst);
//        uint64 seed = 15;
//        cv::imgaug::setSeed(seed);
//        cv::imgaug::randomRotation(src, dst, cv::Vec2d(-20, 20));
//        cv::imgaug::Resize aug(cv::Size(300, 300));
//        aug.call(src, dst);
//        cv::imshow("lena_dst.png", dst);
//        cv::waitKey(0);
//        cv::det::RandomFlip imgaug(-1, 0.5);

        cv::imgaug::det::RandomTranslation aug(cv::Vec2d(30, 30));
//        cv::imgaug::det::Resize aug(cv::Size(200, 200));

        std::vector<cv::Rect> bboxes{
                cv::Rect{100, 200, 100, 200},
        };
        std::vector<int> labels {1};

        cv::Point pt1{bboxes[0].x, bboxes[0].y};
        cv::Point pt2{bboxes[0].x + bboxes[0].width, bboxes[0].y + bboxes[0].height};

        cv::Mat src_copy;
        src.copyTo(src_copy);

        cv::rectangle(src_copy, pt1, pt2, cv::Scalar(), 2);
        cv::imshow("lena_src.png", src_copy);

        aug.call(src, dst, bboxes, labels);

        cv::Point pt3{bboxes[0].x, bboxes[0].y};
        cv::Point pt4{bboxes[0].x + bboxes[0].width, bboxes[0].y + bboxes[0].height};
        cv::rectangle(dst, pt3, pt4, cv::Scalar(), 2);
        cv::imshow("lena_dst.png", dst);
        cv::waitKey(0);
    }
}

void imgaug_test(){
    std::string filename = "/Users/bytedance/Workspace/opencv/samples/data/lena.jpg";
    cv::Mat src = cv::imread(filename);
    cv::Mat dst;

    uint64 seed = 15;
    cv::imgaug::setSeed(seed);
    cv::imgaug::ColorJitter aug(cv::Vec2d(0, 2), cv::Vec2d(0, 2), cv::Vec2d(0, 2), cv::Vec2d(-0.5, 0.5));
    aug.call(src, dst);
//    cv::imshow("lena_dst.png", dst);
//    cv::waitKey(0);
}

int main(int argv, char** argc){
    imgaug_test();
    return 0;
}
