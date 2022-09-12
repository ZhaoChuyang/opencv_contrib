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

void draw_bboxes(cv::Mat& img, std::vector<cv::Rect> bboxes){
    for(cv::Rect bbox: bboxes){
        cv::Point tl {bbox.x, bbox.y};
        cv::Point br {bbox.x + bbox.width, bbox.y + bbox.height};
        cv::rectangle(img, tl, br, cv::Scalar(0, 255, 0), 2);
    }
}

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
                cv::Rect{112, 40, 249, 343},
                cv::Rect{61, 273, 113, 228}
        };

        std::vector<int> labels {1, 2};

        cv::Mat src_copy;
        src.copyTo(src_copy);

        draw_bboxes(src_copy, bboxes);
        cv::imshow("lena_src.png", src_copy);

        aug.call(src, dst, bboxes, labels);

        draw_bboxes(dst, bboxes);
        cv::imshow("lena_dst.png", dst);
        cv::waitKey(0);
    }
}

void imgaug_test(){
    cv::Mat src = imread(cv::samples::findFile("lena.jpg"), cv::IMREAD_COLOR);
    cv::Mat dst;
    cv::imgaug::RandomCrop randomCrop(cv::Size(300, 300));
    cv::imgaug::RandomFlip randomFlip(1);
    cv::imgaug::Resize resize(cv::Size(224, 224));
    std::vector<cv::Ptr<cv::imgaug::Transform> > transforms{&randomCrop, &randomFlip, &resize};
    cv::imgaug::Compose aug(transforms);
    aug.call(src, dst);

//    uint64 seed = 15;
//    cv::imgaug::setSeed(seed);
//    cv::imgaug::ColorJitter aug(cv::Vec2d(0, 2), cv::Vec2d(0, 2), cv::Vec2d(0, 2), cv::Vec2d(-0.5, 0.5));
//    aug.call(src, dst);
    cv::imshow("lena_dst.png", dst);
    cv::waitKey(0);
    cv::imwrite("/Users/bytedance/Desktop/compose_out.jpg", dst, {cv::IMWRITE_JPEG_QUALITY, 50});
}

static void onMouse(int event, int x, int y, int, void*){
    if( event != cv::EVENT_LBUTTONDOWN )
        return;

    std::cout << x << " " << y << std::endl;
}

void get_mouse_click_coordinates(){
    std::string filename = "/Users/bytedance/Workspace/opencv/samples/data/lena.jpg";
    cv::Mat src = cv::imread(filename);
    cv::Mat dst;
    cv::imshow("src", src);
    cv::setMouseCallback("src", onMouse, nullptr);
    cv::waitKey(0);
}

int main(int argv, char** argc){
//    get_mouse_click_coordinates();
//    det_test();
    using namespace cv;
    imgaug_test();
//    imgaug::RandomCrop randomCrop(cv::Size(300, 300));
//    Mat src = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);
//    Mat dst;
//    randomCrop.call(src, dst);

//    imwrite("/Users/bytedance/Desktop/lena.jpg", src, {cv::IMWRITE_JPEG_QUALITY, 50});

    return 0;
}
