//
// Created by Chuyang Zhao on 2022/9/11.
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgaug.hpp>
#include <stdio.h>

using namespace cv;

namespace {
    /** @brief save annotation data into file for later use
     *
     * The first line an integer n represents the number of bounding boxes and labels.
     * Following n lines, each line contains five integers (x, y, w, h, l),
     * representing the x and y coordinates of the top-left corner of the bounding box,
     * width and height of the bounding box and class label of the object in the bounding box respectively.
     *
     * @param path save path of annotation file
     * @param bboxes annotated bounding boxes
     * @param labels annotated labels
     */
    void save_annotation(const String& path, std::vector<Rect>& bboxes, std::vector<int>& labels){
        assert(bboxes.size() == labels.size());
        unsigned n = bboxes.size();

        FILE* fp = fopen(path.c_str(), "wt");
        fprintf(fp, "%d\n", n);

        for(unsigned i = 0; i < n; i++){
            fprintf(fp, "%d %d %d %d %d\n", bboxes[i].x, bboxes[i].y, bboxes[i].width, bboxes[i].height, labels[i]);
        }

        fclose(fp);
    }

    void drawBoundingBoxes(Mat &img, std::vector<Rect> &bboxes) {
        for (cv::Rect bbox: bboxes) {
            cv::Point tl{bbox.x, bbox.y};
            cv::Point br{bbox.x + bbox.width, bbox.y + bbox.height};
            cv::rectangle(img, tl, br, cv::Scalar(0, 255, 0), 2);
        }
    }


    void gen_random_flip_reference(int flipCode = 0) {
        String path = samples::findFile("lena.jpg", IMREAD_COLOR);

        Mat src = imread(path);
        Mat dst;

        std::vector<Rect> bboxes{
                Rect{112, 40, 249, 343},
                Rect{61, 273, 113, 228}
        };

        std::vector<int> labels{1, 2};

        imgaug::det::RandomFlip aug(flipCode);
        aug.call(src, dst, bboxes, labels);

        Mat display;
        dst.copyTo(display);
        drawBoundingBoxes(display, bboxes);

        imshow("display", display);
        waitKey(0);

        save_annotation("/Users/bytedance/Desktop/det_random_flip_test_0.dat", bboxes, labels);
        imwrite("/Users/bytedance/Desktop/det_random_flip_test_0.jpg", dst);
    }


    void gen_resize_reference(const Size& size = Size(224, 224)) {
        String path = samples::findFile("lena.jpg", IMREAD_COLOR);

        Mat src = imread(path);
        Mat dst;

        std::vector<Rect> bboxes{
                Rect{112, 40, 249, 343},
                Rect{61, 273, 113, 228}
        };

        std::vector<int> labels{1, 2};

        imgaug::det::Resize aug(size);
        aug.call(src, dst, bboxes, labels);

        Mat display;
        dst.copyTo(display);
        drawBoundingBoxes(display, bboxes);

        imshow("display", display);
        waitKey(0);

        save_annotation("/Users/bytedance/Desktop/det_resize_test_0.dat", bboxes, labels);
        imwrite("/Users/bytedance/Desktop/det_resize_test_0.jpg", dst);
    }


    void gen_convert_reference(int code = COLOR_BGR2GRAY) {
        String path = samples::findFile("lena.jpg", IMREAD_COLOR);

        Mat src = imread(path);
        Mat dst;

        std::vector<Rect> bboxes{
                Rect{112, 40, 249, 343},
                Rect{61, 273, 113, 228}
        };

        std::vector<int> labels{1, 2};

        imgaug::det::Convert aug(code);
        aug.call(src, dst, bboxes, labels);

        Mat display;
        dst.copyTo(display);
        drawBoundingBoxes(display, bboxes);

        imshow("display", display);
        waitKey(0);

        save_annotation("/Users/bytedance/Desktop/det_convert_test_0.dat", bboxes, labels);
        imwrite("/Users/bytedance/Desktop/det_convert_test_0.jpg", dst);
    }


    void gen_random_translation_reference(Vec2d trans = Vec2d(20, 20)) {
        String path = samples::findFile("lena.jpg", IMREAD_COLOR);

        Mat src = imread(path);
        Mat dst;

        std::vector<Rect> bboxes{
                Rect{112, 40, 249, 343},
                Rect{61, 273, 113, 228}
        };

        std::vector<int> labels{1, 2};

        imgaug::det::RandomTranslation aug(trans);
        aug.call(src, dst, bboxes, labels);

        Mat display;
        dst.copyTo(display);
        drawBoundingBoxes(display, bboxes);

        imshow("display", display);
        waitKey(0);

        save_annotation("/Users/bytedance/Desktop/det_random_translation_test_0.dat", bboxes, labels);
        imwrite("/Users/bytedance/Desktop/det_random_translation_test_0.jpg", dst);
    }


    void gen_random_rotation_reference(Vec2d degrees = Vec2d(-30, 30)) {
        String path = samples::findFile("lena.jpg", IMREAD_COLOR);

        Mat src = imread(path);
        Mat dst;

        std::vector<Rect> bboxes{
                Rect{112, 40, 249, 343},
                Rect{61, 273, 113, 228}
        };

        std::vector<int> labels{1, 2};

        imgaug::det::RandomRotation aug(degrees);
        aug.call(src, dst, bboxes, labels);

        Mat display;
        dst.copyTo(display);
        drawBoundingBoxes(display, bboxes);

        imshow("display", display);
        waitKey(0);

        save_annotation("/Users/bytedance/Desktop/det_random_rotation_test_0.dat", bboxes, labels);
        imwrite("/Users/bytedance/Desktop/det_random_rotation_test_0.jpg", dst);
    }


}

int main(){
    int seed = 0;
    imgaug::setSeed(seed);

//    gen_random_flip_reference(0);
//    gen_resize_reference();
//    gen_convert_reference();
//    gen_random_translation_reference();
    gen_random_rotation_reference();
    return 0;
}