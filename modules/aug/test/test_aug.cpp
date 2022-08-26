#include "test_precomp.hpp"

namespace opencv_test{ namespace{

class CV_Aug_RandomCropBaseTest : public cvtest::BaseTest{
public:
    RNG rng;

    CV_Aug_RandomCropBaseTest(uint64 seed=0){
        rng = RNG(seed);
    }

    void setSeed(uint64 seed){
        rng.state = seed;
    }

    void getRandomCropParams(int h, int w, int th, int tw, int* x, int* y){
        if(h+1 < th || w+1 < tw){
            CV_Error( Error::StsBadSize, "The cropped size is larger than the image size" );
        }
        if(h == th && w == tw){
            (*x) = 0;
            (*y) = 0;
            return;
        }

        (*x) = rng.uniform(0, w-tw+1);
        (*y) = rng.uniform(0, h-th+1);

    }

};

TEST(Aug_RandomCrop, no_padding){
    cout << "run test: no_padding" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);

    CV_Aug_RandomCropBaseTest test;

    int seed = 0;
    test.setSeed(seed);

    int th = 200;
    int tw = 200;
    int i;
    int j;
    test.getRandomCropParams(input.rows, input.cols, th, tw, &i, &j);
    cout << "crop area: (" << i << "," << j << "," << th << "," << tw << ")" << endl;

    string ref_path = findDataFile("aug/random_crop_test_0.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::setSeed(seed);
    cv::imgaug::RandomCrop aug(Size(tw, th));
    Mat out;
    aug.call(input, out);

    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_RandomCrop, padding){
    cout << "run test: padding" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);

    CV_Aug_RandomCropBaseTest test;

    int seed = 0;
    test.setSeed(seed);

    int th = 200;
    int tw = 200;
    int i;
    int j;
    Vec4d padding {10, 20, 30, 40};

    string ref_path = findDataFile("aug/random_crop_test_1.jpg");
    Mat ref = imread(ref_path);

    imgaug::setSeed(seed);
    cv::imgaug::RandomCrop aug(Size(tw, th), padding);
    Mat out;
    aug.call(input, out);
//    imshow("out", out);
//    waitKey(0);
    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_RandomFlip, diagonal){
    cout << "run test: random flip (diagonal)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("aug/random_flip_test_2.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::RandomFlip aug(0, 1);
    aug.call(input, out);
    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_Resize, basic){
    cout << "run test: resize (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("aug/resize_test_3.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::Resize aug(cv::Size(256, 128));
    aug.call(input, out);

    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_CenterCrop, basic){
    cout << "run test: center crop (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("aug/center_crop_test_4.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::CenterCrop aug(cv::Size(400, 300));
    aug.call(input, out);
    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_Pad, basic){
    cout << "run test: pad (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("aug/pad_test_5.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::Pad aug(Vec4i(10, 20, 30, 40), Scalar(0));
    aug.call(input, out);
    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_RandomResizedCrop, basic){
    cout << "run test: random resized crop (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    cv::Size size(1024, 512);
    uint64 seed = 10;
    cv::imgaug::setSeed(seed);

    string ref_path = findDataFile("aug/random_resized_crop_test_6.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::RandomResizedCrop aug(size);

    aug.call(input, out);
    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_RandomRotation, not_expand){
    cout << "run test: random rotation (not_expand)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    cv::Vec2d degrees(-10, 10);
    uint64 seed = 5;
    cv::imgaug::setSeed(seed);

    string ref_path = findDataFile("aug/random_rotation_test_7.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::RandomRotation aug(degrees);

    aug.call(input, out);

    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_GrayScale, basic){
    cout << "run test: gray scale (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("aug/gray_scale_test_8.jpg");
    Mat ref = imread(ref_path, IMREAD_GRAYSCALE);

    cv::imgaug::GrayScale aug;

    aug.call(input, out);

    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_GaussianBlur, basic){
    cout << "run test: gaussian blur (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("aug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("aug/gaussian_blur_test_9.jpg");
    Mat ref = imread(ref_path);
    cv::imgaug::setSeed(15);
    cv::imgaug::GaussianBlur aug(Size(5, 5));

    aug.call(input, out);
    Scalar diff = sum(out - ref);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


}}

