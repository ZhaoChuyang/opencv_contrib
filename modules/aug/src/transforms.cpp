#include "precomp.hpp"
#include <opencv2/highgui.hpp>

namespace cv{

    static void getRandomCropParams(int h, int w, int th, int tw, int* x, int* y);
    static void getRandomResizedCropParams(int height, int width, const Vec2d& scale, const Vec2d& ratio, Rect& rect);

    // NOTE: cv::randomCrop or randomCrop?
    void randomCrop(InputArray _src, OutputArray _dst, const Size& sz, const Vec4i& padding, bool pad_if_need, int fill, int padding_mode){
        // FIXME: whether the size of src should be (src.cols+left+right, src.rows+top+bottom)

        Mat src = _src.getMat();

        if(padding != Vec4i()){
            copyMakeBorder(src, src, padding[0], padding[1], padding[2], padding[3], padding_mode, fill);
        }

        // NOTE: make sure src.rows == src.size().height and src.cols = src.size().width
        // pad the height if needed
        if(pad_if_need && src.rows < sz.height){
            Vec4i _padding = {sz.height - src.rows, sz.height - src.rows, 0, 0};
            copyMakeBorder(src, src, _padding[0], _padding[1], _padding[2], _padding[3], padding_mode, fill);
        }
        // pad the width if needed
        if(pad_if_need && src.cols < sz.width){
            Vec4i _padding = {0, 0, sz.width - src.cols, sz.width - src.cols};
            copyMakeBorder(src, src, _padding[0], _padding[1], _padding[2], _padding[3], padding_mode, fill);
        }

        int x, y;
        getRandomCropParams(src.rows, src.cols, sz.height, sz.width, &x, &y);
        Mat RoI(src, Rect(x, y, sz.width, sz.height));
//        RoI.copyTo(dst);
        // inplace operation
        _dst.move(RoI);
    }

//    CV_EXPORTS_W void randomCropV1(InputOutputArray _src, const Size& sz, const Vec4i& padding, bool pad_if_need, int fill, int padding_mode){
//        Mat src = _src.getMat();
//
//        if(padding != Vec4i()){
//            copyMakeBorder(src, src, padding[0], padding[1], padding[2], padding[3], padding_mode, fill);
//        }
//
//        // NOTE: make sure src.rows == src.size().height and src.cols = src.size().width
//        // pad the height if needed
//        if(pad_if_need && src.rows < sz.height){
//            Vec4i _padding = {sz.height - src.rows, sz.height - src.rows, 0, 0};
//            copyMakeBorder(src, src, _padding[0], _padding[1], _padding[2], _padding[3], padding_mode, fill);
//        }
//        // pad the width if needed
//        if(pad_if_need && src.cols < sz.width){
//            Vec4i _padding = {0, 0, sz.width - src.cols, sz.width - src.cols};
//            copyMakeBorder(src, src, _padding[0], _padding[1], _padding[2], _padding[3], padding_mode, fill);
//        }
//
//        int x, y;
//        getRandomCropParams(src.rows, src.cols, sz.height, sz.width, &x, &y);
//        Mat cropped(src, Rect(x, y, sz.width, sz.height));
//        (*(Mat*)_src.getObj()) = cropped;
//    }

    static void getRandomCropParams(int h, int w, int th, int tw, int* x, int* y){
        if(h+1 < th || w+1 < tw){
            CV_Error( Error::StsBadSize, "The cropped size is larger than the image size" );
        }
        if(h == th && w == tw){
            (*x) = 0;
            (*y) = 0;
            return;
        }
//        time_t t;
//        srand((unsigned)time(&t));
//        (*x) = static_cast<int> (rand() / static_cast<float> (RAND_MAX) * (w-tw+1));
//        (*y) = static_cast<int> (rand()/ static_cast<float> (RAND_MAX) * (h-th+1));
        RNG rng = RNG(getTickCount());
        (*x) = rng.uniform(0, w-tw+1);
        (*y) = rng.uniform(0, h-th+1);

    }

    RandomCrop::RandomCrop(const Size& sz, const Vec4i& padding, bool pad_if_need, int fill, int padding_mode):
        sz (sz),
        padding (padding),
        pad_if_need (pad_if_need),
        fill (fill),
        padding_mode (padding_mode){};

    void RandomCrop::call(InputArray src, OutputArray dst) const{
        randomCrop(src, dst, sz, padding, pad_if_need, fill, padding_mode);
    }

    void randomFlip(InputArray _src, OutputArray _dst, int flipCode, double p){
        /*
         * flipCode:
         * 0 is vertical flip
         * 1 is horizontal flip
         * -1 is flip bott horizontally and vertically
         */

        // initialize RNG with seed of current tick count
        RNG rng = RNG(getTickCount());
        bool flag = rng.uniform(0., 1.) < p;

        Mat src = _src.getMat();
//        _dst.create(src.size(), src.type());
//        Mat dst = _dst.getMat();
        if(!flag){
//            src.copyTo(dst);
            _dst.move(src);
            return;
        }
        flip(src, src, flipCode);
        _dst.move(src);
    }

    RandomFlip::RandomFlip(int flipCode, double p):
        flipCode(flipCode),
        p(p){};

    void RandomFlip::call(InputArray src, OutputArray dst) const{
        randomFlip(src, dst);
    }

    Compose::Compose(std::vector<Ptr<Transform> >& transforms):
        transforms(transforms){};

    void Compose::call(InputArray _src, OutputArray _dst) const{
        Mat src = _src.getMat();

        for(auto it = transforms.begin(); it != transforms.end(); ++it){
            (*it)->call(src, src);
        }
        src.copyTo(_dst);
    }

    Resize::Resize(const Size& sz, int interpolation):
        sz(sz),
        interpolation(interpolation){};

    void Resize::call(InputArray src, OutputArray dst) const{
        resize(src, dst, sz, 0, 0, interpolation);
    }

    // size: (width, height)
    void centerCrop(InputArray _src, OutputArray _dst, const Size& size) {
        Mat src = _src.getMat();
        Mat padded(src);
        // pad the input image if needed
        if (size.width > src.cols || size.height > src.rows) {
            int top = size.height - src.rows > 0 ? static_cast<int>((size.height - src.rows) / 2) : 0;
            int bottom = size.height - src.rows > 0 ? static_cast<int>((size.height - src.rows) / 2) : 0;
            int left = size.width - src.cols > 0 ? static_cast<int>((size.width - src.cols) / 2) : 0;
            int right = size.width - src.cols > 0 ? static_cast<int>((size.width - src.cols) / 2) : 0;

            // fill with value 0
            copyMakeBorder(src, padded, top, bottom, left, right, BORDER_CONSTANT, 0);
        }

        int x = static_cast<int>((padded.cols - size.width) / 2);
        int y = static_cast<int>((padded.rows - size.height) / 2);
        
        Mat cropped(padded, Rect(x, y, size.width, size.height));
        _dst.move(cropped);
//        _dst.create(size, src.type());
//        Mat dst = _dst.getMat();
        // Ensure the size of the cropped image is the same as the size of the dst
//        CV_Assert(cropped.size() == dst.size() && cropped.type() == dst.type());
//        cropped.copyTo(dst);


    }

    CenterCrop::CenterCrop(const Size& size) :
        size(size) {};

    void CenterCrop::call(InputArray src, OutputArray dst) const {
        centerCrop(src, dst, size);
    }

    Pad::Pad(const Vec4i& padding, const Scalar& fill, int padding_mode) :
        padding(padding),
        fill(fill),
        padding_mode(padding_mode) {};

    void Pad::call(InputArray src, OutputArray dst) const {
        copyMakeBorder(src, dst, padding[0], padding[1], padding[2], padding[3], padding_mode, fill);
    }

    void randomResizedCrop(InputArray _src, OutputArray _dst, const Size& size, const Vec2d& scale, const Vec2d& ratio, int interpolation) {
        // Ensure scale range and ratio range are valid
        CV_Assert(scale[0] <= scale[1] && ratio[0] <= ratio[1]);

        Mat src = _src.getMat();

        Rect crop_rect;
        getRandomResizedCropParams(src.rows, src.cols, scale, ratio, crop_rect);
        Mat cropped(src, Rect(crop_rect));
        resize(cropped, _dst, size, 0.0, 0.0, interpolation);
    }

    static void getRandomResizedCropParams(int height, int width, const Vec2d& scale, const Vec2d& ratio, Rect& rect) {
        int area = height * width;

        // initialize random value generator
        RNG rng = RNG(getTickCount());

        for (int i = 0; i < 10; i++) {
            double target_area = rng.uniform(scale[0], scale[1]) * area;
            double aspect_ratio = rng.uniform(ratio[0], ratio[1]);

            int w = static_cast<int>(round(sqrt(target_area * aspect_ratio)));
            int h = static_cast<int>(round(sqrt(target_area / aspect_ratio)));

            if (w > 0 && w <= width && h > 0 && h <= height) {
                rect.x = rng.uniform(0, width - w + 1);
                rect.y = rng.uniform(0, height - h + 1);
                rect.width = w;
                rect.height = h;
                return;
            }
        }

        // Center Crop
        double in_ratio = static_cast<double>(width) / height;
        if (in_ratio < ratio[0]) {
            rect.width = width;
            rect.height = static_cast<int> (round(width / ratio[0]));
        }
        else if (in_ratio > ratio[1]) {
            rect.height = height;
            rect.width = static_cast<int> (round(height * ratio[1]));
        }
        else {
            rect.width = width;
            rect.height = height;
        }
        rect.x = (width - rect.width) / 2;
        rect.y = (height - rect.height) / 2;
        
    }

    RandomResizedCrop::RandomResizedCrop(const Size& size, const Vec2d& scale, const Vec2d& ratio, int interpolation) :
        size(size),
        scale(scale),
        ratio(ratio),
        interpolation(interpolation) {};

    void RandomResizedCrop::call(InputArray src, OutputArray dst) const{
        randomResizedCrop(src, dst, size, scale, ratio, interpolation);
    }

    void colorJitter(InputArray _src, OutputArray _dst, const Vec2d& brightness, const Vec2d& contrast, const Vec2d& saturation, const Vec2d& hue){
        // TODO: check input values
        RNG rng = RNG(getTickCount());

        Mat src = _src.getMat();

        double brightness_factor, contrast_factor, saturation_factor, hue_factor;

        if(brightness != Vec2d())
            brightness_factor = rng.uniform(brightness[0], brightness[1]);
        if(contrast != Vec2d())
            contrast_factor = rng.uniform(contrast[0], contrast[1]);
        if(saturation != Vec2d())
            saturation_factor = rng.uniform(saturation[0], saturation[1]);
        if(hue != Vec2d())
            hue_factor = rng.uniform(hue[0], hue[1]);

        int order[4] = {1,2,3,4};
        std::random_shuffle(order, order+4);

        for(int i : order){
            if(i == 1 && brightness != Vec2d())
                adjust_brightness(src, brightness_factor);
            if(i == 2 && contrast != Vec2d())
                adjust_contrast(src, contrast_factor);
            if(i == 3 && saturation != Vec2d())
                adjust_saturation(src, saturation_factor);
            if(i == 4 && hue != Vec2d())
                adjust_hue(src, hue_factor);
        }

        _dst.move(src);
    }

    ColorJitter::ColorJitter(const Vec2d &brightness, const Vec2d &contrast, const Vec2d &saturation,
                             const Vec2d &hue):
                             brightness(brightness),
                             contrast(contrast),
                             saturation(saturation),
                             hue(hue){};

    void ColorJitter::call(InputArray src, OutputArray dst) const{
        colorJitter(src, dst, brightness, contrast, saturation, hue);
    }

    void randomRotation(InputArray _src, OutputArray _dst, const Vec2d& degrees, int interpolation, bool expand, const Point2f& center, int fill){
        Mat src = _src.getMat();
        RNG rng = RNG(getTickCount());
        // TODO: check the validation of degrees
        double angle = rng.uniform(degrees[0], degrees[1]);

        Point2f pt(src.cols/2., src.rows/2.);
        if(center == Point2f()) pt = center;
        Mat r = getRotationMatrix2D(pt, angle, 1.0);
        // TODO: auto expand dst size to fit the rotated image
        warpAffine(src, _dst, r, Size(src.cols, src.rows), interpolation, BORDER_CONSTANT, fill);
    }

    RandomRotation::RandomRotation(const Vec2d& degrees, int interpolation, bool expand, const Point2f& center, int fill):
        degrees(degrees),
        interpolation(interpolation),
        expand(expand),
        center(center),
        fill(fill){};

    void RandomRotation::call(InputArray src, OutputArray dst) const{
        randomRotation(src, dst, degrees, interpolation, expand, center, fill);
    }

    void grayScale(InputArray _src, OutputArray _dst, int num_channels){
        Mat src = _src.getMat();
        cvtColor(src, src, COLOR_BGR2GRAY);

        if(num_channels == 1){
            _dst.move(src);
            return;
        }
        Mat channels[3] = {src, src, src};
        merge(channels, 3, _dst);
    }

    GrayScale::GrayScale(int num_channels):
        num_channels(num_channels){};

    void GrayScale::call(InputArray _src, OutputArray _dst) const{
        grayScale(_src, _dst, num_channels);
    }

    void randomGrayScale(InputArray _src, OutputArray _dst, double p){
        RNG rng = RNG(getTickCount());
        if(rng.uniform(0.0, 1.0) < p){
            grayScale(_src, _dst, _src.channels());
            return;
        }
        Mat src = _src.getMat();
        _dst.move(src);
    }

    RandomGrayScale::RandomGrayScale(double p):
        p(p){};

    void RandomGrayScale::call(InputArray src, OutputArray dst) const{
        randomGrayScale(src, dst);
    }

}
