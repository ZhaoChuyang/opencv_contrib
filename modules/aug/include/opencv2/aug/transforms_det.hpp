//
// Created by Chuyang Zhao on 2022/8/9.
//

#ifndef OPENCV_TRANSFORMS_DET_HPP
#define OPENCV_TRANSFORMS_DET_HPP

namespace cv{
    namespace det{
        class CV_EXPORTS_W Transform{
        public:
            CV_WRAP virtual void call(InputArray src, OutputArray dst, std::vector<cv::Rect>& target) const = 0;
            CV_WRAP virtual ~Transform() = default;
        };

        class CV_EXPORTS_W Compose{
        public:
            CV_WRAP explicit Compose(std::vector<Ptr<Transform> >& transforms);
            CV_WRAP void call(InputArray src, OutputArray dst, std::vector<cv::Rect>& target) const;

            std::vector<Ptr<Transform> > transforms;
        };

        class CV_EXPORTS_W RandomFlip{
        public:
            CV_WRAP RandomFlip(int flipCode=0, float p=0.5);
            CV_WRAP void call(InputArray src, OutputArray dst, std::vector<cv::Rect>& target) const;
            void flipBoundingBox(std::vector<cv::Rect>& target, const Size& size) const;

            int flipCode;
            float p;
        };
    }
}

#endif //OPENCV_TRANSFORMS_DET_HPP
