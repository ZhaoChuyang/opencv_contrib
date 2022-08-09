#include "precomp.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>


namespace cv{
    namespace det{
        RandomFlip::RandomFlip(int flipCode, float p):
        flipCode(flipCode), p(p)
        {
            if(!(flipCode == 0 || flipCode == 1 || flipCode == 2)){
                CV_Error(Error::Code::StsBadArg, "flipCode is invalid, must be 0 or 1 or 2");
            }
            if(p < 0 || p > 1){
                CV_Error(Error::Code::StsBadArg, "probability p must be between range 0 and 1");
            }
        };

        void RandomFlip::call(InputArray _src, OutputArray _dst, std::vector<cv::Rect>& target) const{
            RNG rng = RNG(getTickCount());
            bool flag = rng.uniform(0., 1.) < p;

            Mat src = _src.getMat();
            if(!flag){
                _dst.move(src);
                return;
            }

            flipBoundingBox(target, src.size());
            flip(src, src, flipCode);
            _dst.move(src);
        }

        void RandomFlip::flipBoundingBox(std::vector<cv::Rect>& target, const Size& size) const{
            for(Rect bbox: target){
                if(flipCode == 0){
                    bbox.y = size.height - bbox.y;
                }else if(flipCode > 0){
                    bbox.x = size.width - bbox.x;
                }else{
                    bbox.x = size.width - bbox.x;
                    bbox.y = size.height - bbox.y;
                }
            }
        }
    }
}