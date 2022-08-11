#ifdef HAVE_OPENCV_AUG
typedef std::vector<cv::Ptr<cv::Transform> > vector_Ptr_Transform;
typedef std::vector<cv::Ptr<cv::det::Transform> > vector_Ptr_det_Transform;

//template<>
//bool pyopencv_to(PyObject *o, std::vector<Ptr<cv::Transform> > &value, const ArgInfo& info){
//    return pyopencv_to_generic_vec(o, value, info);
//}
template<> struct pyopencvVecConverter<Ptr<cv::Transform> >
{
    static bool to(PyObject* obj, std::vector<cv::Ptr<cv::Transform> >& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

};

template<> struct pyopencvVecConverter<Ptr<cv::det::Transform> >
{
    static bool to(PyObject* obj, std::vector<cv::Ptr<cv::det::Transform> >& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

};

#endif