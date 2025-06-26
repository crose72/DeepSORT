#include "deepsort.h"

DeepSORT::DeepSORT() : _metric(nullptr), _max_iou_distance((float)0.7), _max_age((int)30), _n_init((int)3), _tracks() {};

DeepSORT::DeepSORT(float *metric, float max_iou_distance, int max_age, int n_init) : _metric(metric), _max_iou_distance(max_iou_distance), _max_age(max_age), _n_init(n_init), _tracks() {};

DeepSORT::camera_update(cv::Mat frame, float video) // TODO video should not be float
{
    for (int i = 0; i < _tracks.size(); ++i)
    {
        //_track[i].camera_update(frame); // TODO Calls member from Track
    }
}
