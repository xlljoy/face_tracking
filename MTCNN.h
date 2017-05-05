//
// Created by Young on 2016/11/27.
//

//#define CPU_ONLY

#ifndef MTCNN_MTCNN_H
#define MTCNN_MTCNN_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <math.h>


#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"



#include <caffe/util/db.hpp>

using namespace caffe;

class MTCNN {

public:

    MTCNN();
    MTCNN(const std::string &dir);
    MTCNN(const std::vector<std::string> model_file, const std::vector<std::string> trained_file);
    ~MTCNN();

    void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
    void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence);
    void detection_TEST(const cv::Mat& img, std::vector<cv::Rect>& rectangles);

    void Preprocess(const cv::Mat &img);
    void P_Net();
    void P_Net_test();
    void R_Net();
    void O_Net();
    void detect_net(int i);

    void local_NMS();
    void global_NMS();

    void timer_begin();
    void timer_end();
    void record(double num);

    void Predict(const cv::Mat& img, int i);
    void Predict(const std::vector<cv::Mat> imgs, int i);
    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i);
    void WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i);

    float IoU(cv::Rect rect1, cv::Rect rect2);
    float IoM(cv::Rect rect1, cv::Rect rect2);
    void resize_img();
    void generate_init_rectangles();
    void GenerateBoxs(cv::Mat img);
    void BoxRegress(std::vector<cv::Rect>& bounding_box, std::vector<cv::Rect> regression_box);
    void Padding(std::vector<cv::Rect>& bounding_box, int img_w,int img_h);
    cv::Mat crop(cv::Mat img, cv::Rect& rect);

    void img_show(cv::Mat img, std::string name);
    void img_show_T(cv::Mat img, std::string name);


    void feature_get();
    void featre_map2img();
    void create_face_order();
    void feature2file();
    vector<vector<float>> vector_copy(vector<vector<float>> a);
    float cosine_similarity(int dexis_last, int dexis_cur);
    float cosine_similarity(vector<float> a, vector<float> b);
    //param for P, R, O, L net
    std::vector<std::shared_ptr<Net<float>>> nets_;
    std::vector<cv::Size> input_geometry_;
    int num_channels_;



    //variable for the image
    cv::Mat img_;
    cv::Mat img_prev_;
    std::vector<cv::Mat> img_resized_;
    std::vector<cv::Mat> img_output_face_;
    std::vector<double> scale_;

    //variable for the output of the neural network
//    std::vector<cv::Rect> regression_box_;
    std::vector<float> regression_box_temp_;
    std::vector<cv::Rect> bounding_box_;
    std::vector<cv::Rect> bounding_box_restored_;
    std::vector<float> confidence_;
    std::vector<float> confidence_temp_;
    std::vector<std::vector<cv::Point>> alignment_;
    std::vector<float> alignment_temp_;
    std::vector<vector<float>> feature_map_;
    std::vector<vector<float>> feature_map_restored_;
    std::vector<vector<float>> feature_map_tmp_;
    std::vector<cv::Mat> fea_map_img_;
    std::vector<int> name_list_;
    std::vector<int> left_list_;
    std::vector<int> non_match_;
    std::vector<int> candidates_;
    int frame_index_;

    //paramter for the threshold
    int minSize_ = 21;
    float factor_ = 0.709;
    float similarity_threshold = 2.322;
    float threshold_[3] = {0.85, 0.98, 0.92};
    float threshold_NMS_ = 0.5;


    std::chrono::high_resolution_clock::time_point time_begin_, time_end_;
};


#endif //MTCNN_MTCNN_H
