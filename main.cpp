#include <iostream>
#include "MTCNN.h"

using namespace std;
using namespace cv;

int main() {

    vector<string> model_file = {
            "/home/xileli/Documents/program/MTCNN/model/det1.prototxt",
            "/home/xileli/Documents/program/MTCNN/model/det2.prototxt",
            "/home/xileli/Documents/program/MTCNN/model/det3.prototxt"
//            "../model/det4.prototxt"
    };

    vector<string> trained_file = {
            "/home/xileli/Documents/program/MTCNN/model/det1.caffemodel",
            "/home/xileli/Documents/program/MTCNN/model/det2.caffemodel",
            "/home/xileli/Documents/program/MTCNN/model/det3.caffemodel"
//            "../model/det4.caffemodel"
    };
    MTCNN mtcnn(model_file, trained_file);



/*
// image input starts
    vector<Rect> rectangles;
    string img_path = "/home/xileli/Desktop/three.jpeg";
    Mat img = imread(img_path);

    mtcnn.detection(img, rectangles);
//    string img_path_2 = "/home/xileli/Desktop/2.png";
//    img = imread(img_path_2);
//
//    mtcnn.detection(img, rectangles);
//    cout<< 1 <<endl;


    cv::imwrite("/home/xileli/Desktop/Emma1.jpg", mtcnn.fea_map_img_[0]);
    std::cout << "Hello, World!" << std::endl;

    for(int i = 0; i < rectangles.size(); i++) {
        rectangle(img, rectangles[i], Scalar(255, 0, 0), 4);
//        string txt = "face " + to_string(i);
//        Point bottom_left_pos(rectangles[i].x, rectangles[i].y);
//        putText(img, txt, bottom_left_pos, FONT_HERSHEY_SIMPLEX, 1.3, Scalar(255, 0, 255), 3);
    }

    imshow("face", img);
    waitKey(0);

*/





//video output time record----txt file

    string time_output = "/home/xileli/Documents/video/RossVideo/output/time_rec.txt";




//video input starts
//    VideoCapture cap(0);
//    VideoCapture cap("/home/xileli/Downloads/Omarosa and Bethenny Get Into It.mp4");
//    VideoCapture cap("/home/xileli/Documents/program/Xile_research/data_set/test/motinas_multi_face_frontal.avi");
    VideoCapture cap("/home/xileli/Documents/video/recognition/4.mp4");
//    VideoCapture cap("/home/xileli/Documents/video/recognition/2.mp4");
    //VideoCapture cap("/home/xileli/Downloads/llTz.avi");//1
    //VideoCapture cap("/home/xileli/Downloads/llTx.avi");//2
    //VideoCapture cap("/home/xileli/Downloads/llRy.avi");//3
    //VideoCapture cap("/home/xileli/Downloads/jam1.avi");//4
    //VideoCapture cap("/home/xileli/Downloads/jam2.avi");//5
    //VideoCapture cap("/home/xileli/Downloads/llm4.avi");//6
    //VideoCapture cap("/home/xileli/Downloads/llm3r.avi");//7
//    VideoCapture cap("/home/xileli/Downloads/rrm2.avi");//8
//    VideoCapture cap("/home/xileli/Downloads/rrm1.avi");//9
//    VideoCapture cap("/home/xileli/Downloads/ssm5.avi");//10


    Mat img_scr;
    cap >> img_scr;
    VideoWriter out_video;
    out_video.open("/home/xileli/Documents/video/RossVideo/output/output1.avi",
                   CV_FOURCC('M', 'J', 'P', 'G'), 30, img_scr.size());
    int index = 0;

    if (!cap.isOpened())// check if we succeeded
    return -1;

    for(;;)
    {
        Mat img;
        cap >> img; // get a new frame from camera/video

        vector<Rect> rectangles;

        mtcnn.frame_index_ = index;


        mtcnn.timer_begin();
        mtcnn.detection(img, rectangles);
        mtcnn.timer_end();
        vector<int> name_list = mtcnn.name_list_;
        std::fstream file("/home/xileli/Documents/video/RossVideo/output/time_test.txt", ios::app);// record the face num of current frame
        //std::cout << rectangles.size() << std::endl;
        file << rectangles.size() << std::endl;
        file.close();

        for(int i = 0; i < rectangles.size(); i++)
        {
            /*
            //crop the faces
            string pic_name = to_string(name_list[i]);
            string forlder_root = "/home/xileli/Documents/program/Xile_research/data_set/student/";
            string forlder_name = forlder_root + pic_name + "/" + to_string(index) + ".jpg";
            cv::Mat img_o = img(rectangles[i]);
           // cv::imwrite(forlder_name, img_o);//write down each face image
            //cv::imwrite(forlder_name, mtcnn.fea_map_img_[i]); //write down each face feature map
             */


            //draw rectangles
            rectangle(img, rectangles[i], Scalar(255, 0, 0), 4);


            //label face order
            string txt = "face " + to_string(name_list[i]);
            Point bottom_left_pos(rectangles[i].x, rectangles[i].y);
            putText(img, txt, bottom_left_pos, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 255), 3);


        }

        imshow("face", img);
        out_video<<img;
        waitKey(1);

//        cv::imwrite("/home/xileli/Documents/program/Xile_research/data_set/student/random/" + to_string(index) + ".jpg",img);
        //cv::imwrite("/home/xileli/Documents/program/Xile_research/result/clips_2/" + to_string(index) + ".jpg",img);
        cout << index << endl;
        index++;
//        time_rec<<
//        if(waitKey(30) >= 0) break;
    }


    return 0;
}

