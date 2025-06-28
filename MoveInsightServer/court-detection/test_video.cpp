#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./test_video <video_file>" << std::endl;
        return -1;
    }
    
    std::string filename(argv[1]);
    std::cout << "Trying to open: " << filename << std::endl;
    
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cout << "ERROR: Cannot open video file " << filename << std::endl;
        std::cout << "OpenCV Build Info:" << std::endl;
        std::cout << cv::getBuildInformation() << std::endl;
        return -1;
    }
    
    std::cout << "Video opened successfully!" << std::endl;
    std::cout << "Frame count: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    std::cout << "Frame rate: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    
    return 0;
}