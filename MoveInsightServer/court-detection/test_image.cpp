#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "CourtLinePixelDetector.h"
#include "CourtLineCandidateDetector.h"  
#include "BadmintonCourtFitter.h"
#include "DebugHelpers.h"

using namespace cv;

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cout << "Usage: ./test_image image_path [output_path]" << std::endl;
    return -1;
  }
  
  std::string filename(argv[1]);
  std::cout << "Reading image " << filename << std::endl;
  
  Mat frame = imread(filename);
  if (frame.empty())
  {
    std::cerr << "Cannot open image " << filename << std::endl;
    return 1;
  }
  
  std::cout << "Image loaded successfully. Size: " << frame.cols << "x" << frame.rows << std::endl;

  CourtLinePixelDetector courtLinePixelDetector;
  CourtLineCandidateDetector courtLineCandidateDetector;
  BadmintonCourtFitter tennisCourtFitter;

  std::cout << "Starting court line detection algorithm..." << std::endl;
  try
  {
    Mat binaryImage = courtLinePixelDetector.run(frame);
    std::vector<Line> candidateLines = courtLineCandidateDetector.run(binaryImage, frame);
    BadmintonCourtModel model = tennisCourtFitter.run(candidateLines, binaryImage, frame);
    
    std::cout << "Court detection completed successfully!" << std::endl;
    
    if (argc == 2)
    {
      model.drawModel(frame);
      displayImage("Result - press key to exit", frame);
    }
    if (argc >= 3)
    {
      std::string outFilename(argv[2]);
      model.writeToFile(outFilename);
      std::cout << "Result written to " << outFilename << std::endl;
    }
  }
  catch (std::runtime_error& e)
  {
    std::cout << "Processing error: " << e.what() << std::endl;
    return 3;
  }

  return 0;
}