#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "TimeMeasurement.h"
#include "Line.h"
#include "CourtLinePixelDetector.h"
#include "CourtLineCandidateDetector.h"
#include "BadmintonCourtFitter.h"
#include "DebugHelpers.h"

using namespace cv;

int main(int argc, char** argv)
{
  TimeMeasurement::debug = false;
  CourtLinePixelDetector::debug = false;
  CourtLineCandidateDetector::debug = false;
  BadmintonCourtFitter::debug = false;

  if (argc < 2 || argc > 4)
  {
    std::cout << "Usage: ./detect video_path [output_path]" << std::endl;
    std::cout << "       video_path:  path to an input avi file." << std::endl;
    std::cout << "       output_path: path to an output file where the xy court point coordinates will be written." << std::endl;
    std::cout << "                    This argument is optional. If not present, then a window with the result will be opened." << std::endl;
    std::cout << "       output_image_path: path to an output file where the image will be written." << std::endl;
    std::cout << "                    This argument is optional. If not present, then a window with the result will be opened." << std::endl;
    return -1;
  }
  std::string filename(argv[1]);

  std::cout << "Reading file " << filename << std::endl;
  VideoCapture vc(filename);
  if (!vc.isOpened())
  {
    std::cerr << "Cannot open file " << filename << std::endl;
    return 1;
  }
  printVideoInfo(vc);
  Mat frame;
  int frameIndex = int(vc.get(CAP_PROP_FRAME_COUNT)) / 2;
  vc.set(CAP_PROP_POS_FRAMES, frameIndex);
  if (!vc.read(frame))
  {
    std::cerr << "Failed to read frame with index " << frameIndex << std::endl;
    return 2;
  }
  std::cout << "Reading frame with index " << frameIndex << std::endl;

  CourtLinePixelDetector courtLinePixelDetector;
  CourtLineCandidateDetector courtLineCandidateDetector;
  BadmintonCourtFitter tennisCourtFitter;

  std::cout << "Starting court line detection algorithm..." << std::endl;
  try
  {
    TimeMeasurement::start("LineDetection");
    Mat binaryImage = courtLinePixelDetector.run(frame);
    std::vector<Line> candidateLines = courtLineCandidateDetector.run(binaryImage, frame);
    BadmintonCourtModel model = tennisCourtFitter.run(candidateLines, binaryImage, frame);
    int elapsed_seconds = TimeMeasurement::stop("LineDetection");
    std::cout << "Elapsed time: " << elapsed_seconds << "s." << std::endl;
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
    if (argc >= 4)
    {
      std::string outFilename(argv[3]);
      model.drawModel(frame);
      writeImage(outFilename, frame);
    }

  }
  catch (std::runtime_error& e)
  {
    std::cout << "Processing error: " << e.what() << std::endl;
    return 3;
  }


  return 0;
}
