#include <opencv2/opencv.hpp>
using namespace cv;

int main() {
  Mat image = imread(".\\samples\\bg.jpg");
  imshow("MyPicture", image);
  waitKey(0);
  return 0;
}