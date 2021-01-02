#include <opencv2/opencv.hpp>
using namespace cv;

int main() {
  Mat image = imread("D:\\Documents\\Private\\Avatar\\Avatar01.jpg");
  imshow("MyPicture", image);
  waitKey(0);
  return 0;
}