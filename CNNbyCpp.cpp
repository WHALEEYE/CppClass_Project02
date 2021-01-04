#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#define CONSTE 2.7183
#include <opencv2/opencv.hpp>
extern float conv0_weight[];
extern float conv0_bias[];
extern float conv1_weight[];
extern float conv1_bias[];
extern float conv2_weight[];
extern float conv2_bias[];
extern float fc0_weight[];
extern float fc0_bias[];

using namespace cv;
using namespace std;

float mult(float v1[], float v2[], int dim, int st1, int st2) {
  int vec1, vec2;
  long long sum = 0;
  for (int i = 0; i < dim; i++) {
    vec1 = (int)(v1[st1 + i] * 1e3);
    vec2 = (int)(v2[st2 + i] * 1e3);
    sum += (long long)vec1 * vec2;
  }
  return sum / (float)1e6;
}

void conv_relu(const float pic[], const int pic_size, const int pic_cns,
               float fm[], const int fm_size, const int fm_cns,
               const float weight[], const float bias[], const int stride,
               const int kernel_ele = 9, const int pad = 1) {
  float *temp1 = new float[kernel_ele];
  float *temp2 = new float[kernel_ele];
  int centerIdx;
  int kernelStart;
  float sum;
  for (int curOutChannel = 0; curOutChannel < fm_cns; curOutChannel++) {
    for (int curRow = 0; curRow < pic_size; curRow += stride) {
      for (int curCol = 0; curCol < pic_size; curCol += stride) {
        sum = bias[curOutChannel];
        for (int curInChannel = 0; curInChannel < pic_cns; curInChannel++) {
          kernelStart =
              curOutChannel * pic_cns * kernel_ele + curInChannel * kernel_ele;
          for (int idx = 0; idx < kernel_ele; idx++) {
            temp2[idx] = conv0_weight[kernelStart + idx];
          }
          centerIdx =
              curInChannel * pic_size * pic_size + curRow * pic_size + curCol;
          temp1[0] = (curCol == 0 || curRow == 0)
                         ? 0.00f
                         : pic[centerIdx - pic_size - 1];
          temp1[1] = (curRow == 0) ? 0.00f : pic[centerIdx - pic_size];
          temp1[2] = (curRow == 0 || curCol == (pic_size - 1))
                         ? 0.00f
                         : pic[centerIdx - pic_size + 1];
          temp1[3] = (curCol == 0) ? 0.00f : pic[centerIdx - 1];
          temp1[4] = pic[centerIdx];
          temp1[5] = (curCol == (pic_size - 1)) ? 0.00f : pic[centerIdx + 1];
          temp1[6] = (curCol == 0 || curRow == (pic_size - 1))
                         ? 0.00f
                         : pic[centerIdx + pic_size - 1];
          temp1[7] =
              curRow == (pic_size - 1) ? 0.00f : pic[centerIdx + pic_size];
          temp1[8] = (curRow == (pic_size - 1) || curCol == (pic_size - 1))
                         ? 0.00f
                         : pic[centerIdx + pic_size + 1];
          sum += mult(temp1, temp2, kernel_ele, 0, 0);
        }
         fm[curOutChannel * fm_size * fm_size + curRow / stride * fm_size +
           curCol / stride] = (sum < 0.00f ? 0.00f : sum);
        //fm[curOutChannel * fm_size * fm_size + curRow / stride * fm_size +
        //   curCol / stride] = sum;
      }
    }
  }
  delete[] temp1;
  delete[] temp2;
}

void pooling(const float pic[], const int pic_size, const int cns,
             float pic_pool[], const int pic_pool_size) {
  int startIdx;
  float max;
  for (int k = 0; k < cns; k++) {
    for (int i = 0; i < pic_size; i += 2) {
      for (int j = 0; j < pic_size; j += 2) {
        startIdx = k * pic_size * pic_size + i * pic_size + j;
        max = 0;
        for (int idx = 0; idx < 4; idx++) {
          max = pic[startIdx + idx] > max ? pic[startIdx + idx] : max;
        }
        pic_pool[k * pic_pool_size * pic_pool_size + i / 2 * pic_pool_size +
                 j / 2] = max;
      }
    }
  }
}

void printMatrix(const float mtx[], int size, int channels, ofstream &ofs) {
  ofs.setf(ios_base::fixed, ios_base::floatfield);
  for (int i = 0; i < channels; i++) {
    ofs << "Channel " << i + 1 << " :\n";
    for (int k = 0; k < size; k++) {
      for (int j = 0; j < size; j++) {
        ofs << setprecision(6) << mtx[i * size * size + k * size + j] << " ";
      }
      ofs << endl;
    }
  }
}

#define KERNEL_ELE 9
#define IMG0_SIZE 128
#define IMG0_CNS 3
#define IMG1_SIZE 64
#define IMG1_CNS 16
#define IMG1_POOL_SIZE 32
#define IMG2_SIZE 32
#define IMG2_CNS 32
#define IMG2_POOL_SIZE 16
#define IMG3_SIZE 8
#define IMG3_CNS 32

int main() {
  Mat image = imread(".\\samples\\face01.jpg");

  float img0[IMG0_SIZE * IMG0_SIZE * IMG0_CNS];
  for (int curChannel = 0; curChannel < IMG0_CNS; curChannel++) {
    for (int curRow = 0; curRow < IMG0_SIZE; curRow++) {
      for (int curCol = 0; curCol < IMG0_SIZE; curCol++) {
        img0[curChannel * IMG0_SIZE * IMG0_SIZE + curRow * IMG0_SIZE + curCol] =
            image.at<Vec3b>(curRow, curCol)[IMG0_CNS - 1 - curChannel] /
            (float)255;
      }
    }
  }

  float img1[IMG1_SIZE * IMG1_SIZE * IMG1_CNS];
  conv_relu(img0, IMG0_SIZE, IMG0_CNS, img1, IMG1_SIZE, IMG1_CNS, conv0_weight,
            conv0_bias, 2);

  float img1_pool[IMG1_POOL_SIZE * IMG1_POOL_SIZE * IMG1_CNS];
  pooling(img1, IMG1_SIZE, IMG1_CNS, img1_pool, IMG1_POOL_SIZE);

  float img2[IMG2_SIZE * IMG2_SIZE * IMG2_CNS];
  conv_relu(img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2, IMG2_SIZE, IMG2_CNS,
            conv1_weight, conv1_bias, 1);

  float img2_pool[IMG2_POOL_SIZE * IMG2_POOL_SIZE * IMG2_CNS];
  pooling(img2, IMG2_SIZE, IMG2_CNS, img2_pool, IMG2_POOL_SIZE);

  float img3[IMG3_SIZE * IMG3_SIZE * IMG3_CNS];
  conv_relu(img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3, IMG3_SIZE, IMG3_CNS,
            conv2_weight, conv2_bias, 2);

  //ofstream ofs(".\\out.txt");
  //printMatrix(img1, IMG1_SIZE, IMG1_CNS, ofs);
  //printMatrix(img3, IMG3_SIZE, IMG3_CNS, ofs);

   float bg_pow = mult(img3, fc0_weight, 2048, 0, 0) + fc0_bias[0];
   float face_pow = mult(img3, fc0_weight, 2048, 0, 2048) + fc0_bias[1];
   double bg_tensor = pow(CONSTE, bg_pow);
   double face_tensor = pow(CONSTE, face_pow);
   double total_tensor = bg_tensor + face_tensor;
   double bg_score = bg_tensor / total_tensor;
   double face_score = face_tensor / total_tensor;
   cout << "background score: " << bg_score << "\nface score: " << face_score;
  return 0;
}
