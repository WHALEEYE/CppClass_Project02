#pragma once
#include <windows.h>

#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#define CONSTE 2.7183
using namespace cv;
using namespace std;

extern float conv0_weight[];
extern float conv0_bias[];
extern float conv1_weight[];
extern float conv1_bias[];
extern float conv2_weight[];
extern float conv2_bias[];
extern float fc0_weight[];
extern float fc0_bias[];
void conv_relu(const float pic[], const int pic_size, const int pic_cns,
               float fm[], const int fm_size, const int fm_cns,
               const float weight[], const float bias[], const int stride);
float mult(float v1[], float v2[], int dim, int st1, int st2);
void pooling(const float pic[], const int pic_size, const int cns,
             float pic_pool[], const int pic_pool_size);
void conv_relu_t(const float pic[], const int pic_size, const int pic_cns,
                 float fm[], const int fm_size, const int thread_outchannel,
                 const float weight[], const float bias[], const int stride);
void pooling_t(const float pic[], const int pic_size, const int thread_cns,
               float pic_pool[], const int pic_pool_size);
string Lpcwstr2String(LPCWSTR lps);
string select_pic();