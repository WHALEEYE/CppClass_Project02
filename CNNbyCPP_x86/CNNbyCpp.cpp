#include "CNNfunc.hpp"

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
  string imgdir;
  Mat image;
  float *img0 = new float[IMG0_SIZE * IMG0_SIZE * IMG0_CNS];
  float *img1 = new float[IMG1_SIZE * IMG1_SIZE * IMG1_CNS];
  float *img1_pool = new float[IMG1_POOL_SIZE * IMG1_POOL_SIZE * IMG1_CNS];
  float *img2 = new float[IMG2_SIZE * IMG2_SIZE * IMG2_CNS];
  float *img2_pool = new float[IMG2_POOL_SIZE * IMG2_POOL_SIZE * IMG2_CNS];
  float *img3 = new float[IMG3_SIZE * IMG3_SIZE * IMG3_CNS];
  float bg_pow, face_pow, bg_tensor, face_tensor, bg_score, face_score,
      total_tensor;
  char flag;
  cout.setf(ios_base::fixed, ios_base::floatfield);
  while (true) {
    cout << "Please choose a picture that you want to test (must be 128 * "
            "128): ";
    imgdir = select_pic();
    cout << imgdir << endl;
    try {
      image = imread(imgdir);
      for (int curChannel = 0; curChannel < IMG0_CNS; curChannel++) {
        for (int curRow = 0; curRow < IMG0_SIZE; curRow++) {
          for (int curCol = 0; curCol < IMG0_SIZE; curCol++) {
            img0[curChannel * IMG0_SIZE * IMG0_SIZE + curRow * IMG0_SIZE +
                 curCol] =
                image.at<Vec3b>(curRow, curCol)[IMG0_CNS - 1 - curChannel] /
                (float)255;
          }
        }
      }
    } catch (exception) {
      cerr << "\nInvalid directory. Please enter again.\n";
      continue;
    }
    // std::chrono::steady_clock::time_point start =
    //    std::chrono::steady_clock::now();

    // The simple way of the 1st layer of convolution & ReLU //
    /*
    conv_relu(img0, IMG0_SIZE, IMG0_CNS, img1, IMG1_SIZE, IMG1_CNS,
              conv0_weight, conv0_bias, 2);
    */

    // The multithread way of the 1st layer of convolution & ReLU //
    thread conv_t1(conv_relu_t, img0, IMG0_SIZE, IMG0_CNS, img1, IMG1_SIZE, 0,
                   conv0_weight, conv0_bias, 2);
    thread conv_t2(conv_relu_t, img0, IMG0_SIZE, IMG0_CNS, img1, IMG1_SIZE, 4,
                   conv0_weight, conv0_bias, 2);
    thread conv_t3(conv_relu_t, img0, IMG0_SIZE, IMG0_CNS, img1, IMG1_SIZE, 8,
                   conv0_weight, conv0_bias, 2);
    conv_relu_t(img0, IMG0_SIZE, IMG0_CNS, img1, IMG1_SIZE, 12, conv0_weight,
                conv0_bias, 2);
    conv_t1.join();
    conv_t2.join();
    conv_t3.join();

    // The simple way of the 1st layer of pooling //
    pooling(img1, IMG1_SIZE, IMG1_CNS, img1_pool, IMG1_POOL_SIZE);

    // The multithread way of the 1st layer of pooling //
    /*
    thread pool_t1(pooling_t, img1, IMG1_SIZE, 0, img1_pool, IMG1_POOL_SIZE);
    thread pool_t2(pooling_t, img1, IMG1_SIZE, 4, img1_pool, IMG1_POOL_SIZE);
    thread pool_t3(pooling_t, img1, IMG1_SIZE, 8, img1_pool, IMG1_POOL_SIZE);
    pooling_t(img1, IMG1_SIZE, 12, img1_pool, IMG1_POOL_SIZE);
    pool_t1.join();
    pool_t2.join();
    pool_t3.join();
    */

    // The simple way of the 2nd layer of convolution & ReLU //
    /*
    conv_relu(img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2, IMG2_SIZE, IMG2_CNS,
              conv1_weight, conv1_bias, 1);
    */

    // The multithread way of the 2nd layer of convolution & ReLU //
    thread conv_t4(conv_relu_t, img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2,
                   IMG2_SIZE, 0, conv1_weight, conv1_bias, 1);
    thread conv_t5(conv_relu_t, img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2,
                   IMG2_SIZE, 4, conv1_weight, conv1_bias, 1);
    thread conv_t6(conv_relu_t, img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2,
                   IMG2_SIZE, 8, conv1_weight, conv1_bias, 1);
    thread conv_t7(conv_relu_t, img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2,
                   IMG2_SIZE, 12, conv1_weight, conv1_bias, 1);
    thread conv_t8(conv_relu_t, img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2,
                   IMG2_SIZE, 16, conv1_weight, conv1_bias, 1);
    thread conv_t9(conv_relu_t, img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2,
                   IMG2_SIZE, 20, conv1_weight, conv1_bias, 1);
    thread conv_t10(conv_relu_t, img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2,
                    IMG2_SIZE, 24, conv1_weight, conv1_bias, 1);
    conv_relu_t(img1_pool, IMG1_POOL_SIZE, IMG1_CNS, img2, IMG2_SIZE, 28,
                conv1_weight, conv1_bias, 1);
    conv_t4.join();
    conv_t5.join();
    conv_t6.join();
    conv_t7.join();
    conv_t8.join();
    conv_t9.join();
    conv_t10.join();

    // The simple way of the 2nd layer of pooling //
    pooling(img2, IMG2_SIZE, IMG2_CNS, img2_pool, IMG2_POOL_SIZE);

    // The multithread way of the 2nd layer of pooling //
    /*
    thread pool_t4(pooling_t, img2, IMG2_SIZE, 0, img2_pool, IMG2_POOL_SIZE);
    thread pool_t5(pooling_t, img2, IMG2_SIZE, 4, img2_pool, IMG2_POOL_SIZE);
    thread pool_t6(pooling_t, img2, IMG2_SIZE, 8, img2_pool, IMG2_POOL_SIZE);
    thread pool_t7(pooling_t, img2, IMG2_SIZE, 12, img2_pool, IMG2_POOL_SIZE);
    thread pool_t8(pooling_t, img2, IMG2_SIZE, 16, img2_pool, IMG2_POOL_SIZE);
    thread pool_t9(pooling_t, img2, IMG2_SIZE, 20, img2_pool, IMG2_POOL_SIZE);
    thread pool_t10(pooling_t, img2, IMG2_SIZE, 24, img2_pool, IMG2_POOL_SIZE);
    pooling_t(img2, IMG2_SIZE, 28, img2_pool, IMG2_POOL_SIZE);
    pool_t4.join();
    pool_t5.join();
    pool_t6.join();
    pool_t7.join();
    pool_t8.join();
    pool_t9.join();
    pool_t10.join();
    */

    // The simple way of the 3rd layer of convolution & ReLU //
    conv_relu(img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3, IMG3_SIZE, IMG3_CNS,
              conv2_weight, conv2_bias, 2);

    // The multithread way of the 3rd layer of convolution & ReLU //
    /*
    thread conv_t11(conv_relu_t, img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3,
                    IMG3_SIZE, 0, conv2_weight, conv2_bias, 2);
    thread conv_t12(conv_relu_t, img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3,
                    IMG3_SIZE, 4, conv2_weight, conv2_bias, 2);
    thread conv_t13(conv_relu_t, img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3,
                    IMG3_SIZE, 8, conv2_weight, conv2_bias, 2);
    thread conv_t14(conv_relu_t, img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3,
                    IMG3_SIZE, 12, conv2_weight, conv2_bias, 2);
    thread conv_t15(conv_relu_t, img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3,
                    IMG3_SIZE, 16, conv2_weight, conv2_bias, 2);
    thread conv_t16(conv_relu_t, img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3,
                    IMG3_SIZE, 20, conv2_weight, conv2_bias, 2);
    thread conv_t17(conv_relu_t, img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3,
                    IMG3_SIZE, 24, conv2_weight, conv2_bias, 2);
    conv_relu_t(img2_pool, IMG2_POOL_SIZE, IMG2_CNS, img3, IMG3_SIZE, 28,
                conv2_weight, conv2_bias, 2);
    conv_t11.join();
    conv_t12.join();
    conv_t13.join();
    conv_t14.join();
    conv_t15.join();
    conv_t16.join();
    conv_t17.join();
    */

    bg_pow = mult(img3, fc0_weight, 2048, 0, 0) + fc0_bias[0];
    face_pow = mult(img3, fc0_weight, 2048, 0, 2048) + fc0_bias[1];
    bg_tensor = pow(CONSTE, bg_pow);
    face_tensor = pow(CONSTE, face_pow);
    total_tensor = bg_tensor + face_tensor;
    bg_score = bg_tensor / total_tensor;
    face_score = face_tensor / total_tensor;

    cout << "The test result of " << imgdir
         << ":\n      background score: " << setprecision(6) << bg_score
         << "\n      face score: " << setprecision(6) << face_score << endl;

    // std::chrono::steady_clock::time_point end =
    //    std::chrono::steady_clock::now();
    // cout << "Calculation took "
    //     << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
    //            .count()
    //     << " milliseconds.";

    cout << "\nHave another picture to input? [y/n]";
    cin >> flag;
    if (flag == 'n' || flag == 'N') {
      break;
    }
  }
  delete[] img0;
  delete[] img1;
  delete[] img1_pool;
  delete[] img2;
  delete[] img2_pool;
  delete[] img3;
  return 0;
}