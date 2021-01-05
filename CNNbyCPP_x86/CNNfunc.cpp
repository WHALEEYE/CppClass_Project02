#include "CNNfunc.hpp"
string Lpcwstr2String(LPCWSTR lps) {
  int len = WideCharToMultiByte(CP_ACP, 0, lps, -1, NULL, 0, NULL, NULL);
  if (len <= 0) {
    return "";
  } else {
    char *dest = new char[len];
    WideCharToMultiByte(CP_ACP, 0, lps, -1, dest, len, NULL, NULL);
    dest[len - 1] = 0;
    string str(dest);
    delete[] dest;
    return str;
  }
}

string select_pic() {
  OPENFILENAME ofn;
  char szFile[300];

  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = NULL;
  ofn.lpstrFile = (LPWSTR)szFile;
  ofn.lpstrFile[0] = '\0';
  LPTSTR lpstrCustomFilter;
  DWORD nMaxCustFilter;
  ofn.nFilterIndex = 1;
  LPTSTR lpstrFile;
  ofn.nMaxFile = sizeof(szFile);
  ofn.lpstrFilter = L"ALL\0*.*\0Text\0*.TXT\0";
  ofn.lpstrFileTitle = NULL;
  ofn.nMaxFileTitle = 0;
  ofn.lpstrInitialDir = NULL;

  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

  string path_image = "";
  if (GetOpenFileName(&ofn)) {
    path_image = Lpcwstr2String(ofn.lpstrFile);
    return path_image;
  } else {
    return "";
  }
}

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
               const float weight[], const float bias[], const int stride) {
  float temp1[9];
  float temp2[9];
  int centerIdx;
  int kernelStart;
  float sum;
  for (int curOutChannel = 0; curOutChannel < fm_cns; curOutChannel++) {
    for (int curRow = 0; curRow < pic_size; curRow += stride) {
      for (int curCol = 0; curCol < pic_size; curCol += stride) {
        sum = bias[curOutChannel];
        for (int curInChannel = 0; curInChannel < pic_cns; curInChannel++) {
          kernelStart = curOutChannel * pic_cns * 9 + curInChannel * 9;
          for (int idx = 0; idx < 9; idx++) {
            temp2[idx] = weight[kernelStart + idx];
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
          sum += mult(temp1, temp2, 9, 0, 0);
        }
        fm[curOutChannel * fm_size * fm_size + curRow / stride * fm_size +
           curCol / stride] = (sum < 0.00f ? 0.00f : sum);
      }
    }
  }
}

void conv_relu_t(const float pic[], const int pic_size, const int pic_cns,
                 float fm[], const int fm_size, const int thread_outchannel,
                 const float weight[], const float bias[], const int stride) {
  float temp1[9];
  float temp2[9];
  int centerIdx;
  int kernelStart;
  float sum;
  for (int curOutChannel = thread_outchannel;
       curOutChannel < thread_outchannel + 4; curOutChannel++) {
    for (int curRow = 0; curRow < pic_size; curRow += stride) {
      for (int curCol = 0; curCol < pic_size; curCol += stride) {
        sum = bias[curOutChannel];
        for (int curInChannel = 0; curInChannel < pic_cns; curInChannel++) {
          kernelStart = curOutChannel * pic_cns * 9 + curInChannel * 9;
          for (int idx = 0; idx < 9; idx++) {
            temp2[idx] = weight[kernelStart + idx];
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
          sum += mult(temp1, temp2, 9, 0, 0);
        }
        fm[curOutChannel * fm_size * fm_size + curRow / stride * fm_size +
           curCol / stride] = (sum < 0.00f ? 0.00f : sum);
      }
    }
  }
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

void pooling_t(const float pic[], const int pic_size, const int thread_cns,
               float pic_pool[], const int pic_pool_size) {
  int startIdx;
  float max;
  for (int k = thread_cns; k < thread_cns + 4; k++) {
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
