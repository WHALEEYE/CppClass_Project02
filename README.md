# Simple_CNN_by_Cpp

The 2nd project (Simple CNN) of Sustech C++ class 2020 Fall.

## Introduction
In this project, I implemented a simple CNN model using C++.

The CNN model contains 3 layers of convolution, 3 layers of ReLU, and 2 layers of max pooling.

The three kinds of layers are implemented in two functions (the ReLU will be done during each layer of convolution).

This program can correctly output the given picture’s confidence score of background and its confidence score of face, which represents the possibilitiy that the picture doesn’t have a face and the possibilitiy that the picture has a face.

Notice that the given picture must be `128 * 128`.

## Platform support

This program have an ARM version (`cnn_by_cpp_arm.cpp`) and a x86 version (`cnn_by_cpp.cpp`).

![x86](https://github.com/WHALEEYE/CppClass_Project02/blob/master/screenshots/x86_version.png "x86 version")

![ARM](https://github.com/WHALEEYE/CppClass_Project02/blob/master/screenshots/ARM_version.png "ARM version")

## Files

The source codes is a solution of `Visual Studio 2019`. You can find the `Project02.sln` file and open it in Visual Studio 2019.

If you only want to get the source codes, you can get them in the folder `cnn_by_cpp_x86` (x86 version) or `cnn_by_cpp_arm` (ARM version).

## Select Pictures

In the `x86` version, you can choose the picture you want to test by a file selector.

![fileselector](https://github.com/WHALEEYE/CppClass_Project02/blob/master/screenshots/file_selector.png "File Selector")

## Optimize

I used multithreading in the 1st and 2nd layers of convolution to speed up the program.

This can cut down the time cost to about 50% of the original time cost.
