/*
  * Copyright 2020 NXP
  * SPDX-License-Identifier:     BSD-3-Clause
*/

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <unordered_map>

#include <fcntl.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


#define LOG(x) std::cerr
using namespace cv;

void showImage(float *output_tensor,int num)
{
  float *inverse_tensor=(float*)malloc(28*28 * sizeof(float));
  int rows = 28;
  int cols = 28;
  for(int i=0; i<rows*cols;++i){
    inverse_tensor[i]= 0.5 * output_tensor[i]+0.5 ;
  }
  Mat output_mat(rows, cols, CV_32FC1, inverse_tensor);
  output_mat.convertTo(output_mat, CV_8UC3, 255.0); 
  resize(output_mat, output_mat, Size(112, 112));
  imshow("result", output_mat);
  waitKey(200);
  std::ostringstream name;
  name << "result_im_" << num << ".png";
  imwrite(name.str(), output_mat);
}


void init_interpreter()
{
  // Load the model
  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile("dcgan_generator.tflite");
  if (!model)
  {
    LOG(FATAL) << "\nFailed to load model " << "\n";
    exit(-1);
  }

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder(&interpreter);
  if (!interpreter)
  {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  // Allocate tensor buffers.
  interpreter->AllocateTensors();
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  // Check interpreter state
  //tflite::PrintInterpreterState(interpreter.get());

  for(int num=0; num<10;++num){
       // Create vectors filled with noise
      std::vector<float> tensor(100);
      for (int i = 0; i < 100; i++)
      {
        tensor[i] = ((float) rand() / (RAND_MAX));
      }

       // Fill input buffers
      int input = interpreter->inputs()[0];      // input dims are (1,100)
      float* input_data_ptr = interpreter->typed_tensor<float>(input);
      for (int i = 0; i < 100; ++i) 
      {
              *(input_data_ptr) = (float)tensor[i];
              input_data_ptr++;
      }
      
      //Run inference on the model
      if (interpreter->Invoke() != kTfLiteOk)
      {
        LOG(FATAL) << "\nFailed to invoke " << "\n";
      }

      // Read output buffers and show the image generated
      int output = interpreter->outputs()[0];
      float *output_tensor = interpreter->typed_tensor<float>(output);
      auto output_size = 28*28;
      showImage(output_tensor,num);
  }
}

int main(int argc, char **argv)
{
  init_interpreter();
  return 0;
}
