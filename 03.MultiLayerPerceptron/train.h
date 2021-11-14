#pragma once

#include "module.h"

// training , 즉 학습을 위한 class 생성
void train(Module* module, int layer, float input[][2], float* target_output, int input_num);