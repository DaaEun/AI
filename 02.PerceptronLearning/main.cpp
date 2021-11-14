// 과제#2 컴퓨터과학부 2017920036 양다은

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "module.h"
using namespace std;

int main() {

	float input[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };	// X1과 X2
	float and_output[4] = { 0, 0, 0, 1 };	// AND 연산에 대한 output
	float or_output[4] = { 0, 1, 1, 1 };	// OR 연산에 대한 output
	float xor_output[4] = { 0, 1, 1, 0 };	// XOR 연산에 대한 output
	float* target_output = NULL;

	int layer = 2;	// 2차원 <- 지정

	// 실행할 module 생성 및 초기화
	Module* module = (Module*)malloc(sizeof(Module) * layer);
	int in_channels = 2;	// 입력 채널 수 = 2 <- 지정
	int out_channels = 1;	// 출력 채널 수 = 1 <- 지정

	module[0] = init_linear(in_channels, out_channels);
	module[1] = init_sigmoid(in_channels);

	//// layer = 2 지정 -> 아래코드 주석 => n차원일 경우 코드 주석 해제
	//for (int i = 0; i < layer; i += 2) {
	//	cout << "in_channels, out_channels: " << endl;
	//	cin >> in_channels;
	//	cin >> out_channels;

	//	model[i] = init_linear(in_channels, out_channels);
	//	model[i + 1] = init_sigmoid(out_channels);
	//}

	int gate;	// 실행할 연산(gate) 선택을 위한 변수 선언

	// 실행할 gate 선택(1. AND    2. OR    3. XOR)
	cout << endl << "GATE 선택 : 1. AND    2. OR    3. XOR" << endl << "Select : ";
	while (1) {
		cin >> gate;
		if (gate != 1 && gate != 2 && gate != 3) {	// 부적절한 gate 입력하면 발생할 에러 차단
			cout << "1 / 2 / 3 중 선택하기" << endl;
		}
		else break;
	}

	cout <<"------------------------------------------------" << endl;
	if (gate == 1) {
		cout << "AND gate" << endl;
		target_output = and_output;
	}
	else if(gate == 2) {
		cout << "OR gate" << endl;
		target_output = or_output;
	}
	else {
		cout << "XOR gate" << endl;
		target_output = xor_output;
	}
	cout << "------------------------------------------------" << endl;


	// training
	int iter = 0;		// (iteration) 실행횟수
	float acc_cnt = 0;	// (accuracy count) 맞힌 횟수 0 으로 초기화
	int result;			// 계산 값에 대한 결과 값
	float error_sum;

	while (acc_cnt != 4) {
		iter++;
		error_sum = 0;	// error 값 0 초기화
		acc_cnt = 0;	// 맞힌 횟수 0 초기화

		for (int i = 0; i < 4; i++) {
			
			float* output = input[i];

			// forward pass compute -> propagation
			// propagation 진행
			for (int j = 0; j < layer; j++) {
				// module[0]일 때, linear_propagation 적용 => net = X1*W1 + X2*W2 + theta
				// module[1]일 때, sigmoid_propagation 적용 => result = 1 / (1 + exp(-net))
				// out_channels = 1 이므로 output[0] = result 
				output = module[j].propagation(module[j], output);
			}

			// result(!= target output) 도출
			result = output[0] < 0.5 ? 0 : 1;	//sigmoid function 기준 : 0.5
			//cout << "i : " << i << "	output : " << output[0] << "	result : " << result << endl;

			if (result != (int)target_output[i]) {
				float* error = (float*)malloc(sizeof(float));

				// error = (t - o)
				error[0] = target_output[i] - output[0];
				//cout << "i : " << i << "	error : " << error[0] << endl;

				// E = 1/2(t - o)^2
				error_sum += (error[0] * error[0])/2;
				 
				// backward pass compute -> backpropagation
				// backpropagation 진행
				for (int j = layer - 1; 0 <= j; j--) {
					// c (= learning rate) = 0.01 <- 지정(실수)
					// module[1]일 때, sigmoid_backpropagation 적용 => derivative = error * (1 - error)
					// module[0]일 때, linear_backpropagation 적용 => new_loss = error * input, W = W + new_loss * c
					error = module[j].backpropagation(module[j], error, 0.01);
				}
			} // if문 종료
			else {
				acc_cnt++;
			}

		} // for문 종료

		cout << "Iter: " << iter << "	Loss: " << error_sum <<
			"	Accuracy: " << acc_cnt / (float)4 << endl;

	} // while문 종료

	return 0;
}