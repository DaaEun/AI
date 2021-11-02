#include <iostream>
#include <fstream>	// 파일 입출력 헤더파일
#include <cstdlib>
#include <ctime>
#include "module.h"
#include "train.h"
using namespace std;

// training , 즉 학습을 위한 class 생성
void train(Module* module, int layer, float input[][2], float* target_output, int input_num){

	// 파일 출력을 위해 ofstream 객체 생성
	ofstream floss("loss.txt");
	ofstream fresult("result.txt");
	floss << "##### Loss #####" << endl << endl;	// loss.txt에 데이터 쓰기
	fresult << "##### Result #####" << endl << endl << "epoch	input	output	result" << endl;	// result.txt에 데이터 쓰기
	
	floss << "Epoch	Loss	Accuracy" << endl;	// loss.txt에 데이터 쓰기

	clock_t start, end;	// start = 시작시간, end = 종료시간

	int epoch = 0;		// (=iteration) 실행횟수
	float acc_cnt = 0;	// (accuracy count) 맞힌 횟수 0 으로 초기화
	int result;			// 계산 값(=y)와 target의 비교 값
	float accuracy = 0;	// 정확도
	float error_sum;	// error 값들의 합 

	start = clock();	// 시간 측정 시작

	while (accuracy < 0.99) {	
		// 정확도 < 0.99 동안 while문 탈출X
		// 즉, acc_cnt != input_num 과 동일한 조건

		epoch++;
		fresult << epoch << endl;	// result.txt에 데이터 쓰기
		error_sum = 0;	// error값 0 으로 초기화
		acc_cnt = 0;	// 맞힌 횟수 0 으로 초기화

		for (int i = 0; i < input_num; i++) {

			// 계산 값을 y 변수로 지정
			// output의 개념 유사하게 많이 사용되었기 때문
			float* y = input[i];

			fresult << "	(" << input[i][0] << "," << input[i][1] << ")	";	// result.txt에 데이터 쓰기

			// forward pass compute -> propagation
			// propagation 진행
			for (int j = 0; j < layer; j++) {
				// j = 2n 일 때, module[j]에서 linear_propagation 적용 => net = X1*W1 + X2*W2 + ... + Xn*Wn
				// j = 2n+1 일 때, module[j]에서 sigmoid_propagation 적용 => result = 1 / (1 + exp(-net))
				y = module[j].propagation(module[j], y);
			}

			// result(!= target output) 도출
			result = y[0] < 0.5 ? 0 : 1;	//sigmoid function 기준 : 0.5
			fresult << y[0] << "	" << result << endl;	// result.txt에 데이터 쓰기

			// target과 result이 같으면 acc_cnt(= 맞힌 횟수) 1 증가
			// 그렇지 않으면 0 증가, 즉 acc_cnt 변화 없음
			acc_cnt += result == (int)(target_output[i]) ? 1 : 0;

			// Error = (target - y) 계산
			float* error = (float*)malloc(sizeof(float));
			error[0] = -(target_output[i] - y[0]);

			// E = 1/2(target - y)^2
			error_sum += (error[0] * error[0]) / 2;

			// backward pass compute -> backpropagation
			// backpropagation 진행
			for (int j = layer - 1; 0 <= j; j--) {
				// c (= learning rate) = 0.1 <- 지정(실수)
				// j = 2n+1 일 때, module[j]에서 sigmoid_backpropagation 적용 
				// => result's derivative = result * (1 - result) 과 delta = delta * result * (1 - result) 계산
				// j = 2n 일 때, module[j]에서 linear_backpropagation 적용 
				// => delta[i] = delta[j] * w[ji] 와 W = W + (-c * delta * input) 계산
				error = module[j].backpropagation(module[j], error, 0.1);
			}
		} // for 문 종료

		accuracy = acc_cnt / (float)input_num;
		cout << "Epoch: " << epoch << "	Loss: " << error_sum << "	Accuracy: " << accuracy << endl;
		floss << epoch << "	" << error_sum << "	" << accuracy << endl;	// loss.txt에 데이터 쓰기

	} // while 문 종료
	
	cout << "------------------------------------------------" << endl;

	end = clock();	// 시간 측정 종료
	// 총 실행 시간 출력
	cout << endl << "총 실행 시간 : " << (double)(end - start) << "ms" << endl;

	floss << endl << "##### END #####" << endl;		// loss.txt에 데이터 쓰기
	fresult << endl << "##### END #####" << endl;	// result.txt에 데이터 쓰기

	// ofstream 객체 반환
	floss.close();
	fresult.close();
}
