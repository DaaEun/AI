// ����#2 ��ǻ�Ͱ��к� 2017920036 �����

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "module.h"
using namespace std;

int main() {

	float input[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };	// X1�� X2
	float and_output[4] = { 0, 0, 0, 1 };	// AND ���꿡 ���� output
	float or_output[4] = { 0, 1, 1, 1 };	// OR ���꿡 ���� output
	float xor_output[4] = { 0, 1, 1, 0 };	// XOR ���꿡 ���� output
	float* target_output = NULL;

	int layer = 2;	// 2���� <- ����

	// ������ module ���� �� �ʱ�ȭ
	Module* module = (Module*)malloc(sizeof(Module) * layer);
	int in_channels = 2;	// �Է� ä�� �� = 2 <- ����
	int out_channels = 1;	// ��� ä�� �� = 1 <- ����

	module[0] = init_linear(in_channels, out_channels);
	module[1] = init_sigmoid(in_channels);

	//// layer = 2 ���� -> �Ʒ��ڵ� �ּ� => n������ ��� �ڵ� �ּ� ����
	//for (int i = 0; i < layer; i += 2) {
	//	cout << "in_channels, out_channels: " << endl;
	//	cin >> in_channels;
	//	cin >> out_channels;

	//	model[i] = init_linear(in_channels, out_channels);
	//	model[i + 1] = init_sigmoid(out_channels);
	//}

	int gate;	// ������ ����(gate) ������ ���� ���� ����

	// ������ gate ����(1. AND    2. OR    3. XOR)
	cout << endl << "GATE ���� : 1. AND    2. OR    3. XOR" << endl << "Select : ";
	while (1) {
		cin >> gate;
		if (gate != 1 && gate != 2 && gate != 3) {	// �������� gate �Է��ϸ� �߻��� ���� ����
			cout << "1 / 2 / 3 �� �����ϱ�" << endl;
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
	int iter = 0;		// (iteration) ����Ƚ��
	float acc_cnt = 0;	// (accuracy count) ���� Ƚ�� 0 ���� �ʱ�ȭ
	int result;			// ��� ���� ���� ��� ��
	float error_sum;

	while (acc_cnt != 4) {
		iter++;
		error_sum = 0;	// error �� 0 �ʱ�ȭ
		acc_cnt = 0;	// ���� Ƚ�� 0 �ʱ�ȭ

		for (int i = 0; i < 4; i++) {
			
			float* output = input[i];

			// forward pass compute -> propagation
			// propagation ����
			for (int j = 0; j < layer; j++) {
				// module[0]�� ��, linear_propagation ���� => net = X1*W1 + X2*W2 + theta
				// module[1]�� ��, sigmoid_propagation ���� => result = 1 / (1 + exp(-net))
				// out_channels = 1 �̹Ƿ� output[0] = result 
				output = module[j].propagation(module[j], output);
			}

			// result(!= target output) ����
			result = output[0] < 0.5 ? 0 : 1;	//sigmoid function ���� : 0.5
			//cout << "i : " << i << "	output : " << output[0] << "	result : " << result << endl;

			if (result != (int)target_output[i]) {
				float* error = (float*)malloc(sizeof(float));

				// error = (t - o)
				error[0] = target_output[i] - output[0];
				//cout << "i : " << i << "	error : " << error[0] << endl;

				// E = 1/2(t - o)^2
				error_sum += (error[0] * error[0])/2;
				 
				// backward pass compute -> backpropagation
				// backpropagation ����
				for (int j = layer - 1; 0 <= j; j--) {
					// c (= learning rate) = 0.01 <- ����(�Ǽ�)
					// module[1]�� ��, sigmoid_backpropagation ���� => derivative = error * (1 - error)
					// module[0]�� ��, linear_backpropagation ���� => new_loss = error * input, W = W + new_loss * c
					error = module[j].backpropagation(module[j], error, 0.01);
				}
			} // if�� ����
			else {
				acc_cnt++;
			}

		} // for�� ����

		cout << "Iter: " << iter << "	Loss: " << error_sum <<
			"	Accuracy: " << acc_cnt / (float)4 << endl;

	} // while�� ����

	return 0;
}