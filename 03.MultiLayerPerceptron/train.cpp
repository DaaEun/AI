#include <iostream>
#include <fstream>	// ���� ����� �������
#include <cstdlib>
#include <ctime>
#include "module.h"
#include "train.h"
using namespace std;

// training , �� �н��� ���� class ����
void train(Module* module, int layer, float input[][2], float* target_output, int input_num){

	// ���� ����� ���� ofstream ��ü ����
	ofstream floss("loss.txt");
	ofstream fresult("result.txt");
	floss << "##### Loss #####" << endl << endl;	// loss.txt�� ������ ����
	fresult << "##### Result #####" << endl << endl << "epoch	input	output	result" << endl;	// result.txt�� ������ ����
	
	floss << "Epoch	Loss	Accuracy" << endl;	// loss.txt�� ������ ����

	clock_t start, end;	// start = ���۽ð�, end = ����ð�

	int epoch = 0;		// (=iteration) ����Ƚ��
	float acc_cnt = 0;	// (accuracy count) ���� Ƚ�� 0 ���� �ʱ�ȭ
	int result;			// ��� ��(=y)�� target�� �� ��
	float accuracy = 0;	// ��Ȯ��
	float error_sum;	// error ������ �� 

	start = clock();	// �ð� ���� ����

	while (accuracy < 0.99) {	
		// ��Ȯ�� < 0.99 ���� while�� Ż��X
		// ��, acc_cnt != input_num �� ������ ����

		epoch++;
		fresult << epoch << endl;	// result.txt�� ������ ����
		error_sum = 0;	// error�� 0 ���� �ʱ�ȭ
		acc_cnt = 0;	// ���� Ƚ�� 0 ���� �ʱ�ȭ

		for (int i = 0; i < input_num; i++) {

			// ��� ���� y ������ ����
			// output�� ���� �����ϰ� ���� ���Ǿ��� ����
			float* y = input[i];

			fresult << "	(" << input[i][0] << "," << input[i][1] << ")	";	// result.txt�� ������ ����

			// forward pass compute -> propagation
			// propagation ����
			for (int j = 0; j < layer; j++) {
				// j = 2n �� ��, module[j]���� linear_propagation ���� => net = X1*W1 + X2*W2 + ... + Xn*Wn
				// j = 2n+1 �� ��, module[j]���� sigmoid_propagation ���� => result = 1 / (1 + exp(-net))
				y = module[j].propagation(module[j], y);
			}

			// result(!= target output) ����
			result = y[0] < 0.5 ? 0 : 1;	//sigmoid function ���� : 0.5
			fresult << y[0] << "	" << result << endl;	// result.txt�� ������ ����

			// target�� result�� ������ acc_cnt(= ���� Ƚ��) 1 ����
			// �׷��� ������ 0 ����, �� acc_cnt ��ȭ ����
			acc_cnt += result == (int)(target_output[i]) ? 1 : 0;

			// Error = (target - y) ���
			float* error = (float*)malloc(sizeof(float));
			error[0] = -(target_output[i] - y[0]);

			// E = 1/2(target - y)^2
			error_sum += (error[0] * error[0]) / 2;

			// backward pass compute -> backpropagation
			// backpropagation ����
			for (int j = layer - 1; 0 <= j; j--) {
				// c (= learning rate) = 0.1 <- ����(�Ǽ�)
				// j = 2n+1 �� ��, module[j]���� sigmoid_backpropagation ���� 
				// => result's derivative = result * (1 - result) �� delta = delta * result * (1 - result) ���
				// j = 2n �� ��, module[j]���� linear_backpropagation ���� 
				// => delta[i] = delta[j] * w[ji] �� W = W + (-c * delta * input) ���
				error = module[j].backpropagation(module[j], error, 0.1);
			}
		} // for �� ����

		accuracy = acc_cnt / (float)input_num;
		cout << "Epoch: " << epoch << "	Loss: " << error_sum << "	Accuracy: " << accuracy << endl;
		floss << epoch << "	" << error_sum << "	" << accuracy << endl;	// loss.txt�� ������ ����

	} // while �� ����
	
	cout << "------------------------------------------------" << endl;

	end = clock();	// �ð� ���� ����
	// �� ���� �ð� ���
	cout << endl << "�� ���� �ð� : " << (double)(end - start) << "ms" << endl;

	floss << endl << "##### END #####" << endl;		// loss.txt�� ������ ����
	fresult << endl << "##### END #####" << endl;	// result.txt�� ������ ����

	// ofstream ��ü ��ȯ
	floss.close();
	fresult.close();
}
