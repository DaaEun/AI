// forward pass compute -> propagation
// backward pass compute -> backpropagation

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "module.h"
using namespace std;

// Node class 구현
// 1. init_node : 초기 노드
Node init_node(int in_channels) {

	Node node;

	node.in_channels = in_channels;								// 입력 채널 수
	node.weight = (float*)malloc(sizeof(float) * in_channels);	// weight 값
	node.theta = (float*)malloc(sizeof(float) * in_channels);	// theta 값
	node.propagation = node_propagation;						// propagation 
	node.backpropagation = node_backpropagation;				// backpropagation

	srand((unsigned int)time(NULL));	// rand() 함수 사용을 위해 시간에 따라 난수표 초기화
	for (int i = 0; i < in_channels; i++) {
		node.weight[i] = (float)((rand() % 10000) * 0.0001f);
		node.theta[i] = (float)((rand() % 10000) * 0.0001f);
		//cout << "i: " << i << "	weight:" << node.weight[i] << "	theta: " << node.theta[i] << endl;
	}

	return node;
}


// 2. node_propagation : 노드 propagation 계산
float node_propagation(Node self, float* input) {

	float net = 0;

	// net = X1*W1 + X2*W2 + ... + Xn*Wn + theta
	for (int i = 0; i < self.in_channels; i++) {
		net += input[i] * self.weight[i] + self.theta[i];
	}

	return net;
}


// 3. node_backpropagation : 노드 backpropagation 계산
float* node_backpropagation(Node self, float* input, float loss, float c) {

	float* new_loss = (float*)malloc(sizeof(float) * self.in_channels);

	for (int i = 0; i < self.in_channels; i++) {
		// new_loss = error * input
		// W = W + new_loss * c
		// theta = theta + loss * c
		new_loss[i] = loss * input[i];
		self.weight[i] += new_loss[i] * c;
		self.theta[i] += loss * c;

		//cout << "i: " << i << "	weight:" << self.weight[i] << "	theta: " << self.theta[i] << endl;
	}

	return new_loss;
}


// Linear class 구현
// 1. init_linear : 초기 계산
Module init_linear(int in_channels, int out_channels) {

	Module linear;

	linear.in_channels = in_channels;
	linear.out_channels = out_channels;
	linear.input = (float*)malloc(sizeof(float) * in_channels);
	linear.nodes = (Node*)malloc(sizeof(Node) * out_channels);
	linear.propagation = linear_propatation;
	linear.backpropagation = linear_backpropagation;

	for (int i = 0; i < out_channels; i++) {
		linear.nodes[i] = init_node(in_channels);
	}

	return linear;
}


// 2. linear_propatation : linear 적용한 propagation 계산
float* linear_propatation(Module self, float* input) {

	Node node;

	float* result = (float*)malloc(sizeof(float) * self.out_channels);

	for (int i = 0; i < self.in_channels; i++) {
		self.input[i] = input[i];
	}

	for (int i = 0; i < self.out_channels; i++)
	{
		node = self.nodes[i];
		result[i] = node.propagation(node, input);
	}

	return result;
}


// 3. linear_backpropagation : linear 적용한 backpropagation 계산
float* linear_backpropagation(Module self, float* loss, float c) {

	float* temp;
	float* new_loss = (float*)calloc(self.in_channels, sizeof(float));

	for (int i = 0; i < self.out_channels; i++) {
		temp = self.nodes[i].backpropagation(self.nodes[i], self.input, loss[i], c);
		for (int j = 0; j < self.in_channels; j++) {
			new_loss[j] += temp[j];
		}
	}

	// new_loss += new_loss/(채널 입력 수)
	for (int i = 0; i < self.in_channels; i++) {
		new_loss[i] += new_loss[i] / self.in_channels;
	}

	return new_loss;
}


// Sigmoid class 구현
// 1. init_sigmoid : 초기 계산
Module init_sigmoid(int in_channels) {

	Module sigmoid;
	
	sigmoid.in_channels = in_channels;
	sigmoid.out_channels = in_channels;
	sigmoid.input = (float*)malloc(sizeof(float) * in_channels);
	sigmoid.propagation = sigmoid_propatation;
	sigmoid.backpropagation = sigmoid_backpropagation;

	return sigmoid;
}


// 2. sigmoid_propatation : sigmoid 적용한 propagation 계산
float* sigmoid_propatation(Module self, float* input) {

	float* result = (float*)malloc(sizeof(float) * self.out_channels);

	for (int i = 0; i < self.in_channels; i++) {
		self.input[i] = input[i];
	}

	// result = 1 / (1 + exp(-net))
	for (int i = 0; i < self.out_channels; i++)
	{
		result[i] = 1 / (1 + exp(-input[i]));
	}
	return result;
}


// 3. sigmoid_backpropagation : sigmoid 적용한 backpropagation 계산
float* sigmoid_backpropagation(Module self, float* loss, float c) {

	for (int i = 0; i < self.in_channels; i++) {
		loss[i] = loss[i] * (1 - loss[i]);
	}

	return loss;
}