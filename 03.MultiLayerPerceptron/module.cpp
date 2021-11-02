// forward pass compute -> propagation
// backward pass compute -> backpropagation

#include <iostream>
#include <fstream>	// 파일 입출력 헤더파일
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "module.h"
#include "train.h"
using namespace std;

// 파일 출력을 위해 ofstream 객체 생성
ofstream fweight("weight.txt");

// Node class 구현
// 1. init_node : 초기 노드
Node init_node(int in_nodes) {

	Node node;

	node.in_nodes = in_nodes;								// 입력 node 수 
	node.weight = (float*)malloc(sizeof(float) * in_nodes);	// weight 값
	node.bias = (float*)malloc(sizeof(float) * in_nodes);	// bias 값
	node.propagation = node_propagation;						// propagation 
	node.backpropagation = node_backpropagation;				// backpropagation

	srand((unsigned int)time(NULL));	// rand() 함수 사용을 위해 시간에 따라 난수표 초기화
	for (int i = 0; i < in_nodes; i++) {
		node.weight[i] = (float)((rand() % 10000) * 0.0001f);
		node.bias[i] = (float)((rand() % 10000) * 0.0001f);
		//cout << "i: " << i << "	weight:" << node.weight[i] << "	bias: " << node.bias[i] << endl;
	}

	return node;
}


// 2. node_propagation : 노드 propagation 계산
float node_propagation(Node self, float* input) {

	float net = 0;

	// net = X1*W1 + X2*W2 + ... + Xn*Wn
	for (int i = 0; i < self.in_nodes; i++) {
		net += input[i] * self.weight[i] + self.bias[i];
	}

	return net;
}


// 3. node_backpropagation : 노드 backpropagation 계산
void node_backpropagation(Node self, float* input, float delta, float c) {

	//cout << "delta: " << delta << endl;
	for (int i = 0; i < self.in_nodes; i++) {
		// W = W + (-c * delta * input)
		// bias = bias + (-c * delta * 1)
		self.weight[i] += (-1) * c * delta * input[i];
		self.bias[i] += (-1) * c * delta * 1;

		//cout << "i: " << i << "input: " << input[i] << "	weight: " << self.weight[i] << "	bias: " << self.bias[i] << endl;
		fweight << "input: " << input[i] << "	weight: " << self.weight[i] << "	bias: " << self.bias[i] << endl;	// weight.txt에 데이터 쓰기
	}

	fweight << endl;	// weight.txt에 데이터 쓰기
}


// Linear class 구현
// 1. init_linear : 초기 계산
Module init_linear(int in_nodes, int out_nodes) {

	Module linear;

	linear.in_nodes = in_nodes;
	linear.out_nodes = out_nodes;
	linear.input = (float*)malloc(sizeof(float) * in_nodes);
	linear.nodes = (Node*)malloc(sizeof(Node) * out_nodes);
	linear.propagation = linear_propatation;
	linear.backpropagation = linear_backpropagation;

	for (int i = 0; i < out_nodes; i++) {
		linear.nodes[i] = init_node(in_nodes);
	}

	return linear;
}


// 2. linear_propatation : linear 적용한 propagation 계산
float* linear_propatation(Module self, float* input) {

	Node node;

	float* result = (float*)malloc(sizeof(float) * self.out_nodes);

	for (int i = 0; i < self.in_nodes; i++) {
		self.input[i] = input[i];
	}

	for (int i = 0; i < self.out_nodes; i++)
	{
		node = self.nodes[i];
		result[i] = node.propagation(node, input);
	}

	return result;
}


// 3. linear_backpropagation : linear 적용한 backpropagation 계산
float* linear_backpropagation(Module self, float* delta, float c) {

	// 새로운 delta 계산을 위한 변수 선언
	float* new_delta = (float*)calloc(self.in_nodes, sizeof(float));

	for (int i = 0; i < self.in_nodes; i++) {
		new_delta[i] = 0;

		for (int j = 0; j < self.out_nodes; j++) {
			// j번째 node의 i번째 weight 값 = w[ji]
			// delta[i] = delta[j] * w[ji]  
			new_delta[i] += delta[j] * self.nodes[j].weight[i];

		}
	}

	for (int i = 0; i < self.out_nodes; i++) {
		self.nodes[i].backpropagation(self.nodes[i], self.input, delta[i], c);
	}

	return new_delta;
}


// Sigmoid class 구현
// 1. init_sigmoid : 초기 계산
Module init_sigmoid(int in_nodes) {

	Module sigmoid;

	sigmoid.in_nodes = in_nodes;
	sigmoid.out_nodes = in_nodes;
	sigmoid.input = (float*)malloc(sizeof(float) * in_nodes);
	sigmoid.propagation = sigmoid_propatation;
	sigmoid.backpropagation = sigmoid_backpropagation;

	return sigmoid;
}


// 2. sigmoid_propatation : sigmoid 적용한 propagation 계산
float* sigmoid_propatation(Module self, float* input) {

	float* result = (float*)malloc(sizeof(float) * self.out_nodes);

	for (int i = 0; i < self.in_nodes; i++) {
		self.input[i] = input[i];
	}

	// result = 1 / (1 + exp(-net))
	for (int i = 0; i < self.out_nodes; i++)
	{
		result[i] = 1 / (1 + exp(-input[i]));
	}
	return result;
}


// 3. sigmoid_backpropagation : sigmoid 적용한 backpropagation 계산
float* sigmoid_backpropagation(Module self, float* delta, float c) {

	for (int i = 0; i < self.in_nodes; i++) {
		
		// result = 1 / (1 + exp(-net))
		float result = 1 / (1 + exp(-self.input[i]));

		// result's derivative = result * (1 - result)
		// delta = delta * result * (1 - result)
		delta[i] = delta[i] * result * (1 - result);
	}

	return delta;
}