// forward pass compute -> propagation
// backward pass compute -> backpropagation

#include <iostream>
#include <fstream>	// ���� ����� �������
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "module.h"
#include "train.h"
using namespace std;

// ���� ����� ���� ofstream ��ü ����
ofstream fweight("weight.txt");

// Node class ����
// 1. init_node : �ʱ� ���
Node init_node(int in_nodes) {

	Node node;

	node.in_nodes = in_nodes;								// �Է� node �� 
	node.weight = (float*)malloc(sizeof(float) * in_nodes);	// weight ��
	node.bias = (float*)malloc(sizeof(float) * in_nodes);	// bias ��
	node.propagation = node_propagation;						// propagation 
	node.backpropagation = node_backpropagation;				// backpropagation

	srand((unsigned int)time(NULL));	// rand() �Լ� ����� ���� �ð��� ���� ����ǥ �ʱ�ȭ
	for (int i = 0; i < in_nodes; i++) {
		node.weight[i] = (float)((rand() % 10000) * 0.0001f);
		node.bias[i] = (float)((rand() % 10000) * 0.0001f);
		//cout << "i: " << i << "	weight:" << node.weight[i] << "	bias: " << node.bias[i] << endl;
	}

	return node;
}


// 2. node_propagation : ��� propagation ���
float node_propagation(Node self, float* input) {

	float net = 0;

	// net = X1*W1 + X2*W2 + ... + Xn*Wn
	for (int i = 0; i < self.in_nodes; i++) {
		net += input[i] * self.weight[i] + self.bias[i];
	}

	return net;
}


// 3. node_backpropagation : ��� backpropagation ���
void node_backpropagation(Node self, float* input, float delta, float c) {

	//cout << "delta: " << delta << endl;
	for (int i = 0; i < self.in_nodes; i++) {
		// W = W + (-c * delta * input)
		// bias = bias + (-c * delta * 1)
		self.weight[i] += (-1) * c * delta * input[i];
		self.bias[i] += (-1) * c * delta * 1;

		//cout << "i: " << i << "input: " << input[i] << "	weight: " << self.weight[i] << "	bias: " << self.bias[i] << endl;
		fweight << "input: " << input[i] << "	weight: " << self.weight[i] << "	bias: " << self.bias[i] << endl;	// weight.txt�� ������ ����
	}

	fweight << endl;	// weight.txt�� ������ ����
}


// Linear class ����
// 1. init_linear : �ʱ� ���
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


// 2. linear_propatation : linear ������ propagation ���
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


// 3. linear_backpropagation : linear ������ backpropagation ���
float* linear_backpropagation(Module self, float* delta, float c) {

	// ���ο� delta ����� ���� ���� ����
	float* new_delta = (float*)calloc(self.in_nodes, sizeof(float));

	for (int i = 0; i < self.in_nodes; i++) {
		new_delta[i] = 0;

		for (int j = 0; j < self.out_nodes; j++) {
			// j��° node�� i��° weight �� = w[ji]
			// delta[i] = delta[j] * w[ji]  
			new_delta[i] += delta[j] * self.nodes[j].weight[i];

		}
	}

	for (int i = 0; i < self.out_nodes; i++) {
		self.nodes[i].backpropagation(self.nodes[i], self.input, delta[i], c);
	}

	return new_delta;
}


// Sigmoid class ����
// 1. init_sigmoid : �ʱ� ���
Module init_sigmoid(int in_nodes) {

	Module sigmoid;

	sigmoid.in_nodes = in_nodes;
	sigmoid.out_nodes = in_nodes;
	sigmoid.input = (float*)malloc(sizeof(float) * in_nodes);
	sigmoid.propagation = sigmoid_propatation;
	sigmoid.backpropagation = sigmoid_backpropagation;

	return sigmoid;
}


// 2. sigmoid_propatation : sigmoid ������ propagation ���
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


// 3. sigmoid_backpropagation : sigmoid ������ backpropagation ���
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