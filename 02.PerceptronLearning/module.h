// forward pass compute -> propagation
// backward pass compute -> backpropagation

#pragma once

typedef struct Module Module;
typedef struct Node Node;


// module ����
struct  Module {

	int in_channels;	// �Է� ä�� ��
	int out_channels;	// ��� ä�� ��
	Node* nodes;		// ��� ��
	float* input;		// input ��
	float* (*propagation)(struct Module, float* input);			// propagation 
	float* (*backpropagation)(struct Module, float*, float);	// backpropagation
};


// Node ����
struct Node {

	int in_channels;	// �Է� ä�� ��
	float* weight;		// weight ��
	float* theta;		// theta ��
	float (*propagation)(struct Node, float* input);					// propagation 
	float* (*backpropagation)(struct Node, float* input, float, float);	// backpropagation
};


// Node method
Node init_node(int in_channels);	//�ʱ� node
float node_propagation(Node self, float* input);	// propagation ���
float* node_backpropagation(Node self, float* input, float loss, float c);	// backpropagation ���

// class : Linear
// implement : Module
// Linear module method
Module init_linear(int in_channels, int out_channels);	//�ʱ� ���
float* linear_propatation(Module self, float* input);	// linear module ������ propagation ���
float* linear_backpropagation(Module self, float* loss, float c);	// linear module ������ backpropagation ���

// class : Sigmoid
// implement : Module 
// Sigmoid module method 
Module init_sigmoid(int in_channels);	//�ʱ� ���
float* sigmoid_propatation(Module self, float* input);	// sigmoid module ������ propagation ���
float* sigmoid_backpropagation(Module self, float* loss, float c);	// sigmoid module ������ backpropagation ���