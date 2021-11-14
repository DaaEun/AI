// forward pass compute -> propagation
// backward pass compute -> backpropagation

#pragma once

typedef struct Module Module;
typedef struct Node Node;


// module ����
struct  Module {

	int in_nodes;	// �Է� node �� 
	int out_nodes;	// ��� node ��
	Node* nodes;		// ��� ��
	float* input;		// input ��
	float* (*propagation)(struct Module, float* input);			// propagation 
	float* (*backpropagation)(struct Module, float*, float);	// backpropagation
};


// Node ����
struct Node {

	int in_nodes;	// �Է� node ��
	float* weight;		// weight ��
	float* bias;		// bias ��
	float (*propagation)(struct Node, float* input);					// propagation 
	void (*backpropagation)(struct Node, float* input, float, float);	// backpropagation
};


// Node method
Node init_node(int in_nodes);	//�ʱ� node
float node_propagation(Node self, float* input);	// propagation ���
void node_backpropagation(Node self, float* input, float delta, float c);	// backpropagation ���

// class : Linear
// implement : Module
// Linear module method
Module init_linear(int in_nodes, int out_nodes);	//�ʱ� ���
float* linear_propatation(Module self, float* input);	// linear module ������ propagation ���
float* linear_backpropagation(Module self, float* delta, float c);	// linear module ������ backpropagation ���

// class : Sigmoid
// implement : Module 
// Sigmoid module method 
Module init_sigmoid(int in_nodes);	//�ʱ� ���
float* sigmoid_propatation(Module self, float* input);	// sigmoid module ������ propagation ���
float* sigmoid_backpropagation(Module self, float* delta, float c);	// sigmoid module ������ backpropagation ���