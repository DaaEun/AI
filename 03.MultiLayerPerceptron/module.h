// forward pass compute -> propagation
// backward pass compute -> backpropagation

#pragma once

typedef struct Module Module;
typedef struct Node Node;


// module 생성
struct  Module {

	int in_nodes;	// 입력 node 수 
	int out_nodes;	// 출력 node 수
	Node* nodes;		// 노드 수
	float* input;		// input 값
	float* (*propagation)(struct Module, float* input);			// propagation 
	float* (*backpropagation)(struct Module, float*, float);	// backpropagation
};


// Node 생성
struct Node {

	int in_nodes;	// 입력 node 수
	float* weight;		// weight 값
	float* bias;		// bias 값
	float (*propagation)(struct Node, float* input);					// propagation 
	void (*backpropagation)(struct Node, float* input, float, float);	// backpropagation
};


// Node method
Node init_node(int in_nodes);	//초기 node
float node_propagation(Node self, float* input);	// propagation 계산
void node_backpropagation(Node self, float* input, float delta, float c);	// backpropagation 계산

// class : Linear
// implement : Module
// Linear module method
Module init_linear(int in_nodes, int out_nodes);	//초기 계산
float* linear_propatation(Module self, float* input);	// linear module 적용한 propagation 계산
float* linear_backpropagation(Module self, float* delta, float c);	// linear module 적용한 backpropagation 계산

// class : Sigmoid
// implement : Module 
// Sigmoid module method 
Module init_sigmoid(int in_nodes);	//초기 계산
float* sigmoid_propatation(Module self, float* input);	// sigmoid module 적용한 propagation 계산
float* sigmoid_backpropagation(Module self, float* delta, float c);	// sigmoid module 적용한 backpropagation 계산