// 과제#3 컴퓨터과학부 2017920036 양다은

#include <iostream>
#include <fstream>	// 파일 입출력 헤더파일
#include <cstdlib>
#include <cmath>
#include "module.h"
#include "train.h"
using namespace std;

int main() {

	// 파일 출력을 위해 ofstream 객체 생성
	ofstream fconfig("config.txt");
	fconfig << "##### 학습환경 #####" << endl << endl;	// config.txt에 데이터 쓰기


	float gate_input[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };	// (AND, OR, XOR) X1과 X2
	float donut_input[9][2] = {
		{0.,0.}, {0.,1.}, {1.,0.},
		{1.,1.},{0.5,1.}, {1.,0.5},
		{0.,0.5}, {0.5,0.}, {0.5,0.5}
	};	// (DONUT) X1과 X2

	float and_output[4] = { 0, 0, 0, 1 };					// AND 연산에 대한 output
	float or_output[4] = { 0, 1, 1, 1 };					// OR 연산에 대한 output
	float xor_output[4] = { 0, 1, 1, 0 };					// XOR 연산에 대한 output
	float donut_output[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };	// DONUT에 대한 output

	// multi layer 수 입력받기
	int layer;
	cout << "Layer ? ";
	cin >> layer;
	fconfig << "Layer: " << layer << endl << endl;	// config.txt에 데이터 쓰기


	// 실행할 module 생성 및 초기화
	int in_nodes;	// 입력 node 수
	int out_nodes;	// 출력 node 수
	// module 생성 시, 하나의 layer 당 두 개의 module 생성
	// 따라서 layer*2 배로 동적할당
	Module * module = (Module*)malloc(sizeof(Module) * (layer * 2));
	for (int i = 0; i < (layer*2); i += 2) {

		// layer 당 node 수 입력받기
		cout << (i/2)+1 <<"번째 입력 노드 수와 출력 노드 수 ? ";
		cin >> in_nodes >> out_nodes;

		module[i] = init_linear(in_nodes, out_nodes);
		module[i + 1] = init_sigmoid(out_nodes);
	}


	//// layer 당 node 수 입력받기
	//int* layer_nodes = (int*)malloc(sizeof(Module) * (layer+1));
	//for (int i = 0; i < layer; i++) {
	//	cout << i+1<<"번째 layer의 node 수 ? ";
	//	cin >> layer_nodes[i];
	//}
	//layer_nodes[layer] = 1;	// 마지막 out_node 수 = 1 

	//// config.txt에 데이터 쓰기
	//for (int i = 0; i < layer; i++) {
	//	fconfig << i + 1 << "번째 layer의 node 수: " << layer_nodes[i] << endl; 
	//}

	//// 실행할 module 생성 및 초기화
	//int in_nodes;		// 입력 node 수 
	//int out_nodes;	// 출력 node 수
	//// module 생성 시, 하나의 layer 당 두 개의 module 생성
	//// 따라서 layer*2 배로 동적할당
	//Module* module = (Module*)malloc(sizeof(Module) * (layer*2)); 
	//for (int i = 0; i < layer; i ++) {
	//	in_nodes = layer_nodes[i];
	//	out_nodes = layer_nodes[i+1];
	//	module[i] = init_linear(in_nodes, out_nodes);
	//	module[i + 1] = init_sigmoid(out_nodes);
	//}

	int gate;	// 실행할 연산 선택을 위한 변수 선언

	// 실행할 gate 선택(1. AND    2. OR    3. XOR	4. DONUT)
	cout << endl << "GATE 선택 : 1. AND    2. OR    3. XOR	4. DONUT" << endl << "Select : ";
	while (1) {
		cin >> gate;
		if (gate != 1 && gate != 2 && gate != 3 && gate != 4) {	// 부적절한 gate 입력하면 발생할 에러 차단
			cout << "1 / 2 / 3 / 4 중 선택하기" << endl;
		}
		else break;
	}

	cout << "------------------------------------------------" << endl;
	if (gate == 1) {
		cout << "AND gate" << endl << endl;
		fconfig << endl << "AND gate Training " << endl;	// config.txt에 데이터 쓰기
		train(module, layer, gate_input, and_output, 4);
	}
	else if (gate == 2) {
		cout << "OR gate" << endl << endl;
		fconfig << endl << "OR gate Training " << endl;	// config.txt에 데이터 쓰기
		train(module, layer, gate_input, or_output, 4);
	}
	else if (gate == 3) {
		cout << "XOR gate" << endl << endl;
		fconfig << endl << "XOR gate Training " << endl;	// config.txt에 데이터 쓰기
		train(module, layer, gate_input, xor_output, 4);
	}
	else if (gate == 4) {
		cout << "DONUT" << endl << endl;
		fconfig << endl << "DONUT Training " << endl;	// config.txt에 데이터 쓰기
		train(module, layer, donut_input, donut_output, 9);
	}
	cout << "학습 종료" << endl;

	// ofstream 객체 반환
	fconfig.close();

	return 0;
}