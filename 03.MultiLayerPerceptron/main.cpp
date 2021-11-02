// ����#3 ��ǻ�Ͱ��к� 2017920036 �����

#include <iostream>
#include <fstream>	// ���� ����� �������
#include <cstdlib>
#include <cmath>
#include "module.h"
#include "train.h"
using namespace std;

int main() {

	// ���� ����� ���� ofstream ��ü ����
	ofstream fconfig("config.txt");
	fconfig << "##### �н�ȯ�� #####" << endl << endl;	// config.txt�� ������ ����


	float gate_input[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };	// (AND, OR, XOR) X1�� X2
	float donut_input[9][2] = {
		{0.,0.}, {0.,1.}, {1.,0.},
		{1.,1.},{0.5,1.}, {1.,0.5},
		{0.,0.5}, {0.5,0.}, {0.5,0.5}
	};	// (DONUT) X1�� X2

	float and_output[4] = { 0, 0, 0, 1 };					// AND ���꿡 ���� output
	float or_output[4] = { 0, 1, 1, 1 };					// OR ���꿡 ���� output
	float xor_output[4] = { 0, 1, 1, 0 };					// XOR ���꿡 ���� output
	float donut_output[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };	// DONUT�� ���� output

	// multi layer �� �Է¹ޱ�
	int layer;
	cout << "Layer ? ";
	cin >> layer;
	fconfig << "Layer: " << layer << endl << endl;	// config.txt�� ������ ����


	// ������ module ���� �� �ʱ�ȭ
	int in_nodes;	// �Է� node ��
	int out_nodes;	// ��� node ��
	// module ���� ��, �ϳ��� layer �� �� ���� module ����
	// ���� layer*2 ��� �����Ҵ�
	Module * module = (Module*)malloc(sizeof(Module) * (layer * 2));
	for (int i = 0; i < (layer*2); i += 2) {

		// layer �� node �� �Է¹ޱ�
		cout << (i/2)+1 <<"��° �Է� ��� ���� ��� ��� �� ? ";
		cin >> in_nodes >> out_nodes;

		module[i] = init_linear(in_nodes, out_nodes);
		module[i + 1] = init_sigmoid(out_nodes);
	}


	//// layer �� node �� �Է¹ޱ�
	//int* layer_nodes = (int*)malloc(sizeof(Module) * (layer+1));
	//for (int i = 0; i < layer; i++) {
	//	cout << i+1<<"��° layer�� node �� ? ";
	//	cin >> layer_nodes[i];
	//}
	//layer_nodes[layer] = 1;	// ������ out_node �� = 1 

	//// config.txt�� ������ ����
	//for (int i = 0; i < layer; i++) {
	//	fconfig << i + 1 << "��° layer�� node ��: " << layer_nodes[i] << endl; 
	//}

	//// ������ module ���� �� �ʱ�ȭ
	//int in_nodes;		// �Է� node �� 
	//int out_nodes;	// ��� node ��
	//// module ���� ��, �ϳ��� layer �� �� ���� module ����
	//// ���� layer*2 ��� �����Ҵ�
	//Module* module = (Module*)malloc(sizeof(Module) * (layer*2)); 
	//for (int i = 0; i < layer; i ++) {
	//	in_nodes = layer_nodes[i];
	//	out_nodes = layer_nodes[i+1];
	//	module[i] = init_linear(in_nodes, out_nodes);
	//	module[i + 1] = init_sigmoid(out_nodes);
	//}

	int gate;	// ������ ���� ������ ���� ���� ����

	// ������ gate ����(1. AND    2. OR    3. XOR	4. DONUT)
	cout << endl << "GATE ���� : 1. AND    2. OR    3. XOR	4. DONUT" << endl << "Select : ";
	while (1) {
		cin >> gate;
		if (gate != 1 && gate != 2 && gate != 3 && gate != 4) {	// �������� gate �Է��ϸ� �߻��� ���� ����
			cout << "1 / 2 / 3 / 4 �� �����ϱ�" << endl;
		}
		else break;
	}

	cout << "------------------------------------------------" << endl;
	if (gate == 1) {
		cout << "AND gate" << endl << endl;
		fconfig << endl << "AND gate Training " << endl;	// config.txt�� ������ ����
		train(module, layer, gate_input, and_output, 4);
	}
	else if (gate == 2) {
		cout << "OR gate" << endl << endl;
		fconfig << endl << "OR gate Training " << endl;	// config.txt�� ������ ����
		train(module, layer, gate_input, or_output, 4);
	}
	else if (gate == 3) {
		cout << "XOR gate" << endl << endl;
		fconfig << endl << "XOR gate Training " << endl;	// config.txt�� ������ ����
		train(module, layer, gate_input, xor_output, 4);
	}
	else if (gate == 4) {
		cout << "DONUT" << endl << endl;
		fconfig << endl << "DONUT Training " << endl;	// config.txt�� ������ ����
		train(module, layer, donut_input, donut_output, 9);
	}
	cout << "�н� ����" << endl;

	// ofstream ��ü ��ȯ
	fconfig.close();

	return 0;
}