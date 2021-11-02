// ����#1 ��ǻ�Ͱ��к� 2017920036 �����
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {

	srand((unsigned int)time(NULL));	// rand() �Լ� ����� ���� �ð��� ���� ����ǥ �ʱ�ȭ
	
	cout << "2���� Input�� AND ���꿡 ���� Output ���" 
		<< endl << "(Perceptron�� weight�� theta ���� ���� ����)" << endl;	// weight = ����ġ, theta = �Ӱ谪
	
	int* weight_arr = new int[2];	// weight�� W1�� W2 �迭  
	int theta;
	int input[4][2] = { {0,0},{0,1},{1,0},{1,1} };	// X1�� X2
	int output[4] = { 0,0,0,1 };	// AND ���꿡 ���� output
	int count = 0;	// �� ����Ƚ��

	while (1) {

		int net = 0;	// ��갪 0 �ʱ�ȭ
		int result;		// ��갪�� ���� �����
		int incorrect = 0;	// Ʋ�� Ƚ�� 0 �ʱ�ȭ

		// 1. W1�� W2 �����ϰ� �ʱ�ȭ
		for (int i = 0; i < 2; i++) {
			*(weight_arr + i) = rand() % 100 + 1;	// 1 ~ 100 �� �ϳ��� ��
		}

		// 2. theta �����ϰ� �ʱ�ȭ
		theta = rand() % 100 + 1;	// 1 ~ 100 �� �ϳ��� ��

		// W1, W2, theta �� ���
		cout << "W1 : " << *weight_arr 
			<< "		W2 : " << *(weight_arr + 1) 
			<< "		theta : " << theta << endl;

		for (int i = 0; i < 4; i++) {
			// 3-1. net = X1*W1 + X2*W2
			for (int j = 0; j < 2; j++) {
				net += input[i][j] * weight_arr[j];
			}

			// 3-2. net = net - theta
			net -= theta;

			// 4. result ����
			// net = X1*W1 + X2*W2 - theta
			// net > 0 �̸� result = 1
			// net <= 0 �̸� result = 0
			if (net > 0) result = 1;
			else result = 0;

			// outout, result �� ���
			cout << "output : " << output[i] 
				<< "	result : " << result << endl;
			
			// 5. result�� outout�� ���� ������ incorrect(= Ʋ�� Ƚ��) + 1
			if (result != output[i]) incorrect++;
		}
		
		// �� ���� Ƚ�� + 1
		count++;
		// Ʋ�� Ƚ�� ���
		cout << "Ʋ�� Ƚ�� : " << incorrect << endl;
		// 6. Ʋ�� Ƚ���� ���ٸ�, ���� ���� Ż��
		if (incorrect == 0) break;
	}

	// �� ���� Ƚ�� ���
	cout << endl << "�� ���� Ƚ�� : " << count << endl;
	
	return 0;
}