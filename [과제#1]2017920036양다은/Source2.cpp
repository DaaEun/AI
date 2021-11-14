// ����#1 ��ǻ�Ͱ��к� 2017920036 �����
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {

	srand((unsigned int)time(NULL));	// rand() �Լ� ����� ���� �ð��� ���� ����ǥ �ʱ�ȭ

	cout << "N Dimension 1-layer Perceptron" << endl
		<< "(weight�� theta ���� ���� ����)" << endl << endl;	// weight = ����ġ, theta = �Ӱ谪
	
	int N, Case;	// N = input�� ����, Case = input ������ ��

	// 1. N �� �Է�
	cout << "N ? ";
	cin >> N;

	// input ������ �� = N^2, 1�� N��ŭ ������ ����Ʈ ����
	Case = 1 << N;	

	// 2. input �� ���� (X1, X2, ... Xn) 
	int** inputs = (int**)malloc(sizeof(int*) * Case);	// Case�� �����ϴ� inputs �迭 ���� �� �޸� �����Ҵ�
	for (int i = 0; i < Case; i++) {
		inputs[i] = (int*)malloc(sizeof(int) * N);		// N���� input�� �����ϴ� �޸� �����Ҵ�
	}
	for (int i = 0; i < Case; i++) {
		for (int j = N - 1; j >= 0; j--) {
			int input = i >> j & 1;
			//cout << input;
			inputs[i][j] = input;
		}
		//cout << endl;
	}

	// 3. AND ���꿡 ���� output �� ���� (O1=0, O2=0, ... On=1)
	int* outputs = (int*)malloc(sizeof(int*) * Case);	// Case���� output�� �����ϴ� outputs �迭 ���� �� �޸� �����Ҵ�
	for (int i = 0; i < Case; i++) {
		int output = 1;
		for (int j = 0; j < N; j++) {
			if (inputs[i][j] == 0) {
				output = 0;
				break;
			}
		}
		//cout << output;
		outputs[i] = output;
	}

	clock_t start, end;	// start = ���۽ð�, end = ����ð�
	int* weight_arr = (int*)malloc(sizeof(int*) * N);	// N���� weight�� �����ϴ� �迭 ���� �� �޸� �����Ҵ�
	int theta;
	int count = 0;	// �� ����Ƚ��

	start = clock();	// �ð� ���� ����

	while (1) {

		int net = 0;		// ��갪 0 �ʱ�ȭ
		int result;			// ��갪�� ���� �����
		int incorrect = 0;	// Ʋ�� Ƚ�� 0 �ʱ�ȭ

		// 4. weight �����ϰ� �ʱ�ȭ (W1, W2, ... Wn)
		for (int i = 0; i < N; i++) {
			*(weight_arr + i) = rand() % 10 + 1;	// 1 ~ 10 �� �ϳ��� ��
		}

		// 5. theta �����ϰ� �ʱ�ȭ
		theta = rand() % 10 + 1;	// 1 ~ 10 �� �ϳ��� ��

		// 6-1. input�� weight ���
		// net = X1*W1 + X2*W2 + ... + Xn*Wn
		for (int i = 0; i < Case; i++) {
			for (int j = 0; j < N; j++) {
				net += inputs[i][j] * weight_arr[j];
			}

			// 6-2. net = net - theta
			net -= theta;

			// 7. result ����
			// net > 0 �̸� result = 1
			// net <= 0 �̸� result = 0
			if (net > 0) result = 1;
			else result = 0;

			// outout, result �� ���
			//cout << "output : " << outputs[i]
			//	<< "	result : " << result << endl;

			// 8. result�� outout�� ���� ������ incorrect(= Ʋ�� Ƚ��) + 1
			if (result != outputs[i]) incorrect++;
		}

		// �� ���� Ƚ�� + 1
		count++;
		// Ʋ�� Ƚ�� ���
		cout << count << "��° Ʋ�� Ƚ�� : " << incorrect << endl;
		// 9. Ʋ�� Ƚ���� ���ٸ�, ���� ���� Ż��
		if (incorrect == 0) break;
	}
	
	end = clock();	// �ð� ���� ����

	// �� ���� Ƚ�� ���
	cout << endl << "�� ���� Ƚ�� : " << count << endl;
	// �� ���� �ð� ���
	cout << "�� ���� �ð� : " << (double)(end - start) << "ms" << endl;

	// 10. �����Ҵ��� ���� ����
	for (int i = 0; i < N; i++) free(inputs[i]);
	free(inputs);
	free(outputs);
	free(weight_arr);

	return 0;
}