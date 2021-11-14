// 과제#1 컴퓨터과학부 2017920036 양다은
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {

	srand((unsigned int)time(NULL));	// rand() 함수 사용을 위해 시간에 따라 난수표 초기화

	cout << "N Dimension 1-layer Perceptron" << endl
		<< "(weight와 theta 값의 영향 받음)" << endl << endl;	// weight = 가중치, theta = 임계값
	
	int N, Case;	// N = input의 차원, Case = input 조합의 수

	// 1. N 값 입력
	cout << "N ? ";
	cin >> N;

	// input 조합의 수 = N^2, 1을 N만큼 오른쪽 쉬프트 연산
	Case = 1 << N;	

	// 2. input 값 생성 (X1, X2, ... Xn) 
	int** inputs = (int**)malloc(sizeof(int*) * Case);	// Case를 포함하는 inputs 배열 선언 및 메모리 동적할당
	for (int i = 0; i < Case; i++) {
		inputs[i] = (int*)malloc(sizeof(int) * N);		// N개의 input을 포함하는 메모리 동적할당
	}
	for (int i = 0; i < Case; i++) {
		for (int j = N - 1; j >= 0; j--) {
			int input = i >> j & 1;
			//cout << input;
			inputs[i][j] = input;
		}
		//cout << endl;
	}

	// 3. AND 연산에 대한 output 값 생성 (O1=0, O2=0, ... On=1)
	int* outputs = (int*)malloc(sizeof(int*) * Case);	// Case개의 output을 포함하는 outputs 배열 선언 및 메모리 동적할당
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

	clock_t start, end;	// start = 시작시간, end = 종료시간
	int* weight_arr = (int*)malloc(sizeof(int*) * N);	// N개의 weight를 포함하는 배열 선언 및 메모리 동적할당
	int theta;
	int count = 0;	// 총 실행횟수

	start = clock();	// 시간 측정 시작

	while (1) {

		int net = 0;		// 계산값 0 초기화
		int result;			// 계산값에 대한 결과값
		int incorrect = 0;	// 틀린 횟수 0 초기화

		// 4. weight 랜덤하게 초기화 (W1, W2, ... Wn)
		for (int i = 0; i < N; i++) {
			*(weight_arr + i) = rand() % 10 + 1;	// 1 ~ 10 중 하나의 수
		}

		// 5. theta 랜덤하게 초기화
		theta = rand() % 10 + 1;	// 1 ~ 10 중 하나의 수

		// 6-1. input과 weight 계산
		// net = X1*W1 + X2*W2 + ... + Xn*Wn
		for (int i = 0; i < Case; i++) {
			for (int j = 0; j < N; j++) {
				net += inputs[i][j] * weight_arr[j];
			}

			// 6-2. net = net - theta
			net -= theta;

			// 7. result 도출
			// net > 0 이면 result = 1
			// net <= 0 이면 result = 0
			if (net > 0) result = 1;
			else result = 0;

			// outout, result 값 출력
			//cout << "output : " << outputs[i]
			//	<< "	result : " << result << endl;

			// 8. result와 outout이 같지 않으면 incorrect(= 틀린 횟수) + 1
			if (result != outputs[i]) incorrect++;
		}

		// 총 실행 횟수 + 1
		count++;
		// 틀린 횟수 출력
		cout << count << "번째 틀린 횟수 : " << incorrect << endl;
		// 9. 틀린 횟수가 없다면, 무한 루프 탈출
		if (incorrect == 0) break;
	}
	
	end = clock();	// 시간 측정 종료

	// 총 실행 횟수 출력
	cout << endl << "총 실험 횟수 : " << count << endl;
	// 총 실행 시간 출력
	cout << "총 실행 시간 : " << (double)(end - start) << "ms" << endl;

	// 10. 동적할당한 변수 해제
	for (int i = 0; i < N; i++) free(inputs[i]);
	free(inputs);
	free(outputs);
	free(weight_arr);

	return 0;
}