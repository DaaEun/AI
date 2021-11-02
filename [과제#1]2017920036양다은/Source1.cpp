// 과제#1 컴퓨터과학부 2017920036 양다은
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {

	srand((unsigned int)time(NULL));	// rand() 함수 사용을 위해 시간에 따라 난수표 초기화
	
	cout << "2개의 Input과 AND 연산에 대한 Output 출력" 
		<< endl << "(Perceptron의 weight와 theta 값의 영향 받음)" << endl;	// weight = 가중치, theta = 임계값
	
	int* weight_arr = new int[2];	// weight인 W1과 W2 배열  
	int theta;
	int input[4][2] = { {0,0},{0,1},{1,0},{1,1} };	// X1과 X2
	int output[4] = { 0,0,0,1 };	// AND 연산에 대한 output
	int count = 0;	// 총 실행횟수

	while (1) {

		int net = 0;	// 계산값 0 초기화
		int result;		// 계산값에 대한 결과값
		int incorrect = 0;	// 틀린 횟수 0 초기화

		// 1. W1과 W2 랜덤하게 초기화
		for (int i = 0; i < 2; i++) {
			*(weight_arr + i) = rand() % 100 + 1;	// 1 ~ 100 중 하나의 수
		}

		// 2. theta 랜덤하게 초기화
		theta = rand() % 100 + 1;	// 1 ~ 100 중 하나의 수

		// W1, W2, theta 값 출력
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

			// 4. result 도출
			// net = X1*W1 + X2*W2 - theta
			// net > 0 이면 result = 1
			// net <= 0 이면 result = 0
			if (net > 0) result = 1;
			else result = 0;

			// outout, result 값 출력
			cout << "output : " << output[i] 
				<< "	result : " << result << endl;
			
			// 5. result와 outout이 같지 않으면 incorrect(= 틀린 횟수) + 1
			if (result != output[i]) incorrect++;
		}
		
		// 총 실행 횟수 + 1
		count++;
		// 틀린 횟수 출력
		cout << "틀린 횟수 : " << incorrect << endl;
		// 6. 틀린 횟수가 없다면, 무한 루프 탈출
		if (incorrect == 0) break;
	}

	// 총 실행 횟수 출력
	cout << endl << "총 실행 횟수 : " << count << endl;
	
	return 0;
}