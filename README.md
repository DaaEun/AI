2021년 2학기 인공지능 수강중

- 과제#1 - done
- 과제#2 - done
- 과제#3 - done
- 과제#4 - done
- 과제#5 - 

# LOG

## vscode에서 실습환경 구축

pytorch 실습을 진행하기에 앞서 교수님께서 Google Colab 환경에서 코딩하기를 추천하셨다.    
colab도 병행하여 사용할 예정이나, 학습한 코드를 구글드라이브에서만 저장가능하고, 코드수정이 좀 불편하다고 판단했다.    
그래서 vscode에서 개발환경을 설치해보고자 한다.   
 _하지만 금방 무수한 에러를 경험한다..._     

1. test_CvsF.py 생성과 코드 작성

```python
import numpy as np
import torch
```

2. (Run)실행 --> numpy 모듈 없다고 출력
   
3. numpy, torch 라이브러리 설치
```
$pip install numpy
$pip install torch
```

4. Error 발생
```
WARNING: pip is configured with locations that require TLS/SSL, however the ssl module in Python is not available.   
Could not fetch URL https://pypi.org/simple/pip/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/pip/ (Caused by SSLError("Can't connect to HTT PS URL because the SSL module is not available.")) - skipping
```

5. proxy환경에서 pip를 사용하면 잘 안되는 경우가 있다고 한다. 그럴때 두 가지 방법이 있다.

   1. 첫번째, pip 옵션 사용 --trusted-host=<url>
    ```
    $pip install numpy --trusted-host=pypi.org --trusted-host=pypi.python.org --trusted-host=files.pythonhosted.org --proxy=http://70.10.15.10:8080
    $pip install torch --trusted-host=pypi.org --trusted-host=pypi.python.org --trusted-host=files.pythonhosted.org --proxy=http://70.10.15.10:8080
    ```

   2. 두번째, pip.ini 생성(windows)
      - C:\users\<사용자명>\pip\pip.ini 설치파일 생성
      - 다음 내용 저장
        > [global]   
        > proxy = http://PROXYSERVERIP:PORT   
        > cert = C:\\CERTIFICATION.cer   
        > trusted-host = pypi.python.org   
        >             pypi.org   
        >             files.pythonhosted.org      

6. numpy, torch 라이브러리 재설치 시도
```
$pip install numpy
$pip install torch
```

7. Error 발생 -> Anaconda 설치    
2020년 인턴 경험에서 Anaconda를 사용했기에, 이미 노트북에 설치되어 있다.     
그런데, 파일은 두개이며, 환경변수도 엉망진창 설정되어 있어 다시 삭제하고 설치하고자 한다.   

8. Anaconda 삭제 
Anaconda는 프로그램 추가/삭제가 안된다.    
Anaconda 파일안에 Uninstall-Anaconda.exe 실행.   
삭제 완료.   

9. Anaconda 재설치
    1. [아나콘다 홈페이지](https://www.anaconda.com/products/individual#Downloads)에 접속해서 해당 운영체제 선택
    2. Anaconda3-2021.11-Windows-x86_64.exe 실행
    3. 설치경로 : C:\Anaconda3 변경

10. 환경변수 설정
시스템 경로 환경변수에 다음과 같이 추가(_기존 것은 전부 삭제했다..._)
    - C:\Anaconda3
    - C:\Anaconda3\Library\mingw-w64\bin
    - C:\Anaconda3\Library\usr\bin
    - C:\Anaconda3\Library\bin
    - C:\Anaconda3\Scripts

11. Anaconda Navigator (Anaconda3) 실행
[Anaconda로 PyTorch 설치하고 GPU 사용하기](https://pakalguksu.github.io/development/Anaconda%EB%A1%9C-PyTorch-%EC%84%A4%EC%B9%98%ED%95%98%EA%B3%A0-GPU-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/)

12. 어플리케이션 vscode "lanuch" 

13. 아나콘다 개발환경 확인
```
$conda activate base
```

14.  numpy, torch 라이브러리 재설치 시도
```
$pip install numpy
$pip install torch
```

15. RUN -> Successfully!!
_이제 된다. 휴... 그냥 colab에서 할걸..._

## Anaconda 환경
- vscode에서 .py와 .ipynb 확장자 실행할 수 있다.
