'''
큰 프로그램을 작성하는 방식을 배우자.
'''

# 5.1 스탠드얼론 standalone 프로그램

'''
따로 IDE 를 사용하지 않는다면 워드 프로그램으로 코드를 작성하고 .py 확장자로 저장
 터미널에서 python file_name.py 명령어로 실행한다.
'''

# 5.2 커맨드 라인 인자

# import sys
# print('program arguments :', sys.argv)
    # program arguments : ['C:/dl/DeepLearningPythonStudy/DeepLearning/PythonCoding/06_Python_SongJW/python_ch5.py']
    # 프로그램이 실행될때 python 이 받는 인자를 출력한다.

# 5.3 모듈과 import 문

'''
모듈 = 파이썬 코드를 담은 파일.
코드의 상향식 구조 : 데이터타입 - 선언문 statement - 함수 - 모듈 module
'''

## 5.3.1 모듈 임포트하기

'''
모듈은 .py 확장자가 없는 파이썬 파일의 이름.

함수 내에서도 import 가 가능하다.

1. import module_name -> module_name.object_name 로 호출
2. from module_name import object_name 로 호출
3. 1 방법 뒤에 as alias 를 붙여 alias.object_name 로 호출
   2 방법 뒤에 as alias 를 붙여 alias 를 object_name 대신 사용
'''

# def random_func():
#     import random
#     random_num = random.random()
#     print(random_num)
#
# random_func()   # 랜덤 실수 출력

# print(random.random()) : 에러. 함수내에서 임포트된 모듈은 함수 내에서만 사용 가능하다.

## 5.3.2 다른 이름으로 모듈 임포트하기 : import module_name as alias

## 5.3.3 필요한 모듈만 임포트하기 : from module_name import object_name as alias

## 5.3.4 모듈 검색 경로
'''
디렉토리 이름 리스트와 sys 모듈의 ZIP 아카이브 파일을 변수 PATH 로 사용한다.
'''

# import sys
# for place in sys.path:
#     print(place)    # import 시에 모듈을 검색할 디렉토리들이 출력된다.


# 5.4 패키지

'''
모듈과 __init__.py 파일을 디렉토리에 모아놓으면 그 디렉토리는 패키지로 취급된다.
'''

# 5.5 파이썬 표준 라이브러리

## 5.5.1 누락된 키 처리하기 : setdefault(), defaultdict()

'''
존재하지 않는 키로 딕셔너리 접근시 예외 발생.
dict.get() 을 이용하면 존재하지 않는 키의 경우 None 이 반환된다.
setdefault() 는 get() 과 같은 역할을 하지만 키가 누락된 경우 딕셔너리에 항목을 할당할 수 있다.
'''

### dict.setdefault(key, value)
# periodic_table = {'Hydrogen':1, 'Helium':2}
# print(periodic_table)
#
# carbon = periodic_table.setdefault('Carbon', 12)    # 있다면 값을 반환하고 없다면 키와 값을 추가한 후 반환.
# hydrogen = periodic_table.setdefault('Hydrogen', 1)
# print(carbon, hydrogen,periodic_table)  # 12 1 {'Helium': 2, 'Carbon': 12, 'Hydrogen': 1}


### defaultdict()
# from collections import defaultdict
# print(int())    # int() 는 인자가 없다면 0을 반환한다.
# periodic_table2 = defaultdict(int)  # periodic_table2 의 값의 기본값을 0으로 설정
# periodic_table2['Hydrogen'] = 1 # 키와 값 부여
# print(periodic_table2['Lead'])  # 존재하지 않는 키의 값을 참조 -> 기본값을 설정했으므로 0 출력
# print(periodic_table2)  # Lead 키와 값 0 이 추가되어있음
#
# def no_idea():  # 기본값을 반환할 함수 선언
#     return 'no brain.'
# dict1 = defaultdict(no_idea) # 기본값을 반환할 함수를 인자로 전달
#
# dict1['A'] = 'apple'
# dict1['B'] = 'bubble'
# print(dict1['A'], dict1['B'], dict1['C'])   # 존재하지 않는 키의 경우 no_idea 함수의 반환을 값으로 하여 딕셔너리에 추가
# print(dict1)

#### lambda 사용
# from collections import defaultdict
#
# dict2 = defaultdict(lambda : 'lambda-man')  # 람다식을 이용하면 간편하다.
# print(dict2['ho'])
# print(dict2)

#### 좋은 예제
from collections import defaultdict

food_counter = defaultdict(int)

for food in ['spam', 'spam', 'eggs', 'spam']:
    food_counter[food] += 1 # 키가 없다면 바로 추가하고 연산, 키가 있다면 기존의 값에 연산

for food in food_counter:
    print(food, food_counter[food])

'''
int() 는 0, list() 는 빈 리스트 [], dict() 는 빈 딕셔너리 {} 를 반환한다. 또는 람다식을 쓰자.
defaultdict() 와 같이 인자를 주지 않는 경우 없는 키의 값은 None 으로 설정된다.'''

## 5.5.2 항목세기 : counter()