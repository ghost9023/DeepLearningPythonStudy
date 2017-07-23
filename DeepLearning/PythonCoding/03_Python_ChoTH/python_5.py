# # 다른 이름으로 모듈 임포트하기
# # import report as wr
# # description = wr.get_description()
# # print("Today's weather:", description)
# #
# # 필요한 모듈만 임포트하기
# # 모듈 전체 또는 필요한 부분만 임포트할 수 있다.
# from report import get_description
# description = get_description()
# print("Today's weather:", description)
#
# # 모듈검색경로
# # 파이썬은 임포트할 파일을 어디서 찾을까?
# import sys
# for place in sys.path:
#      print(place)
#      # 아주 유용하구나!!!!!
# # 중복된 이름의 모듈이 있다면 첫번째 조건을 사용한다.
# # 즉 앞선 검색경로에 존재하는 모듈을 사용한다는 뜻
#
# # 패키지
# # 파이썬 어플리케이션을 좀 더 확장가능하게 만들기 위해 패키지라는 계층구조에
# # 구성할 수 있다.
# # from 에는 패키지(디렉토리), 모듈(파일)
# # import 에는 모듈(파일), 메소드(함수)
# # 이렇게 입력가능하다.
# # 실행파일과 같은 경로에 있는 패키지나 모듈은 자동으로 패키지 사용경로로 설정된다.
# from sources import daily, weekly
# # from 패키지 import 모듈, 모듈
# # sources 디렉토리에 __init__.py라는 파일이 있어야 그 디렉토리를 패키지로 인식한다.
# # 이 파일의 내용은 비워도 상관없다.
# #
# # 파이썬 표준 라이브러리
# # 빌트인모듈이라고 생각하면 된다.
# #
# # 누락된 키 처리하기: setdefault(), defaultdict()
# # 존재하지 않는 키로 딕셔너리에 접근을 시도하면 예외가 발생한다.
# # setdefault()함수는 get()함수와 비슷하지만 키가 누락된 경우 딕셔너리에 항목을 할당할 수 있다.
# periodic_table = {'Hydrogen': 1, 'Helium': 2}
# print(periodic_table)
#
# carbon = periodic_table.setdefault('Carbon', 12)
# periodic_table
# periodic_table['Carbon'] = 12   # 이거랑 다른게 뭔데??? 이건 그냥 할당
# # 다른게 뭐냐면 setdefault()는 딕셔너리에 접근해서 원하는 값을 할당함과 동시에 접근한다.
# # 대신 존재하는 키에 접근해서 setdefault()를 사용하면 키에 대한 원래값이 반환되고 새로운 값을 할당하지 않는다.
# helium = periodic_table.setdefault('Helium', 947)
# helium
# periodic_table
# # 즉 해당 키의 값을 반환해보고 키가 없어서 반환할게 없으면 새로운 키를 만들고 값을 할당한다.
#
# # defaultdict()함수도 비슷하다.
# # 다른점은 딕셔너리를 생성할 때 모든 새키에 대한 기본값을 먼저 지정한다는 것이다.
# #
# from collections import defaultdict
# periodic_table = defaultdict(int)   # 새로 만들어지는 키에 대한 밸류를 기본으로 0을 지정하는 딕셔너리 생성
# # a = dict(), 그냥 이거 진화한 형태라고 보면 됩니다.
# print(periodic_table)   # 처음엔 빈 리스트
#
# periodic_table['Hydrogen'] = 1  # 1할당
# periodic_table['Lead']          # 0할당
# periodic_table
#
# # defaultdict()의 인자는 값을 누락된 키에 할당하여 반환하는 함수이다.
# from collections import defaultdict
# def no_idea():
#      return 'huh?'
# bestiary = defaultdict(no_idea)   # 이 함수의 매개변수로 함수를 실행한 값을 지정할 수 없다.
# bestiary['A'] = 'Abominable Snowman'
# bestiary['B'] = 'Basilisk'
# print(bestiary['A'])
# print(bestiary['B'])
# print(bestiary['C'])
# # 빈 기본값을 반환하기 위해 int()는 정수 0, list() 함수는 빈 리스트[], dict()함수는 빈 딕셔너리를 반환한다.
# # 인자를 입력하지 않으면 새로운 키의 초기값이 None로 설정된다.
#
# bestiary = defaultdict(lambda: 'Huh?')   # 기본값을 huh?로 할당
# bestiary['E'] # huh? 출력
#
# # 카운터를 만들기 위해 아래와같이 int()함수를 사용할 수 있다.
# from collections import defaultdict
# food_counter = defaultdict(int)
# for food in ['spam', 'spam', 'eggs', 'spam']:
#      food_counter[food] += 1   # 위 리스트에 해당하는 키가 들어올때마다 1씩 더해서 새로 갱신한다.
# for food, count in food_counter.items():
#      print(food, count)   # eggs 1, spam 3
#
# # 위와같은 방법을 defaultdict()를 사용하지 않고 구현하려면 아래와 같이한다.
# dict_counter = {}
# for food in ['spam', 'spam', 'eggs', 'spam']:
#      if not food in dict_counter:
#           dict_counter[food] = 0    # 딕셔너리에 존재하지 않으면 0을 할당하고 있으면 통과
#      dict_counter[food] += 1        # if 절에서 나와서 1을 더해주고
# for food, count in dict_counter.items():
#      print(food, count)
#
# # 항목세기: counter()
from collections import Counter
breakfast = ['spam', 'spam', 'eggs', 'spam']
breakfast_counter = Counter(breakfast)
# breakfast_counter   # Counter({'spam':3, 'eggs':1})

# most_common() 함수는 모든 요소를 내림차순으로 반환한다.
print(breakfast_counter.most_common())
print(breakfast_counter.most_common(2))   # 숫자를 입력하는 경우 그 숫자만큼 상위요소를 반환한다.

# 카운터를 결합할 수도 있다.
breakfast_counter = {'eggs': 3, 'eggs': 1}
# Counter({'spam':3, 'eggs':1})
lunch = ['spam', 'eggs' , 'bacon']
lunch_counter = Counter(lunch)
lunch_counter   # Counter({'eggs':2, 'bacon':1})

breakfast_counter + lunch_counter   # 합집합
breakfast_counter - lunch_counter   # 차집합
breakfast_counter & lunch_counter   # 교집합
























































































