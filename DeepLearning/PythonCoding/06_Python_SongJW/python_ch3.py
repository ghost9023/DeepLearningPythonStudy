'''
python ch.3 - sequence
'''

'''
리스트
 - 순서 존재. 요소의 타입 제한 없음. mutable.
'''
# # 빈 리스트 : list(), =[]
# empty_list1=[]
# empty_list2=list()
# print(empty_list1, empty_list2)
#
# # 리스트로 변환 : list(elem)
# print(list('cat'))
# print(list(('ready', 'aim', 'fire')))   # 튜플을 리스트로
# print('a,b,c,d'.split(',')) # split 함수의 리턴도 리스트
#
# # 리스트 병합 : .extend(list), +=list
# alph = ['A', 'B', 'C']
# abet = ['D', 'E', 'F']
# alph.extend(abet)   # 두 리스트 병합
# print(alph)
# alph+=abet  # 병합
# print(alph)
#
# # 요소 추가 : .append(elem), .insert(ind, elem)
# a = [1,2,3,5]
# a.insert(3, 4)  # params : 차례로 요소를 넣을 인덱스, 요소
# a.insert(-99, 6) # 인덱스가 범위를 넣으면 끝에 붙는다.
# print(a)
#
# # 요소 삭제 : del, .pop(ind)
# del a[0]    # 객체를 삭제하는 것이 아닌 객체로부터 이름을 제거. 이름이 여러개인 경우 모두 지우지 않으면 가비지컬렉션 일어나지 않음.
# print(a)
#
# for i in a :
#     print(i)
#     del i
# print(a)    # [1, 2, 3, 4, 5] : 각 요소에 붙은 임시 이름 'i' 만 제거함. a[ind] 로 이름이 붙어있으니 메모리에서 제거되지 않음
#
# a.remove(3) # 인덱스가 아닌 값으로 요소 삭제
# print(a)
# b=[1,2,1,2,3]
# b.remove(2) # 가장 처음 만나는 것만 삭제한다.
# print(b)
#
# print(a)
# print(a.pop(), a)   # pop() : 가장 끝 요소를 반환하고 삭제.
# print(a.pop(0), a)  # 인덱스를 지정할 수 있다.
#
# # 인덱스 찾기 : .index(elem)
# a=[0,1,2,3,4,5]
# print(a.index(5))   # 요소값으로 인덱스를 찾을 수 있다.
#
# # 요소 존재 확인 : val in list
# print(3 in a)   # 3이 a 안에 존재하는가? : True
#
# # 요소의 수 확인 : .count(elem)
# a=[1,2,2,3,3,3]
# print(a.count(1), a.count(2), a.count(3))
#
# # 문자형 요소들을 연결하여 문자열로 반환 : str.join(list)
# b=['1','2','3']
# print(', '.join(b))
#
# # 정렬 : .sort(reverse=T or F), sorted(list, reverse=T or F)
# a=[1,3,2,5,4]
# print(sorted(a), sorted(a, reverse=True)) # sorted(list) : 리스트를 정렬한 결과를 반환. 기본은 오름차순
# print(a)
# a.sort()
# print(a)    # .sort() : 리스트 내부에서 정렬하고 결과를 유지
# a.sort(reverse=True)
# print(a)
#
# # 리스트 요소 갯수 : len(list)
# print(len(a))
#
# # 할당 : =, .copy(), list(list), =[:]
# a=[1,2,3]
# b=a
# b[0] = 'ho!'
# print(a)    # 할당연산자로 리스트를 할당하는 경우 하나의 리스트에 여러개의 이름을 부여함.
#     # 한 이름으로 리스트를 수정하면 모든 이름에서 같은 리스트를 가리키므로 결과적으로 결과를 공유하는것과 같다.
#
# a=[1,2,3]
# b=a.copy(); b[0]='b'
# c=list(a); c[0]='c'
# d=a[:]; d[0]='d'
# print(a,b,c,d)  # 한 리스트에 여러개의 이름을 부여하지 않고 복사본을 만들어 이름을 부여. 이름 각각 다른 리스트를 가리킴

'''
튜플
 - 리스트와 같되 요소를 변경할 수 없음 immutable.
 - 장점 : 리스트보다 작은 공간, 요소를 안전하게 보관, 딕셔너리키로 사용가능, 네임드 튜플?, 함수 인자들은 튜플로 전달
'''

# # 튜플 생성 : (), tuple()
# empty_tuple1 = ()
# empty_tuple2 = tuple()
# print(empty_tuple1, empty_tuple2)
# tp1 = 'ho!',    # 괄호없이 요소를 나열해도 튜플로 인식
# print(tp1)
# tp2 = 'ho!', 'ha!', 'he!'
# tp3 = ('ho!', 'ha!', 'he!')
# print(tp2, tp3)
#
# # 튜플 packing, unpacking
# tp4 = 'a', 'b', 'c' # 나열된 요소를 튜플로 만듬 : packing
# a, b, c = tp4   # 튜플 요소를 여러 변수에 동시에 할당함 : unpacking
# print(tp4, a, b, c)
# a, b, c = 'a', 'b', 'c' # unpacking
#
# # 튜플로 변환 : tuple()
# a=[1,2,3]
# b=tuple(a)
# print(b)

'''
딕셔너리
 - mutable. 순서 없음. 요소는 키 : 값 쌍. 키는 immutable 객체여야 가능.
'''

# # 딕셔너리 생성 : {}
# empty_dict1 = {}
# print(empty_dict1)
# bierce = {'day':'A period of twenty-four hours, mostly misspent',
#           'positive':'Mistaken at the top of one\'s voice',
#           'misfortune':'The kind of fortune that never misses'}
# print(bierce)
#
# # 딕셔너리로 변환 : dict()
# lol = [['a','b'], ['c','d'], ['e', 'f']]    # 튜플, 리스트의 어떤 조합도 가능
# print(dict(lol))
# lolol = ((1,2),(3,4),(5,6))
# print(dict(lolol))
# lololol = ['ab','cd','12']  # 두개의 문자만 있는 경우 둘을 분리해서 키, 값으로 사용. (2개 초과는 불가)
# print(dict(lololol))
#
# # 항목 추가 / 변경 : dict[key]
# dict1 = {1:2, 3:4, 5:6, 7:8}
# print(dict1)
# print(dict1[1])
# dict1[1] *= 10
# print(dict1)
# dict1[100] = 100
# print(dict1)
#
# # 딕셔너리 결합 : update()
# dict2 = {'a':'A', 'b':'B', 'c':'C', 'd':'D'}
# dict3 = {'e':'E', 'f':'F'}
# dict2.update(dict3)
# print(dict2)
#     # 결합하려는 두 딕셔너리에 같은 키가 있다면 뒤에 이어붙이려는 딕셔너리에 있는 값이 최종적으로 키에 할당된다.
#
# # 항목 삭제 : del
# del dict2['f']
# print(dict2)
#
# # 모든 항목 삭제하기 : clear()
# dict1.clear(); dict2.clear(); dict3.clear()
# print(dict1, dict2, dict3)
#
# # 키의 존재 확인 : key in dict
# dict1 = {'a':'A', 'b':'B', 'c':'C'}
# print('a' in dict1, 'd' in dict1)
#
# # 값 얻기 : dict[key]
# print(dict1['a'])   # 없는 키의 경우 에러발생
# print(dict1.get('a'), dict1.get('d'))   # dict.get(key) : 키에 대응하는 값 반환. 없는 키라면 None 반환
#
# # 모든 키, 값, 키값쌍 얻기 : dict.keys() / dict.values() / dict.items()
# print(dict1.keys()) # dict_keys(['a', 'b', 'c']) : iterable
# print(dict1.values())
# print(dict1.items())
#
# # 할당 : = , 복사 : dict.copy(), dict()
# dict2 = dict1
# print(dict1, dict2)
# dict2['a'] = '@'
# print(dict1, dict2)
# dict1['a'] = 'A'
# dict3 = dict1.copy()
# dict3['a'] = '@'
# print(dict1, dict3)
# dict4 = dict(dict1)
# dict4['a'] = '@'
# print(dict1, dict4)

'''
set
 - 값 없는 딕셔너리. 요소는 유일하며 순서 없음. 존재여부만 판단할때 사용.
'''

# set 생성 : set(), {}
empty_set = set()
print(empty_set)
even_number = {0, 2, 4, 6, 8}
odd_number = set([1, 9, 5, 7, 3])   # set 생성시 리스트의 요소 순서와 다르게 생성된다.
print(even_number, odd_number)
set1 = {1,1,3,5,3,7,9,9,7}
print(set1) # 중복 모두 제거된다.

# set 으로 변경하기 : set()
print(set('letters'))   # 역시 중복 't' 제거
dict1 = {'a':'A', 'b':'B', 'c':'C'}
print(set(dict1))   # key 들만 set 으로 만들어진다. 리스트, 튜플도 모두 set 으로 변경가능.

# 요소 확인 : elem in set
dict_drinks = {'martini' : {'vodka', 'vermouth'},
               'black russian' : {'vodka', 'kahlua'},
               'white russian' : {'cream', 'kahlua', 'vodka'},
               'manhattan' : {'rye', 'vermouth', 'bitters'},
               'screwdriver' : {'orange juice', 'vodka'}}
for name, contents in dict_drinks.items() : # dict.items() 는 키, 값 쌍이 반환되므로 두개의 변수로 unpacking?
    if 'vodka' in contents :    # vodka 가 첨가된 주류의 이름을 출력함.
        print(name)

# 교집합 : 셋 인터섹션 연산자 &, set.intersection(set)
# 두 set 을 비교하여 같은 요소를 갖고있다면 True, 아니면 False 반환
print()
for name, contents in dict_drinks.items() :
    if contents & {'vermouth', 'orange juice'} :    # {'vermouth', 'orange juice'} 와 contents set 의 교집합 찾음.
        print(name)                                 # 교집합이 공집합이 아니라면 True 판정, 공집합이라면 False 판정.

a = {1, 2, 3, 5}
b = {0, 2, 3, 4}
print(a & b)    # 교집합 출력
print(a.intersection(b))    # 교집합 출력
a.intersection_update(b)    # 교집합으로 set 을 덮어씀
print(a)

# 합집합 : 유니온 set.union(set),  |
print()
a = {1,2,3,5}
b = {0,2,3,4}
print(a|b)  # 합집합
print(a.union(b))

# 차집합 : set.difference(set), -
print()
a = {1,2,3,5}
b = {0,2,3,4}
print(a-b)
print(a.difference(b))
a.difference_update(b)
print(a)    # 차집합으로 set 을 덮어씀

# 대칭 차집합 : 두 set 에서 한쪽에만 들어있는 원소들 symmetric_difference, ^
print()
a = {1,2,3,5}
b = {0,2,3,4}
print(a^b)  # {0, 1, 4, 5}
print(a.symmetric_difference(b))    # 차집합과는 다르게 a-b, b-a 의 합집합.
print(a.difference(b).union(b.difference(a)))

# 부분집합 : <=, issubset()
print()
a = {1,2,3}
b = {0,1,2,3,4,5}
print(a<=b, a.issubset(b))

# 진부분집합 proper subset : <
# 부분집합을 포함하고 그 이상의 요소들도 필요
print()
print(a<b, a<a) # a 는 a 의 서브셋이지만 그 외의 요소는 없기에 진부분집합이 아님.

# 슈퍼셋 : 서브셋의 반대 : >=, issuperset()
print()
print(b>=a, b.issuperset(a))

print(a<b, b>a) # a는 b의 proper subset, b 는 a 의 proper superset

'''
딕셔너리의 키
 - immutable 한 객체만이 딕셔너리의 키가 될 수 있다. 즉, 리스트, 셋, 딕셔너리는 키가 될 수 없다.
'''