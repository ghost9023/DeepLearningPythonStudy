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

