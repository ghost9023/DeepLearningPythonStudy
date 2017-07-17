# CHAPTER 3
# 리스트는 변경가능, 튜플은 불변

# list(리스트)
# 리스트는 데이터를 순차적으로 파악하는데 유용
# 내용의 순서가 바뀔 수도 있어 유용
# 삭제, 덮어쓰기 가능
empty_list = []
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
big_birds = ['emu', 'ostrich', 'cassowary']
first_names = ['Graham', 'John', 'Terry', 'Terry', 'Michael']

another_empty_list = list()   # 빈 리스트 할당 []
print(another_empty_list)

list('cat')   # 다른 데이터 타입을 리스트로 변환한다.
a_tuple = ('ready', 'fire', 'aim')
list(a_tuple)   # 튜플을 리스트로 변환

birthday = '1/6/1952'
birthday.split('/')   # split()는 문자열을 구분자로 나누어서 리스트로 변환한다.

splitme = 'a/b//c/d///e'
splitme.split('/')   # 구분자가 연속으로 여러개가 있을때 '' 리스트에 빈 문자열이 들어간다.

# [offset]으로 항목 얻기
marxes = ['Groucho', 'Chico', 'Harpo']
marxes[0]
marxes[1]
marxes[2]
# 입력된 오프셋의 값이 리스트의 범위를 넘어갈 경우 에러(예외) 발생

small_birds = ['hummingbird', 'finch']
extinct_birds = ['dodo', 'passenger pigeon', 'Norwegian Blue']
carol_birds = [3, 'French hens', 2, 'turtledoves']
all_birds = [small_birds, extinct_birds, 'macaw', carol_birds]
all_birds   # 리스트 안에 리스트와 아이템이 함께 들어있다.
all_birds[0]
all_birds[2]

marxes = ['Groucho', 'Chico', 'Harpo']
marxes[2] = 'Wanda'
marxes

marxes = ['Groucho', 'Chico', 'Harpo']
marxes[0:2] = ['apple', 'blue']   # 이렇게 바꿀 수도 있음
marxes

marxes[::2]
marxes[::-2]
marxes[::-1]

# 리스트의 끝에 항목 추가하기
marxes.append('Zeppo')
marxes

# 리스트 병합하기
marxes = ['Groucho', 'Chico', 'Harpo', 'Zeppo']
others = ['Gummo', 'Karl']
marxes.extend(others)
marxes   # 오른쪽에 갖다 붙이기

marxes = ['Groucho', 'Chico', 'Harpo', 'Zeppo']
others = ['Gummo', 'Karl']
marxes += others  # 이렇게도 붙인다.
marxes

# append()를 사용하면 항목을 병합하지 않고 others가 하나의 리스트로 병합된다.
marxes = ['Groucho', 'Chico', 'Harpo', 'Zeppo']
others = ['Gummo', 'Karl']
marxes.append(others)   # 리스트로 들어간다.
marxes

# 오프셋과 insert()로 항목 추가하기
marxes = ['Groucho', 'Chico', 'Harpo', 'Zeppo']
marxes.insert(3, 'Gummo')  # 3번째 오프셋에 'Gummo' 삽입
marxes

marxes = ['Groucho', 'Chico', 'Harpo', 'Zeppo']
marxes.insert(10, 'Karl')  # 10번째 오프셋이 없으니까 가장 끝에 'Karl' 삽입
marxes

# 오프셋으로 항목삭제하기:del()
del marxes[-1]   # 끝에 있는 아이템 삭제
marxes

# 오프셋으로 리스트의 특정 항목을 삭제하면 제거된 항목 이후의 항목들이 한 칸씩 앞으로 당겨지고 리스트의 길이가 1 감소
del marxes[2:4]   # 이렇게 두개도 삭제가능
marxes

# 값으로 항목 삭제하기:remove()
marxes = ['Groucho', 'Chico', 'Harpo', 'Gummo', 'Zeppo']
marxes.remove('Gummo')
marxes

# 오프셋으로 항목을 얻은 후 삭제하기: pop()
# 항목들을 가져오면서 삭제, 즉 '빼낸다'고 생각하면 편함
marxes = ['Groucho', 'Chico', 'Harpo', 'Gummo', 'Zeppo']
marxes.pop()
marxes
marxes.pop(2)   # 두번째 오프셋 가져오기
marxes
marxes.pop('Groucho')   # 이렇게 값을 가져올 수는 없다!

# 값으로 항목 오프셋 찾기: index()
# 항목값의 리스트 오프셋을 알기
marxes = ['Groucho', 'Chico', 'Harpo', 'Gummo', 'Zeppo']
marxes.index('Chico')

# 존재여부 확인하기
marxes = ['Groucho', 'Chico','Harpo', 'Zeppo']
'Groucho' in marxes
'Bob' in marxes

words = ['a', 'deer', 'a', 'female', 'deer']
'deer' in words   # 적어도 하나의 값만 존재하면 True를 반환한다.

# 값 세기: count()
marxes = ['Groucho', 'Chico', 'Harpo']
marxes.count('Harpo')   # 'Harpo' 몇개이니
snl_skit = ['cheeseburger', 'cheeseburger', 'cheeseburger']
snl_skit.count('cheeseburger')

