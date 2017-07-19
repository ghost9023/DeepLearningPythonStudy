# CHAPTER 4. 파이크러스트:코드 구조
# 라인 유지하기
alphabet = ''
alphabet += 'abcdefg'
alphabet += 'hijklmn'
alphabet += 'opqrstu'
alphabet += 'vwxyz'
alphabet

alphabet = 'abcdefg' + \
    'hijklmn' + \
    'opqrstu' + \
    'vwxyz'
alphabet

# 비교하기: if, elif, else
disaster = True
if disaster:
    print('woe!')
else:
    print('whee!')
# if와 else는 조건이 참인지 거짓인지 확인하는 파이썬의 선언문(statement)
# print()는 파이썬의 내장함수
# 파이썬 대화형 인터프리터를 사용할 경우 탭과 스페이스를 혼용하지 않는 것을 권한다.
furry = True
small = True
if furry:
    if small:
        print("It's a cat")
    else:
        print("It's a bear")
else:
    if small:
        print("It's a skink")
    else:
        print("It's a human. Or a hairless bear")

color = 'puce'
if color == 'red':
    print("It's a tomato")
elif color == 'green':
    print("It's a green pepper")
elif color == 'bee purple':
    print("I don't know what it is, but only bees can see it")
else:
    print("I've never heard of the color puce", color)

# 파이썬의 비교연산자
# == 같다.
# != 다르다.
# < 보다 작다.
# <= 보다 작거나 같다.
# > 보다 크다.
# >= 보다 크거나 같다.
# in ... 멤버쉽
# 비교연산자는 부울값 True 혹은 False를 반환한다.
x = 7
x == 5
x == 7
x < 10
x > 10
# 만약 동시에 여러개의 식을 비교해야 한다면 최종 부울값을 판단하기 위해 and, or, not와 같은 부울 연산자를 사용할 수 있다.
5 < x and x < 10
(5<x) and (x<10)
5 < x < 10
5 < x < 10 < 999
# 이렇게 비교도 가능하다.

# False는 명시적으로 False라고 표시할 필요가 없다.
# 다음은 모두 False로 간주한다.
# None
# 0
# 0.0
# ''
# []
# ()
# {}
# set()
some_list = []
if some_list:
    print("There's a something in here")
else:
    print("Hey it's empty")

# 반복하기: while
count = 1
while count <= 5:
    print(count)
    count += 1
# while문은 count의 값이 5보다 작거나 같은지 계속 비교한다.
# 변수가 5에서 6으로 증가될 때까지 계속 수행한다.

# 중단하기: break
# 언제 어떤 일이 일어날지 확실하지 않다면 루프 안에 break를문을 사용한다.
while True:
    stuff = input("String to capitalize [type q to quit]:")
    if stuff == 'q':
        break   # q를 입력하면 나와라
    print(stuff.capitalize())

# 건너뛰기: continue
# 반복문을 중단하고 싶지는 않지만 이번 루프의 작동을 완료하고 다음 루프로 건너뛰고 싶을때 사용한다.
while True:
    value = input("Integer, please [q to quit]")
    if value == 'q':
        break
    number = int(value)
    if number % 2 == 0:
        continue
    print(number, "squared is", number**2)

# break 확인하기: else
# break는 어떤 것을 체크하고 반복문을 나가는 명령어이다.
# 하지만 while문이 모두 완료될 때까지 break조건을 찾지 못하면 else가 실행된다.
numbers = [1,3,5]
position = 0
while position < len(numbers):
    number = numbers[position]
    if number % 2 == 0:
        print('Found even number', number)
        break
    position += 1
else:
    print('No even number found')


# 순회하기: for
# 데이터가 메모리에 맞지 않더라도 데이터 스트림을 처리할 수 있도록 허용해준다.
rabbits = ['Flospy', 'Mopsy', 'Cottontail', 'Peter']
current = 0
while current < len(rabbits):
    print(rabbits[current])
    current += 1

# 파이써닉한 우아한 방법
for rabbit in rabbits:
    print(rabbit)

word = 'cat'
for letter in word:
    print(letter)   # 이런 것도 되는구나

accusation = {'room':'balloom', 'weapon':'pipe', 'person':'Col. Mustard'}
for card in accusation:
    print(card)

for value in accusation.values():
    print(value)

for item in accusation.items():
    print(item)

# 한번에 튜플 하나씩만 할당할 수 있다.
# 튜플의 첫번째 내용(키)은 card에 두번째 내용(밸류)는 contents에 할당된다.
for card, contents in accusation.items():
    print('card',card, 'has the contents', contents)

# 중단하기: break
# for문의 break는 while문의 break와 똑같이 동작한다.

# 건너뛰기: continue
# for문의 continue는 while문의 continue와 똑같이 동작한다.

# break 확인하기: else
cheeses = []
for cheese in cheeses:
    print('This shop has some lovely', cheese)
    break
else:
    print('This is not much of a cheese shop, is it?')
# 





