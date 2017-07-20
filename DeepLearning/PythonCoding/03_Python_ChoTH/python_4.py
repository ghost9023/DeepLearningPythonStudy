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
    break     # break에 의해 반복문이 중단되었는지 확인하는 것
else:         # 즉 break가 작동되지 않고 정상적으로 반복문이 모두 완료되면 else가 작동한다.
    print('This is not much of a cheese shop, is it?')

# 좀 더 직관적으로 알아볼 수 있는 break, else 하드코드
cheese = []
found_one = False
for cheese in cheeses:
    found_one = True
    print('This shop has some lovely', cheese)
    break
if not found_one:
    print('This is not much of a cheese shop, is it?')

# 여러시퀀스 순회하기: zip()
days = ['Monday', 'Tuesday', 'Wednesday']
fruits = ['banana', 'orange', 'peach']
drinks = ['coffee', 'tea', 'beer']
desserts = ['tiramisu', 'ice cream', 'pie', 'pudding']
for day, fruit, drink, dessert in zip(days, fruits, drinks, desserts):
    print(day, '/', fruit, '/', drink, '/', dessert)
# 여러 시퀀스중 가장 짧은 시퀀스가 완료되면 zip()은 멈춘다.

english = ('Monday', 'Tuesday', 'Wednesday')
french = ('Lundi', 'Mardi', 'Mercredi')
# 두개의 튜플을 순회 가능한 튜플로 만들기 위해 zip()을 사용한다.
# zip()에 의해 반환되는 값은 튜플이나 리스트 자신이 아니라 하나로 반환될 수 있는 순회 가능한 값이다.
list(zip(english, french))    # [('Monday', 'Lundi'), ('Tuesday', 'Mardi'), ('Wednesday', 'Mercredi')]
tuple(zip(english, french))   # (('Monday', 'Lundi'), ('Tuesday', 'Mardi'), ('Wednesday', 'Mercredi'))
dict(zip(english, french))    # {'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi'}

# 숫자 시퀀스 생성하기: range()
for x in range(0,3):
    print(x)
list(range(0,3))

for x in range(2,-1,-1):
    print(x)
list(range(2,-1,-1))

list(range(0,11,2))

# 컴프리헨션
# 컴프리헨션이란 하나 이상의 이터레이터로부터 파이썬의 자료구조를 만드는 방법
number_list = []
for number in range(1,6):
    number_list.append(number)
number_list

number_list = list(range(1,101))
number_list = [a for a in range(1, 101)]
number_list = [a+1 for a in range(1,101)]

# [표현식 for 항목 in 순회가능한 객체 if 조건]
a_list = []
for number in range(1,6):
    if number % 2 == 1:
        a_list.append(number)
a_list

a_list = [number for number in range(1,6) if number%2 == 1]
a_list   # 이렇게 더 콤팩트한 방법 사용가능

# 이중루프 컴프리헨션
rows = range(1,4)
cols = range(1,3)
for row in rows:
    for col in cols:
        print(row, col)

rows = range(1,4)
cols = range(1,3)
cells = [(row, col) for row in rows for col in cols]
for row, col in cells:
    print(row, col)

# 딕셔너리 컴프리헨션
# {키_표현식:값_표현식 for 표현식 in 순회 가능한 객체}
word = 'letters'
letter_counts = {letter:word.count(letter) for letter in word}
letter_counts

# 셋 컴프리헨션
a_set = {number for numbers in range(1,6) if number%3 == 1}
a_set

# 제너레이터 컴프리헨션
# 튜플은 컨프리헨션이 없다. 컴프리헨션의 []를 ()로 바꿔서 사용해도 튜플 컴프리헨션이 생성되지 않는다.
number_thing = (number for number in range(1,6))
type(number_thing)

for number in number_thing:
    print(number)   # 이렇게 제너레이터 객체를 순회할 수 있다.

number_list = list(number_thing)
number_list   # 리스트 컴프리헨션처럼 만들기 위해 제너레이터 컴프리헨션에 list() 호출을 통해 랩핑할 수 있다.

try_again = list(number_thing)
try_again
#####★★★★★★★만약 다시 순회하려고 한다면 아무것도 볼 수 없다.
# 제너레이터는 한번만 실행될 수 있다. 리스트, 셋, 문자열, 딕셔너리는 메모리에 존재하지만, 제너레이터는 즉석에서 그 값을 생성하고,
# 이터레이터를 통해서 한번에 값을 하나씩 처리한다. 제너레이터는 이 값을 기억하지 않기 때문에 다시 시작하거나 제너레이터를 백업할 수 없다.

# 함수
# 함수는 입렵 매개변수로 모든 타입을 여러개 취할 수 있다. 그리고 반환값으로 모든 타입을 여러 개 반환할 수 있다.

def do_nothing():
    pass   # 아무것도 하지 않는다는 것을 보여주기 위해 pass

def make_a_sound():
    print('quack')
make_a_sound()

def agree():
    return True
if agree():
    print('Splendid')
else:
    print('That was unexpected')

def echo(anything):
    return anything + ' ' + anything
echo('Rumplestiltskin')

def commentary(color):
    if color == 'red':
        return "It's a tomato"
    elif color == 'green':
        return "It's a green pepper"
    elif color == 'bee purple':
        return "I don't know what it is, but only bees can see it"
    else:
        return "I've never heard of the color " + color + ","
comment = commentary('blue')   # 매개변수가 들어가서 반환된 값이 저장된다.
print(comment)

# 만약 함수가 명시적으로 return을 호출하지 않으면 호출자는 반환값으로 return을 얻는다.

# 유용한 None
thing = None
if thing:
    print("It's something")
else:
    print("It's nothing")
# True는 아니지만 False도 아니다.

if thing is None:
    print("It's nothing")
else:
    print("It's something")

def is_none(thing):
    if thing is None:
        print("It's None")
    elif thing:
        print("It's True")
    else:
        print("It's False")

is_none(None)
is_none(True)
is_none(False)
is_none(0)
is_none(0.0)
is_none(())
is_none([])
is_none({})
is_none(set())




