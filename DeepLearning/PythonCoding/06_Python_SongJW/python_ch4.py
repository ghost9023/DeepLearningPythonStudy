'''
python ch.4 - code structure
'''

'''
코멘트 : #
'''
# 호우!

'''
라인 유지 : \
'''
# alphabet = 'abcdefg' +\
#     'hijklmn' +\
#     'opqrstu' +\
#     'vwxyz'
# print(alphabet)
# print(1+2+3+\
#       4+5+6+7+8+9+10)

'''
비교 : if, elif, else
'''
# disaster = True
# if disaster : print('ho!')
# else : print('ha!') # if, else 는 disaster 가 참, 거짓인지 확인하는 선언문 statement
#
# print()
# furry = True
# small = True
# if furry :  # 내포된 if. 2중 if
#     if small :
#         print('cat')
#     else:
#         print('bear')
# else :
#     if small :
#         print('skink')
#     else :
#         print('human')
#
# print()
# color = 'puce'  # 조건의 경우의 수가 많은 경우
# if color == 'red' : print('tomato')
# elif color == 'green' : print('green pepper')
# elif color == 'bee purple' : print('?')
# else : print('wtf')

'''
연산자
==, !=, <, <=, in : 비교연산자
    in : 멤버쉽. 시퀀스의 요소이면 True, 아니면 False
and, or, not : 부울 연산자
    부울연산자는 비교연산자보다 우선순위가 낮다.
'''
# x = 7
# print(x==5, x==7, 5<x, x<10)
# print(5<x and x<10) # 연산자 우선순위. 비교연산자가 부울연산자보다 먼저 수행된다. 가능하면 괄호를 이용한다.
# print(5<x and not x>10) # not 이 and 보다 우선순위가 높다. 외우기 싫으면 괄호쳐라.
# print(5<x<10)   # 5<x and x<10 과 같다.
# y=12
# print(5<x<10<y<15)  # 이런것도 된다.

'''
True, False
    None = null
'''
# if None or 0 or 0.0 or '' or [] or () or {} or set() :  # None, 0, 0.0, '', [], {}, (), set() 모두 False 를 의미한다.
#     print('저 중에 True 가 존재한다.')                  # 이외에는 모두 True

'''
반복 - while
'''
# cnt = 1
# while cnt <= 5: # cnt 가 6이 되면 cnt <= 5 가 false 가 되어 while 탈출
#     print(cnt)
#     cnt+=1

'''
중단 - break
'''
# while True :
#     stuff = input('string to capitalize [type q to quit] : ')
#     if stuff == 'q' :   # 입력이 q 이면 탈출
#         break
#     print(stuff.capitalize())   # 첫 단어의 첫 글자를 대문자로 변경.

'''
건너뛰기 - continue
 다음 루프로 건너뛴다.
'''
# cnt = 0
# while True :
#     cnt += 1
#     if cnt == 5 :   # 5 건너뛴다. continue 아래의 실행문들은 실행되지 않고 바로 다음 루프로 넘어간다.
#         continue
#     print(cnt)
#     if cnt == 10 :
#         break

'''
순회 - for
 iterator : 순회 객체
 iterable 객체 : 순회 가능한 객체 - 문자열, 튜플, 딕셔너리, 리스트, 셋 등
'''
# lst = ['a', 'b', 'c', 'd']
# for i in lst :
#     print(i)
# print()
# for j in 'haha' :
#     print(j)

# dict_for = {1:2, 3:4, 5:6}
# for i in dict_for : # key 가 순회된다.
#     print(i)
#
# for i in dict_for.values() : # value 순회된다.
#     print(i)
#
# for i in dict_for.items() : # key-values fair 가 순회된다.
#     print(i)


'''
☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆
while, for + break + else
 반복문의 수행동안 break 문이 실행되지 않는다면 반복문 직후의 else 가 수행된다
☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆
'''
# i=0
# while i <= 5 :
#     if i == 6:  # while 이 i 가 5 보다 작거나 같은 동안만 수행되므로 break 문이 수행될 수 없다.
#         break
#     print(i)
#     i+=1
# else :
#     print('6에 도달하지 못함.')    # break 문이 수행되지 않으므로 else 절의 실행문이 실행된다.
#
# print()
# for i in ['a', 'b', 'c', 'd'] :
#     if i == 'e' :
#         break
# else :
#     print('e 가 없다.')

'''
☆☆☆☆☆☆☆☆☆☆☆☆☆☆
여러 시퀀스 순회하기 : zip()
 여러 시퀀스의 병렬 순회
☆☆☆☆☆☆☆☆☆☆☆☆☆☆
'''

# day = ['monday', 'tuesday', 'wednesday']
# fruits = ['banana', 'orange', 'peach']
# drinks = ['coffee', 'tea', 'beer']
# dessert = ['tiramisu', 'ice cream', 'pie', 'pudding']
# for a, b, c, d in zip(day, fruits, drinks, dessert):    # 가장 짧은 시퀀스가 끝나면 끝난다. dessert 에서 pudding 가 나오지 않았다.
#     print(a,'- drink :', c ,', fruit :', b, ', dessert :', d)
#
# lower = ['a', 'b', 'c', 'd']
# upper = ['A', 'B', 'C', 'D']
# lst = list(zip(lower, upper))
# print(dict(lst))    # 쉽게 두개의 리스트나 튜플을 섞어 쉽게 딕셔너리를 만들 수 있다.

'''
숫자 시퀀스 - range
 리스트나 튜플과 같은 자료구조를 생성하여 저장하지는 않고 특정 범위의 숫자 스트림을 반환
'''
# print(type(range(5)))
# for i in range(5) :
#     print(i, end=' ')
# print()
# for i in range(1,5) :
#     print(i, end=' ')
# print()
# for i in range(0,5,2) : # 시작, 끝, 스텝
#     print(i, end=' ')
# print()
# print(list(range(5))) # 리스트 생성
# print(list(range(5,0,-1)))  # 거꾸로

'''
컴프리헨션 comprehension 함축
 하나 이상의 이터레이터(ex - for 문)로부터 파이썬 자료구조를 만드는 방법
'''

# 리스트 컴프리헨션
# lst_comp1 = [num**2 for num in range(5)] # [표현식 for 항목 in ]
# print(lst_comp1)
# lst_comp2 = [num**2 for num in range(5) if num % 2 ==0] # [표현식 for 항목 in 시퀀스 if 조건]
# print(lst_comp2)    # 짝수만 제곱하여 리스트로 만든다.
# lst_comp3 = [(str1, str2) for str1 in ['a','b','c'] for str2 in ['A', 'B', 'C']]    # for 문의 중첩도 가능하다.
# print(lst_comp3)
# lst_comp4 = [(num1, num2) for num1 in range(3) for num2 in range(3) if num2 > 1 and num1 >0]    # 조건문도 작동
# print(lst_comp4)
# lst_comp5 = [(num1, num2) for num1 in range(3) if num1 >0 for num2 in range(3) if num2 > 1]     # 복잡도 감소?
# print(lst_comp5)

# 딕셔너리 컴프리헨션
# dict_comp1 = {i : i.upper() for i in ['a', 'b', 'c', 'd']}
# print(dict_comp1)
# word = 'aaabbbbdd'
# dict_comp2 = {i : word.count(i) for i in set(word)} # 시퀀스를 word 대신에 set(word) 를 사용하면 불필요한 연산이 줄어든다. 같은 문자에 대해 count 를 해줘야 하니 낭비.
# print(dict_comp2)

# 셋 컴프리헨션
# set_comp1 = {i for i in range(5) if i%3==1}
# print(set_comp1)

# 제너레이터 컴프리헨션
# gene_comp1 = (i for i in range(5))  # 괄호안에서 컴프리헨션을 사용하면 튜플이 아닌 제너레이터가 생성된다.
# print(type(gene_comp1)) # 제너레이터 객체
# for i in gene_comp1 :   # 제너레이터는 이터레이터에 데이터를 제공하는 방법중 하나
#     print(i)
'''
제너레이터는 재사용이 불가능하다. 메모리에 값을 두고 순회하는것이 아닌 즉석에서 값을 생성하고 순회. 그리고 잊는다.
따라서 재사용 불가.
'''
# print(list(gene_comp1)) # 이미 한번 이터레이터에서 사용된 제너레이터는 재사용이 불가능하다.
# gene_comp2 = (i for i in range(5))
# print(list(gene_comp2)) # 새로운 제너레이터로 리스트 만들기
# print(list(gene_comp2)) # 역시 재사용 불가.

'''
함수
 코드의 재사용을 위해 코드를 하나로 묶어 관리.
  정의 - 호출
'''
'''
None 은 False 와 다르다
is None 을 이용한다.
'''

'''
위치인자
 함수에 전달되는 인자 argument 의 순서대로 함수의 매개변수 parameter 에 복사된다.
'''
# def menu(wine, entree, dessert) :
#     return {'wine':wine, 'entree':entree, 'dessert':dessert}
#
# print(menu('chardonnay', 'chicken', 'cake'))

'''
키워드인자
 매개변수에 상응하는 이름을 인자에 지정
'''
# print(menu(wine='chardonnay', entree='chicken', dessert='cake'))
#
# # 위치인자와 키워드인자를 섞어 사용할때에는 위치인자가 앞에 위치해야한다.
# print(menu('chardonnay', entree='chicken', dessert='cake')) # 위치인자가 앞에 위치
# print(menu(wine='chardonnay', entree='chicken', ='cake')) # 에러 발생
# print(menu(wine='chardonnay', 'chicken', dessert='cake')) # 에러 발생

'''
기본매개변수값
 함수 호출자가 필요한 수만큼의 인자를 전달하지 않으면 매개변수는 기본설정값을 사용한다.
 기본매개변수값은 함수가 정의될때 계산된다. 
'''
# def menu2(wine, entree, dessert='pudding') :
#     return {'wine':wine, 'entree':entree, 'dessert':dessert}
#
# print(menu2('chardonnay', 'chicken'))
#
# def plus(a, b=1+3) :
#     return a+b
# print(plus(3))
#
# def ex_func(a=[1,2,3]) :    # 함수의 매개변수의 기본값이 리스트. 매개변수의 기본값을 함수 정의시에 계산되므로
#     for i in range(len(a)) :    # a 에는 기본값으로 [1,2,3] 리스트가 할당되있는데 이를 조작하면 리스트가 수정되므로
#         a[i] += 1               # 다음 매개변수 기본값을 이용시에 수정된 리스트를 사용하게된다.
#     return a                    # mutable 한 기본값은 값이 수정되게 된다.
# print(ex_func())
# print(ex_func([5,6,7]))
# print(ex_func())    # 똑같이 인자를 전달하지 않았음에도 결과가 달라진다.
#
# def ex_func2(a=3) : # 기본값이 immutable 한 정수. 이 정수에는 수정을 가해도 함수 리턴에 변화가 없음.
#     a+=1            # 예상 : 메모리에 매개변수 기본값을 저장해두고 인자가 주어지지 않으면 기본값 객체를 가져다 사용하는 듯.
#     return a        #   그렇기때문에 immuatable 한 기본값은 수정하면 파라미터가 새로운 객체를 가리키고 함수 수행. 다시 인자를 주지 않으면
# print(ex_func2())   #   다시 파라미터가 기본값 객체를 가리킴. 따라서 immutable한 기본값은 값이 변하지 않음.
# print(ex_func2())
#
# def buggy(arg, result = []) :   # mutable 한 매개변수 기본값
#     result.append(arg)
#     return result
# print(buggy('a'))   # [a]
# print(buggy('b'))   # 의도 [b], 결과 [a, b]
#
# def nonbuggy(arg, result = None) :  # 매개변수 기본값을 mutable 한 값을 사용하지 않고 매번 새롭게 값을 주도록 한다.
#     if result is None :
#         result = []
#     result.append(arg)
#     return result
#
# print(nonbuggy('a'))
# print(nonbuggy('b'))

'''
예상?
매개변수 기본값이 mutable 할때와 immutable 할때 :
매개변수 기본값을 X 라고 하자. 메모리에 X 가 위치하고 함수에 인자가 전달되지 않는다면 매개변수는 메모리에서 X 를 찾아가
X 를 참조. 이 X 를 수정을 할때 X 가 mutable 한 타입이라면 메모리의 X 가 직접 수정이 되어서 다음 함수호출에서 X 를 사용하면
수정된 X 가 사용됨. 따라서 호출때마다 기본값 X 가 달라짐. X 가 immutable 한 타입이라면 X 에 수정을 하면 X 가 수정된 결과가
새로운 Y 로 메모리에 올라가고 파라미터는 Y 를 이용하여 함수를 수행. 다음 함수 호출때는 다시 Y 가 아닌 X 를 참조하므로
매개변수 기본값을 사용하는 함수호출에서 결과같이 변함없다.
'''
'''
위치인자 모으기 : *
 함수의 매개변수에 * 애스터리스크 를 사용하면 위치매개변수들을 튜플로 묶는다.
'''
#
# def print_args(*args):
#     return args
#
# print(print_args(1,2,3,4,5,6))  # tuple 이 반환된다.
# print(type(print_args(1,2,3,4,5)))  # tuple 타입
# print(print_args()) # 빈 튜플
#
# def print_args2(a, b, *args):   # 위치 매개변수에 해당하는 2번째 까지의 인자들을 제외하고 모든 인자를 튜플로 묶음
#     print(a)
#     print(b)
#     print(args)
#
# print_args2(1,2,3,4,5,6,7)

'''
키워드인자 모으기 : **
 함수의 매개변수에 ** 을 사용하면 키워드매개변수들을 딕셔너리로 묶는다.
'''

# def make_dict(**kwargs):
#     return kwargs
#
# print(make_dict(a='A', b='B', c='C'))   # 키워드매개변수와 인자가 쌍을 이뤄 딕셔너리가 된다.
#
# def tup_and_dict(*args, **kwargs) :
#     print(args)
#     print(kwargs)
# tup_and_dict(1,2,3,a='ha',b='ho')   # 둘을 동시에 사용할때는 위치인자모으기, 키워드인자모으기 순으로.

'''
docstring
'''
# def echo(str) : # 함수 바디 시작부분에 함수에 대한 설명을 붙이는데 이를 docstring 이라 한다. 여러줄도, 한줄도 좋다.
#     '''
#     echooooooooo chooooo chooo choo cho c...
#     :param str: string
#     :return: None
#     '''
#     print(str, str, str, str)
#
# echo('ho')
# print()
#
# help(echo)  # help(함수명) : help() 함수로 docstring 을 출력할 수 있다.
# print('-------------')
# print(echo.__doc__) # 서식없이 출력

'''
함수 이것저것
'''
# def answer():
#     print(42)
# def run_something(func) :
#     func()
# run_something(answer)   # 함수를 인자로 받아 실행한다. 함수호출시 괄호가 함수를 호출한다는 의미이고
#                         # 함수명은 그저 객체로서의 함수를 가리키는 이름이다.
# print(type(run_something), type(answer))    # <class 'function'>
#
# def returnnnn(*args):
#     print(args)
# returnnnn((1,2,3,4))    # 나열된 값이 아닌 튜플 자체가 들어가면 튜플이 하나의 요소로 취급됨
#
# def run_something2(func, *args) :   # 위치매개변수 모으기
#     return func(*args)  # 위 예제에서처럼 튜플이 요소로 취급되지 않도록 *args 로 전달한다.
# def summm(*args) :
#     return sum(args)
# print(run_something2(summm, 1,2,3,4,5))
#
# def run_something3(func, **kwargs): # 키워드매개변수 모으기
#     return func(**kwargs)
# def dictttt(**kwargs):
#     print(kwargs)
# run_something3(dictttt, a=1, b=2, c=3)

'''
내부함수
 함수 내부에 또 다른 함수를 정의. 함수 내부에서의 코드 중복을 피하기 위해 사용.
 특정한 함수 내부에서만 여러번 사용되는 코드를 중복 없이 수행할때 이용.
'''
#
# def outer(a, b) :
#     def inner(c, d):
#         return c+d
#     return inner(a, b)
#
# print(outer(1,3))
#
# def knights(saying):
#     def inner(quote):
#         return 'We are the knights who say : "%s"' % quote
#     return inner(saying)
#
# print(knights('치킨!'))

'''
클로져 closure
 다른 함수에 의해 동적으로 생성되는,
 바깥함수로부터 생성된 변수값을 저장하고 변경하는 함수.
'''

# def knights2(saying):
#     def inner2():
#         return 'We are the knights who say : "%s"' % saying
#     return inner2
#
# # knights2 함수는 내부 함수인 inner2 를 호출하지 않고 함수 자체를 반환함
# # inner2 함수는 인자를 전달받지 않지만 knights2 함수의 매개변수 saying 의 값에 접근가능함
# # 결과적으로 knight2 함수는 saying 의 값을 담은 inner2 함수 자체를 반환하게 됨
#
# a = knights2('피자!')
# b = knights2('치킨!')
# print(type(a), type(b)) # 둘다 함수
# print(a,b)
# c = a()
# d = b()
# print(c,'\n',d) # 다른 값을 저장한 결과값이 나온다.

'''
익명함수 lambda()
 람다함수는 단일문으로 표현되는 익명함수.
'''

# lst = ['ho', 'ha', 'he']
#
# def edit_story(words, func) :
#     for word in words:
#         print(func(word))
# def enliven(word):
#     return word.capitalize() + '!'
# edit_story(lst, enliven)
#
# # 새로 함수를 정의하지 않고 임시적인 함수를 사용할 수 있다.
# edit_story(lst, lambda word : word.capitalize()+'!')
#
# # 조건문도 가능하다.
# lam_func = lambda x : x+1
# print(lam_func(3))
# lam_func2 = lambda x : x if x>0 else -x
# print(lam_func2(3), lam_func2(-3))

'''
제너레이터
 파이썬의 시퀀스를 생성하는 객체. 전체 시퀀스르 생성해 메모리에 올릴 필요 없이 순회 가능.
 이터레이터에 대한 데이터 소스로 자주 사용된다. range() 가 그 예.
'''
# # 제너레이터는 순회할때마다 마지막으로 호출된 항목을 기억하고 다음 값을 반환.
# print(sum(range(1,11)))
#
# # 복습 : 제너레이터 컴프리헨션 ex) (i for i in ['a', 'b', 'c'])
# print((i for i in ['a', 'b', 'c']))
# a = (i for i in ['a', 'b', 'c'])
# for i in a :
#     print(i)
#
# # 제너레이터 컴프리헨션 코드가 복잡해진다면 제너레이터 함수를 사용한다.
# def my_range(first = 0, last = 10, step = 1):
#     number = first
#     while number < last :
#         yield number
#         number += step
#         # 리턴절 없음
#
# print(my_range)
# ranger = my_range(1, 5) # 제너레이터 함수에 인자를 넘겨주고 반환받는데 리턴절 없음.
# print(ranger)   # 제너레이터
#
# for x in ranger :   # 순회가 가능해진다.
#     print(x)

'''
데커레이터
 이미 정의된 함수를 수정할때 사용.
 완전히 정의를 수정하기보다는 정의된 함수들에 똑같은 기능을 추가하는 용도로 사용한다.
'''

# def decorator_func(f):
#     def return_func(*args, **kwargs):
#         print('func_name :', f.__name__)
#         print('position arguments :', args)
#         print('keyword arguments :', kwargs)
#         result = f(*args, **kwargs)
#         print('result =', result)
#         return result
#     return return_func
#
# def add_func(a,b):
#     return a + b
#
# add_func = decorator_func(add_func)
# add_func(3, 4)
# add_func(a=3, b=4)
#
# print()
#
# @decorator_func # 함수 정의 위에 @데코레이터_이름 으로 데코레이터를 적용시킬 수 있다.
# def mult_func(a, b):
#     return a * b
#
# mult_func(3,4)
#
# print()
# def square_it(f) :
#     def return_func(*args, **kwargs):
#         result = f(*args, **kwargs)
#         return result ** 2
#     return return_func
#
# @square_it
# def mult_func(a, b):
#     return a * b
#
# print(mult_func(3,4))
#
# # 데코레이터를 중첩시킬수도 있다. 이때는 함수정의와 가장 가까운(바로 위의) 데코레이터부터 적용된다.
# print()
# @decorator_func
# @square_it
# def mult_func(a, b):
#     return a * b
#
# print(mult_func(3,4))   # 두수의 곱을 제곱한 후 인자, 결과를 출력하므로 결과와 출력이 일치
#
# print()
# @square_it
# @decorator_func
# def mult_func(a, b):
#     return a * b
#
# print(mult_func(3,4))   # 인자, 결과를 출력하고 제곱을 하므로 결과와 출력이 일치하지 않음.

'''
def 라는 구문이 "함수 객체를 생성하는" 구문이다.
내부함수, 클로져, 데코레이터함수 모두 내부에 def 구분을 포함하는데
함수가 수행될때 내부의 def 구문이 실행되며 새로운 함수 객체를 메모리에 올리게된다.
그렇기때문에 클로져나 데코레이터가 바깥함수의 데이터를 가질 수 있는 것.
매개변수로 인자를 받는 것이 아니기때문에 바깥함수의 데이터를 변수로 가리키는 것.
그리고 내부함수, 클로져, 데이코레이터에 의해 만들어진 함수가 리턴될때 참조했던 상위네임스페이스의 변수는
따로 저장하여 보관하므로 그 값을 계속 유지할 수 있게된다.
'''

'''
4.10
네임스페이스와 스코프

 네임스페이스의 이름은 유일. 이름은 네임스페이스에 따라서 다른 의미를 갖는다.
 메인프로그램의 네임스페이스 : 전역변수
 각 함수의 네임스페이스 : 각각의 함수만의 네임스페이스 생성
'''

# # 예제 1. 함수 내부에서 전역변수에 접근
#
# global_var = 3
#
# def func1():
#     print(global_var)
# func1() # 전역변수인 global_var 를 출력한다. 가능하다.
#
# # def func2():
# #     '''
# #     의도 : 전역변수 global_var 의 값을 출력하고, 값을 수정하고, 다시 출력하려고 함.
# #     실제 : 아직 값이 할당되지 않은 gloabl_var 의 값을 출력하고 그 후에야 값을 배정, 다시 출력
# #
# #     UnboundLocalError: local variable 'global_var' referenced before assignment
# #
# #     의도와 다르게 함수가 정의될때 'global_var=5' 실행문이 있으므로
# #         함수의 지역변수 globar_var 를 참조하게됨.
# #      따라서 값이 배정되기 전에 globar_var 를 참조하기때문에 에러가 발생.
# #     '''
# #     print('before modify :',global_var)
# #     global_var = 5
# #     print('after modify :', global_var)
# # func2()
#
# # 예제 2. 함수내에서 전역변수에 접근하려면?
#
# def func3():
#     global global_var   # 전역변수인 global_var 에 접근하겠다고 명시적으로 global 키워드를 작성한다.
#     print('before modify :', global_var)
#     global_var = 5
#     print('after modify :', global_var)
#
# func3()
#
# # locals() & globals()
#     # locals() : 로컬 네임스페이스의 내용이 담긴 딕셔너리 반환
#     # globals() : 글로벌 네임스페이스의 내용이 담긴 딕셔너리 반환
#
# def func4():
#     A = 'a'
#     B = 'b'
#     C = 'c'
#     print(locals())
#
# a = 'A'
# b = 'B'
# c = 'C'
#
# func4() # 로컬변수들이 담긴 딕셔너리가 출력된다.
# print(globals())    # 선언된 함수, docstring, 전연벽수, __name__ 등의 변수 등 별게 다 들어있는 딕셔너리 출력.

'''
4.10.1 이름에 _ 와 __ 사용
 
 __??__ 형태의 변수명은 예약된 키워드. 변수선언시 사용하지 않는다.
    ex)
        함수의 이름 : function.__name__
        모듈의 이름 : __name__ : 특히 메인 모듈(임포트가 아닌 직접 실행되는 모듈) 은 __name__ == '__main__'
        함수의 docstring : functions.__doc__
'''
#
# def underbar_var():
#     '''
#     docstring~~
#     '''
#     print(underbar_var.__name__)
#     print(underbar_var.__doc__)
#
# underbar_var()  # 차례로 underbar_var의 함수명과 docstring 이 출력된다.

'''
4.11 에러처리하기
    
    파이썬에서는 에러를 예외 exception 이라고 한다.
    실패할 가능성이 있는 코드를 수행할때 예외를 처리하기 위해 에외처리가 필요하다.
    예외가 발생할때 이를 처리하여 프로그램을 중단시키지 않거나,
    처리하지 못하더라도사용자에게 어떤 문제가 발생했는지를 알리는게 목적이다.
'''

# short_lst = [1,2,3]
# # print(short_lst[5]) # IndexError: list index out of range 가 발생한다.
#
# # 예외처리 1. 모든 종류의 예외 처리
#     # 예외처리를 하는 경우. try 절의 실행문에 에러가 발생하지 않는다면 except 절을 생략하고
#     # 에러가 발생하는 경우 except 절의 실행문을 수행한다.
# try :
#     print(short_lst[5])
# except :    # 별다른 표기가 없이 except : 만 있다면 모든 종류의 예외를 처리한다.
#     print('exception 발생')
#     print('0 ~ 2 사이의 인덱스를 입력하세요.')
#
# # 예외처리 2. 특정 종류의 예외 처리 - 예외의 종류마다 다른 처리를 한다.
#     # except 특정종류의예외 as 예외를참조할변수명 :
# try :
#     n = input('인덱스를 입력 : ')
#     print(short_lst[int(n)])
# except IndexError as ind_e :
#     print(ind_e, ': 0~2 범위의 index 를 입력하세요.')
# except ValueError as val_e :
#     print(val_e, ': 숫자형 값을 입력하세요.')
#
#     # 리스트의 인덱스 범위 밖의 값을 입력하면 IndexError : list index out of range 발생
#     # 숫자형이 아닌 값을 입력하면 int(n) 에서 ValueError : invalid literal for int() ... 발생
#     # 위 두개의 예외를 각각 다른 except 절로 처리하여준다.

'''
4.12 예외 만들기

다양한 예외는 이미 파이썬 표준 라이브러리에 정의되어있음. ex) ValueError, IndexError, ...
표준 라이브러리에 정의된 상황 외에도 사용자가 직접 예외를 정의할 수도 있다.
'''
# 예외는 클래스. Exception 클래스의 자식이다.

# 예제. 예외 만들기 - 숫자가 입력되면 예외 발생시키기

class IHateNumber(Exception) :
    pass    # 어떤 행동도 안한다.

# while True :
#     n = input('숫자는 입력하지 마세요. : ')
#     if n.isnumeric() :
#         raise IHateNumber(n)
#     print(n)

    # 예외 클래스에 n 을 인자로 넘겨줄때
    # 숫자 3을 입력하면 __main__.IHateNumber: 3 과 같이 정의한 예외와 함께 어떤 값을 입력했는지 나온다.

try :
    n = input()
    if n.isnumeric():
        raise IHateNumber(n)
except IHateNumber as e :
    print(e)

    # 예외 클래스에 별도의 코드 작성이 없다면 예외를 발생시킬때 넘겨주는 인자가 예외의 내용이 된다.