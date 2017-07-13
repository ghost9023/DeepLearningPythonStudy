'''
파이썬은 모든 것이 객체.
모든 객체가 강타입으로 타입을 바꿀 수 없다.
변수이름은 단지 객체를 가리킬뿐이다 :
'''

print(type(99.9))   # type : 객체의 타입 반환

'''
변수명은 a-z, A-Z, 0-9, _ 만 사용가능.
숫자로는 시작할 수 없다.
에약어도 사용할 수 없다.
'''

# print(06) # 0으로 시작할 수 없다.

print(divmod(9, 5)) # (1, 4) - 나눗셈의 몫, 나머지를 반환

print(10, 0b10, 0o10, 0x10)   # 0b : 2진수, 0o : 8진수, 0x : 16진수
    # 10 2 8 16

'''
형변환
'''
print(int(True), int('-10'), int(88.88), int(1e+1))
# print(int('1e+1'))    # 이건 안된다.
print(float(True), float('-10'), float(88), float('1e-1'))  # 여기서는 된다.
print(str(123))

'''
문자열
'''
print('\n############### 문자열 ################\n')
print('"큰따옴표"', "'작은따옴표'")
print('''
3개는
여러줄''')

base=''
print(base+'빈문자열은 base')

# \n : 개행, \t : 탭, \' | \" | \\ : \ 다음의 기호들이 기호 그 자체로 인식.

print('ho' 'ho','ho')    # 쉼표로 구분하지 않는다면 공백없이 합쳐진다.
print('Henny'.replace('H', 'P'))    # replace : 문자 교체. 교체할문자, 교체될문자, 횟수
a='abcdefghijklmnopqrstuvwxyz'
print(a[:])
print(a[::2])   # 슬라이싱은 [start:end:step]. start, end 의 공란은 전체를 의미한다.
print(a[:-1])
print(a[::-1])  # step 이 음수이면 거꾸로 출력

a='a,b,c,d,e,f,g'
b=a.split(',')
print(b)    # 구분자 기준으로 분리 후 리스트로 반환. 구분자를 정하지 않으면 공백을 기준으로함.
c=','.join(b)
print(c)    # 리스트를 연결자로 연결하여 문자열로 반환.
# 기타
# startswith, endswith : 특정 문자열로 시작, 끝
# find, rfind : 특정 문자열이 처음 발견되는, 마지막으로 발견되는 인덱스
# count : 특정 문자열의 출현 횟수
# isalnum : 문자와 숫자로 이루어졌는지 확인
# strip : 특정 문자열을 문자열 좌우 끝에서 삭제
# capitalize : 첫 단어의 첫 글자를 대문자로
# title : 각 단어의 첫 글자를 대문자로
# upper, lower : 전체를 대문자, 소문자로 변환
# swapcase : 대소문자를 서로 전환한다
# center, ljust, rjust : 지정한 필드 크기에서 문자열을 가운데, 좌, 우 정렬한다.