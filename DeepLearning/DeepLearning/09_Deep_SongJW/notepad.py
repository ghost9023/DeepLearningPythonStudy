def deco_func(func) :
    def funcfunc(*args):
        print('배고프다 그래서')
        func(*args)
        print('그래도 배가 고팠다.')
    return funcfunc

@deco_func
def 최지원(food):
    print('최지원은',food,'을/를 먹었다.')

def 최지원2(food):
    print('최지원은', food, '을/를 먹었다.')

최지원('김밥')
print()

좀더진화한최지원=deco_func(최지원2)
좀더진화한최지원('피자')