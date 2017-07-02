''''''
'''
딥러닝스터디 1주차: 퍼셉트론과 신경망

◈ 목차
1.퍼셉트론의 개념
 1) 퍼셉트론의 개념
 2) 가중치 
 3) 편향

2. 논리회로
 1) 논리합과 논리곱
 2) AND게이트, NAND게이트, XOR게이트

3. 다층 퍼셉트론 
 1) 다층 퍼셉트론의 개념
 2) 게이트 조합 
 3) 논리 게이트 구현

4. 단층 신경망
 1) 신경망의 개념
 2) 활성 함수
 3) 다차원 배열의 계산
 4) 신경망 구현하기 
 5) 예제 데이터셋 사용해보기

시작하기에 앞서서 딥러닝과 인공신경망의 정의에 대해 간략하게 알아보면 인공 신경망은 인간의 뇌가 패턴을 인식하는 방식을 모사한 알고리즘입니다.
인공 신경망은 시각, 청각 입력 데이터를 퍼셉트론이나 분류, 군집을 이용하여 해석하는데, 이렇게 해석한 결과 이용하면 이미지, 소리, 문자, 
시계열 데이터에서 특정 패턴을 인식할 수 있습니다.인공 신경망을 이용하면 각종 분류(classification) 및 군집화(clustering) 가 가능합니다.
지금부터 자세히 살펴보겠지만, 단순하게 표현하면 분류나 군집화를 원하는 데이터 위에 여러 가지 층(layer) 을 얹어서 원하는 작업을 하게 됩니다.
각 층에서는 라벨링이 되어있지 않은 데이터를 서로 비교하여 유사도를 구해주거나, 라벨링이 되어있는 데이터를 기반으로 분류기를 학습하여 
자동으로 데이터를 분류하도록 할 수 있습니다.(구체적으로 이야기하면, 인공 신경망으로 특징을 추출하고 그 특징을 다시 다른 기계학습 알고리즘의
입력으로 사용하여 분류나 군집화를 할 수 있습니다.즉, 심층 신경망을 전체 기계학습 시스템의 구성 요소로 생각하면 됩니다.
여기서 전체 시스템이란 강화학습(Reinforced learning), 분류 및 회귀를 말합니다.) 심층 신경망은 신경망 알고리즘 중에서 여러 개의 층으로 이루어진
신경망을 의미합니다.보통 신경망은 입력층, 은닉층, 출력층으로 구분되는데 총 층수의 합이 4 이상인 경우를 심층 신경망(Deep Neural Network) 라고
합니다.이 심층 신경망을 이용한 기계학습을 딥러닝 이라고 합니다.심층 신경망의 한 층은 여러 개의 노드로 이루어져 있습니다.노드에서는 실제로 연산이
일어나는데, 이 연산 과정은 인간의 신경망을 구성하는 뉴런에서 일어나는 과정을 모사하도록 설계되어있습니다.노드는 일정 크기 이상의 자극을 받으면
반응을 하는데, 그 반응의 크기는 입력 값과 노드의 계수(또는 가중치, weights)를 곱한 값와 대략 비례합니다.일반적으로 노드는 여러 개의 입력을 받으며
입력의 개수만큼 계수를 가지고 있습니다.따라서 이 계수를 조절함으로써 여러 입력에 다른 가중치를 부여할 수 있습니다.최종적으로 곱한 값들은 전부 더해지고
그 합은 활성 함수(activation function)의 입력으로 들어가게 됩니다.활성 함수의 결과가 노드의 출력에 해당하며 이 출력값이 궁극적으로 분류나 회귀 분석에
쓰이게 됩니다.심층 신경망을 이용해보고 싶다면 우선 어떤 문제를 해결하고 싶은지 생각해 보는 것이 좋습니다.즉 어떤 분류를 하고 싶은지, 
그리고 내가 어떤 정보를 취할 수 있는지를 정해야 합니다.예를 들면 이메일 데이터를 스팸과 스팸 아님으로 분류한다든지, 고객을 친절한 고객과 악덕고객, 
불만이 많은 고객과 만족하는 고객으로 분류할 수 있습니다.이렇게 어떤 분류를 원하는지 정한 뒤엔 분류에 필요한 데이터를 가지고 있는지 생각해 보아야
합니다.예를 들어 이미 스팸과 스팸 아님으로 라벨링이 된 이메일 데이터가 있는지, 없다면 내가 직접 데이터셋을 만들 수 있는지를 고민해야 합니다.
또, 이 데이터로 원하는 라벨링이 과연 가능한 것인지도 생각해봐야 합니다.예를 들어, 고 위험군에 속하는 암 환자들을 분류하는 알고리즘을 만들기 위해서는
우선 암 환자와 아닌 사람의 데이터가 필요합니다.데이터는 사람들의 나이, 흡연 습관 같은 쉬운 특징이나 일일 운동량, 온라인 활동 로그 같은 간접적인 
특징 등 무엇이든 가능합니다.그러나 사람들의 건강과 관련된 개인 정보가 전혀 없는 상태라면 아무리 좋은 알고리즘이 있어도 암을 예측하기는 어려울 것
입니다.필요한 데이터가 있다면 이제 인공 신경망이 사람들의 암 발병률을 예측하도록 학습할 수 있습니다.즉, 암에 걸린 / 걸리지 않은 사람들이 각각
어떤 행동 패턴을 갖는지, 어떤 것을 기준으로 어떻게 분류하면 되는지를 신경망이 학습하도록 하는 것 입니다.가지고 있는 데이터로 학습이 잘 되었다면
이젠 사람들의 행동 패턴을 이용해 그 사람들이 암에 걸릴 확률을 예상할 수 있습니다.같은 원리로 다양한 분류 작업이 가능합니다.예를 들어 소셜 데이팅 서비스를
기획한다면 서로 잘 어울릴 수 있는 상대를 골라주거나, 신인 중에서 장래가 유망한 운동선수, 회사에 도움이 될 직원 등을 분류하는 것이 가능합니다.
물론 작업마다 다른 데이터가 필요합니다.예를 들어 사람들의 취미, 나이, 사회 활동이나 운동선수의 운동량, 체형, 직원의 근로 실적 등을 이용할 수 있습니다.이
딥러닝을 하기 위해 차례차례 퍼셉트론부터 시작해서 단층 신경망, 다층 신경망, 심층 신경망으로 차근차근 설명해 나아가겠습니다.

1.
퍼셉트론

1) 퍼셉트론의 개념

1957 년 프랑크 로젠블라트가 고안한 알고리즘으로 신경망의 기원이 되는 알고리즘입니다.사람의 시각 인지 과정에서 영감을 얻어서 퍼셉트론 이론을
정립한 다음 이를 이용해서 인공적인 시각 인지과정을 구현하는데 성공했습니다. 단층 퍼셉트론은 다수의 신호를 입력받아 하나의 신호를 출력합니다.
이 신호를 보통 0 과 1 의 두 가지 값을 가지게 되는데 보통 각각의 케이스에 대해 O와 X의 논리를 0 과 1 로 전달합니다.단층  퍼셉트론의 기본 원리는
아래의 원리가 전부입니다.

2) 가중치{여러 개의 Input
값을
입력받고
각각의
가중치를
곱해서
계산
한
뒤
더해준
다음
출력
값을
거치는
과정을
거칩니다.가중치는
각각의
입력신호(input)
에
따라
모두
다르며
고유한
값입니다.퍼셉트론은
이
가중치와
입력
값의
곱의
총
합이
일정
기준을
초과하면
1, 이하면
0
을
출력해줍니다.여기서
1
과
0
을
출력하는
기준이
되는
값을
임계
값이라고
하고
보통
로
표현합니다.이
과정을
수식으로
표현하면
아래와
같습니다.아래와
같이
퍼셉트론이나
신경망의
뉴런을
활성화
시키는
함수
혹은
방정식을
활성
함수(Activation
Function) 라고
합니다.

[수식1.단층
퍼셉트론의
활성
함수]


{
    즉
가중치와
입력
값의
곱의
총
합이
임계
값을
넘기면
1, 그렇지
않으면
0
을
출력
값
는
가지게
됩니다.여기서
가중치는
각
신호(입력
값, input)가
결과에
주는
영향력의
크기라고
생각
할
수
있습니다.왜냐하면
동일한
입력
값이
입력되었을
경우
가중치가
크면
클수록
해당
신호가
가지는
값이
더
커지기
때문입니다.예를
들어
입력
값
3
개
이고
각각의
가중치가
인
단층
퍼셉트론을
예로
들면
이
퍼셉트론의
출력
값은
다음과
같습니다.

[수식2.활성
함수의
예]

이
퍼셉트론에
입력
값으로
전부
1(1, 1, 1)
을
입력하는
경우
임력
값의
합은
입니다.이
경우
각
입력
값의
가중치는
임계
값에
대한
지분으로
생각해
볼
수
있습니다.따라서
가중치가
크면
클수록
해당
가중치의
입력
값은
임계
값에
대한
영향력이
더
크다고
할
수
있습니다.
    위에서
소개했던
퍼셉트론의
구조(그림1)
에서
각각의
동그란
원을
뉴런
혹은
노드라고
부릅니다.

    .
3) 편향

위에서
소개했던
수식1.에서
임계
값
를
로
치환하고
양변에
빼주게
되면
아래의
식으로
바뀝니다.

{

    [수식3.편향이 포함된 활성 함수]

처음
식과
바뀐
식은
기호
표기만
바뀌었을
뿐
의미는
같습니다.바뀐
식에서
를
편향(Bias)
이라고
합니다.나머지
가중치와
입력
값은
동일합니다.바뀐
식의
관점에서
해석해보면
퍼셉트론은
입력
값에
가중치를
곱한
값과
편향을
합하여
그
값이
0
을
넘으면
1
을
출력하고
그렇지
않으면
0
을
출력하게
합니다.여기서
가중치의
곱과
입력
값
그리고
편향의
총
합을
net값
이라고
합니다.이
과정을
파이썬에서
실행해보면
다음과
같습니다.

[net값의
계산]
# 식2.1와 식2.2를 계산import numpy as np# 입력값x = np.array([0, 1])# 가중치w = np.array([0.5, 0.5])# 편향b = -0.7print(w*x)# [ 0.   0.5]print(np.sum(w*x))# 0.5print(np.sum(w*x) + b)# -0.2

단층
퍼셉트론이
가지는
구조는
이다(yes)
혹은
아니다(no)
와
같은
논리회로의
구조와
같습니다.따라서
이를
이해하고
사용하기
위해
논리회로에
대해
알아보겠습니다.

2.
논리회로

1) 논리합과
논리곱

Boolean
Logic(Boolean
함수) 라는
것이
있습니다.Boolean이
낯익으실
수
있습니다.True, False를
파이썬에서
Bool형
데이터라고
부르던
것을
보셨을겁니다.불
함수는
수학자
조지
불의
이름을
딴
함수로
논리곱(AND)
이나
논리합(OR)
과
같은
단순
논리
함수를
의미합니다.

    우선
논리곱이란
AND
관계를
의미합니다.예를
들면 ‘야채를
먹고(AND)
배가
고픈
경우에만
푸딩을
먹을
수
있다’ 라는
문장이
존재한다고
가정합니다.이
문장은
푸딩을
먹는
조건을
담고
있으며
푸딩을
먹는
조건이
1.
야채를
먹는
것, 2.
배가
고픈
경우인
것
총
두
가지
존재하는
것을
알
수
있습니다.이
문장에
의하면
야채를
먹는
경우 + 배가
고픈
경우 -> 푸딩을
먹을
수
있다.로
해석할
수
있습니다.이처럼
두
가지
조건이
모두
충족되면
푸딩을
먹을
수
있고
이는
두
조건이
모두
참
이라는
것을
의미합니다.따라서
논리곱이
참이
되는
것은
모든
조건이
참인
경우
논리곱이
참이
됩니다.야채만
먹는
경우, 혹은
배만
고픈
경우는
한쪽만
참인
경우이므로
논리곱이
참이
될
수
없고
거짓인
경우
입니다.따라서
푸딩을
먹을
수
없습니다.야채를
먹었지만
배가
고프지
않다면
푸딩을
먹을
수
없고
야채를
먹지
않았으면
배가
고프더라도
푸딩을
먹을
수
없다는
것을
의미합니다.

    논리합이란
OR
관계를
의미합니다.논리합은
논리곱과는
다르게
일부만
충족되어도
참이
됩니다. ‘야채를
먹거나(OR)
배가
고픈
경우
푸딩을
먹을
수
있다’ 라는
문장을
보시면
이해가
되실
겁니다.야채를
먹으면
배가
고프건
안고프건
푸딩을
먹을
수
있고
야채를
먹지
않았어도
배가
고프다면
푸딩을
먹을
수
있습니다.불
논리함수는
회로의
기본이
되기
때문에
컴퓨팅에서
매우
중요한
개념이고
신경망에서도
핵심
개념
중
하나입니다.

2) AND게이트, NAND게이트, OR게이트, XOR게이트

위에서
논리곱과
논리합, 불
함수에
대해서
알아보았습니다.신경망을
하기
전에
신경망에
대해서
생각해보면
신경망도
입력
값이
들어오면
출력
값이
나오는
하나의
회로라고
볼
수
있습니다.때문에
불
함수가
중요한
것입니다.이
불
함수로
이루어진
입출력
과정을
게이트라고
합니다.따라서
AND
논리회로를
통한
입출력을
AND게이트, Not
AND
논리회로를
통한
입출력을
NAND
게이트, OR
논리회로를
통한
입출력을
OR게이트, Exclusive
OR
논리회로를
통한
입출력을
XOR게이트라고
합니다.
    AND게이트는
논리곱에서
입력
값이
모두
참인
경우
출력
값을
참으로
출력합니다.Not
AND게이트는
AND게이트의
정반대
논리로
입력
값이
모두
참이면
출력
값으로
거짓을
출력하고
나머지
경우는
모두
참을
출력합니다.OR게이트는
논리합에서
입력
값
중
어느
하나라도
참인
경우
출력
값을
참으로
출력합니다.XOR게이트는
입력
값이
서로
다른
경우
참이
됩니다.이렇게
문자로
논리
게이트를
표현하면
이해하기가
매우
난해합니다.이를
더
쉽게
이해할
수
있도록
입력
신호와
출력
신호의
대응
표를
만들었고
이를
진리표라고
합니다.각
논리
게이트의
진리표는
아래에서
확인하실
수
있습니다.

    입력
A
입력
B
출력
값
0
0
0
0
1
0
1
0
0
1
1
1

[표1.AND게이트
진리표]



입력
A
입력
B
출력
값
0
0
1
0
1
0
1
0
0
1
1
0

[표2.NAND게이트
진리표]



입력
A
입력
B
출력
값
0
0
0
0
1
1
1
0
1
1
1
1

[표3.OR게이트
진리표]


입력
A
입력
B
출력
값
0
0
0
0
1
1
1
0
1
1
1
0

[표4.XOR게이트
진리표]

이
모든
논리
게이트는
퍼셉트론으로
정의할
수
있습니다.입력
값이
어떻게
되던
간에
가중치와
임계
값이란
매개
변수만
바꿔준다면
같은
구조의
퍼셉트론에서
AND게이트, NAND게이트, OR게이트, XOR게이트
모두
사용이
가능합니다.이
논리
회로를
파이썬에서
사용자함수를
통해
구현해보겠습니다.함수
내에서
매개변수(가중치, 임계
값)를
초기화
하면서
선언해주고
net값을
계산한
뒤
net
값을
바탕으로
0
이나
1
을
조건문을
통해
리턴하게
해주면
됩니다.

[AND게이트의
구현]
# AND게이트def AND(x1, x2):# 입력값을 받아서 array에 저장	x = np.array([x1, x2]) # 가중치를 array에 저장	w = np.array([0.5, 0.5])   # 편향 선언	b = -0.7# net값 선언	net = np.sum(w*x) + b	if net <= 0:		return 0	else:		return 1


[NAND게이트의 구현]
    # NAND게이트def AND(x1, x2):# 입력값을 받아서 array에 저장	x = np.array([x1, x2])# 가중치를 array에 저장. 구조는 AND게이트함수와 똑같지만 가중치와 편향값이 다르다	w = np.array([-0.5, -0.5])# 편향 선언	b = 0.7# net값 선언	net = np.sum(w*x) + b	if net <= 0:		return 0	else:		return 1

[OR게이트의
구현]

# OR게이트def OR(x1, x2):# 입력값을 받아서 array에 저장	x = np.array([x1, x2]) # 가중치를 array에 저장. 구조는 AND게이트함수와 똑같지만 가중치와 편향값이 다르다	w = np.array([0.5, 0.5])# 편향 선언	b = -0.2# net값 선언	net = np.sum(w*x) + b	if net <= 0:		return 0	else:		return 1

위
세
가지
논리게이트는
net값을
선형방정식인
활성
함수로
표현할
수
있었습니다.따라서
파이썬
사용자함수로
손쉽게
구현할
수
있었습니다.하지만
XOR게이트는
선형방정식으로
표현이
불가능합니다.다음의
그래프들에서
그
이유를
확인할
수
있습니다.

[그림2.선형
활성함수를
사용하는
논리게이트(OR게이트)]

위
그래프는
가중치가
0.5
0.5, 편향이 - 0.5
인
OR게이트의
그래프입니다.가(0, 0)
일
땐
0
을
출력하고(0, 1), (1, 0), (1, 1)
일
땐
1
을
출력합니다.
    활성
함수
는
1
차
선형
방정식이며
이
직선을
통해
입력
값의
쌍들에
대해
0
혹은
1
이
어떨
때
출력되는지
알
수
있습니다.AND게이트와
NAND게이트
그리고
OR게이트의
경우
활성
함수에
의해
명확하게
0(녹색)
과
1(붉은색)
이
구분되어집니다.하지만
XOR게이트의
경우
다릅니다.아래
그래프를
보시면
알
수
있습니다.

[그림4.XOR게이트의
그래프]


위
그래프를
보시면
선형
방정식
하나로는
0
과
1
을
구분할
수
없습니다.이러한
0
과
1
을
구분하려면
비선형
영역이
필요하고
대부분의
단층
퍼셉트론은
위와
같은
XOR게이트처럼
비선형
영역을
구현할
수
없기
때문에
실질적으로
단층
퍼셉트론은
거의
사용되지
않습니다.

    임계값(threshold): 어떠한
값이
활성화되기
위한
최소값을
임계값(임계치)
라고
한다.

    가중치(weight): 퍼셉트론의
학습
목표는
학습
벡터를
두
부류로
선형
분류하기
위한
선형
경계를
찾는
것이다.가중치는
이러한
선형
경계의
방향성
또는
형태를
나타내는
값이다.

    편향(bias): 선형
경계의
절편을
나타내는
값으로써, 선형방정식의
경우
y절편을
나타낸다.

    net값: 입력값과
가중치의
곱을
모두
합한
값으로써, 기하학적으로
해석하면
선형
경계의
방정식과
같다.

    활성
함수(activation
function): 뉴런에서
계산된
net값이
임계치보다
크면
1
을
출력하고, 임계치보다
작은
경우에는
0
을
출력하는
함수이다.이
정의는
단층
퍼셉트론에서만
유효하며, 다층
퍼셉트론에서는
다른
형태의
활성함수를
이용한다.

    뉴런(neuron): 인공신경망을
구성하는
가장
작은
요소로써, net값이
임계치보다
크면
활성화되면서
1
을
출력하고, 반대의
경우에는
비활성화
되면서
0
을
출력한다.

[퍼셉트론
관련
용어
정리]


3.
다층
퍼셉트론

1) 다층
퍼셉트론의
개념

위에서
봤던
XOR게이트의
경우처럼
단층
퍼셉트론의
경우
선형
활성
함수
밖에
사용하지
못하므로
한계점이
많습니다.이를
해결하기
위해
단층
퍼셉트론을
여러
층
쌓은
다음
연결해서
층이
여러
개인
퍼셉트론을
사용합니다.

[그림5.다층
퍼셉트론의
구조]

정확히
정의하자면
다층
퍼셉트론은
입력
노드와
출력
노드
사이에
하나
이상의
중간층이
존재하는
다중
퍼셉트론
구조로
여기서부터
신경망과
굉장히
유사한
형태를
가집니다.구조
자체는
단층
퍼셉트론과
유사하지만
중간층과
각
노드들의
입력, 출력을
비선형으로
함으로써
전체
네트워크의
능력을
향상시켜
단층
퍼셉트론의
여러
가지
단점들을
해결했습니다.입력
노드의
퍼셉트론에
데이터가
입력되면
중간층의
노드를
거쳐서
출력
노드를
통해
값을
출력합니다.과정을
자세히
표현하면
아래와
같습니다.

[그림6.다층
퍼셉트론의
진행
과정]






[그림7.다층 퍼셉트론의 도식화 예]

2) 게이트
조합

단층
퍼셉트론에서
구현이
불가능
했던
XOR게이트는
다층
퍼셉트론을
이용해서
구현할
수
있습니다.앞서서
소개했던
AND게이트, NAND게이트, OR게이트를
적절하게
조합하면
XOR게이트를
구현할
수
있습니다.각
게이트를
회로
기호로
표현하면
아래와
같다.

[그림8.AND
게이트
회로
기호]









[그림9.NAND 게이트 회로 기호]

[그림10.OR
게이트
회로
기호]



회로
기호를
이용해서
XOR게이트를
회로로
구현한
결과는
아래와
같습니다.

[그림11.XOR
게이트
회로
기호]
위
회로도로
구현한
XOR게이트의
진리표를
살펴보면
아래와
같습니다.

0
0
1
0
0
1
0
1
0
1
0
1
1
0
1
1
1
0
1
0

[표5.다층
퍼셉트론으로
구현한
XOR
게이트
진리표]

진리표를
확인해보면
입력
값인
과
가
첫
게이트에
입력되어서
과
가
출력되고
이
과
를
입력
값으로
다시
받아서
마지막
게이트에서
를
출력하게
되는
논리회로를
확인해볼
수
있습니다.[표5]
의
진리표
출력
값
가[표4.XOR
게이트
진리표]의
출력
값과
동일함을
확인해
볼
수
있습니다.따라서
XOR게이트를
파이썬으로
구현하는
것
역시
만들어
놓은
AND게이트
함수와
NAND게이트
함수, OR게이트
함수를
연결해서
사용하면
동일한
결과를
받을
수
있습니다.코드는
아래와
같습니다.

    # XOR게이트def XOR(x1, x2):# 위에서 생성한 NAND 함수를 s1 으로 사용	s1 = NAND(x1, x2)# 위에서 생성한 OR 함수를 s2 로 사용	s2 = OR(x2, x2)# 위에서 생성한 AND 함수를 출력 노드인 y의 함수로 사용	y = AND(s1, s2)	return y



    최종적으로
다층
구조의
XOR
네트워크를
확인하면
다음과
같습니다.

[그림12.다층
퍼셉트론으로
구현한
XOR
게이트
네트워크]

이
다층
퍼셉트론은
뒤이은
3
장의
신경망에서
배우게
될
비선형
함수인
시그모이드
함수를
활성
함수로
사용하면
어떠한
임의의
함수도
표현할
수
있다는
사실이
이미
증명되어
있습니다.다층
퍼셉트론은
다양한
응용
분야에
성공적으로
적용되고
있는
대표적인
신경회로망
모델입니다.하지만
다층
퍼셉트론의
학습에서
나타나는
느린
학습
속도와
지역
극소는
실제
응용문제에
적용함에
있어서
가장
큰
문제로
지적되어왔습니다.따라서
다층
퍼셉트론은
딥러닝의
개념과
뉴런
회로구조를
이해하는
용도로
사용하고
본격적으로
신경망에
대해
배워보도록
하겠습니다.

4.
단층
신경망

서두에서
소개했듯이
딥러닝이란
것은
심층
신경망(Deep
Neural
Network)을
이용하여
지도학습, 비
지도학습, 강화학습을
수행해서
데이터를
분류, 분석, 예측하는
것을
의미합니다.딥러닝에서
사용하는
심층
신경망
기법은
이미지
분석과
NLP(자연어
처리, Natural
Language
Process) 그리고
알파고에
사용되어서
이름이
알려져
있는
CNN(Convolutional
Neural
Network)과
배열과
시간순서가
중요한
경우
사용하는
NLP와
시계열
분석, 음성처리, 신호(패턴)
인식
등에
주로
사용되고
있는
RNN(Recursive
Neural
Network) 등을
꼽을
수
있습니다.이러한
고급
딥러닝
알고리즘
역시
큰
틀에서
보면
다층
신경망(Multi - Layer
Neural
Network)이며
마찬가지로
이름에서
알
수
있듯이
다층
신경망은
단층
퍼셉트론과
다층
퍼셉트론의
관계처럼
단층
신경망(Single - Layer
Neural
Network)에서
시작되었습니다.따라서
딥러닝을
체계적으로
공부하려면
퍼셉트론에서
시작해서
단층
신경망을
거쳐
다층
신경망을
통해
심층
신경망으로
차근차근
나가는
것이
좋은
방법이라고
할
수
있습니다.이번
장에서
배우는
신경망은
단층
신경망이며
여기서
배우게
되는
학습률(Learning
rate)과
활성
함수(Activation
Function)에서
사용하는
시그모이드
함수(Sigmoid
Function), ReLU(Rectified
Linear
Unit) 함수, 소프트맥스
함수(Softmax
Function) 등은
심층
신경망에서도
쓰이는
개념이므로
확실하게
공부하고
가야
합니다.

1) 신경망의
개념

사람의
뇌는
biological
한
뉴런
네트워크로
이루어져
있다고
볼
수
있습니다.
    신경(뉴런)
들은
여러
전기신호를
수상돌기(dendrite)
에서
받습니다.신호들을
소마에서
처리
한
후
축색
돌기(axon)
를
통해
output을
다른
뉴런으로
신호를
전달하는
원리가
인간의
신경계라고
할
수
있습니다.이
신경계의
뉴런구조를
수학적으로
구현한
것이
인공
신경망
이라고
할
수
있습니다.
[그림13.사람의
신경망과
인공
신경망의
비교]


[그림14.사람의 신경망의 시냅스와 인공 신경망의 시냅스 구조 비교]

뉴런의
입력은
다수이고
출력은
하나이며, 여러
신경세포로부터
전달되어
온
신호들은
합산되어
출력됩니다.합산된
값이
임계
값(임계
치, Threshold) 이상이면
출력
신호가
생기고
이하이면
출력
신호가
없는
구조를
가집니다.다수의
Neuron이
연결되어
의미
있는
작업을
하듯이
인공신경망도
노드들을
연결시켜
Layer를
만들고
연결
강도는
가중치로
처리합니다.이처럼
사람의
인지
및
사고
구조를
수학적으로
해석해서
구현한
것이
기본적인
신경망에
대한
개념이라고
할
수
있습니다.

[그림15.신경망의
기본
구조]

[그림15]
에서
볼
수
있듯이
단층
신경망, 다층
신경망, 심층
신경망
모두
기본적으로
이
구조를
가집니다.입력을
받는
입력층, 학습이나
분류가
진행되는
은닉층, 은닉층의
결과를
받아서
출력을
해주는
출력층으로
나뉩니다.이중
은닉층은
사람의
눈으로
볼
수
없기
때문에
은닉층이라고
부릅니다.은닉층이
0
개인
신경망
구조를
단층
신경망, 1
개인
신경망을
다층
신경망, 2
개
이상인
신경망을
심층
신경망이라고
합니다.구조적으로
이전에
배운
퍼셉트론과
큰
차이가
없는
것을
볼
수
있습니다.

[그림16.단층
신경망의
기본
구조]







[그림17.다층 신경망의 기본 구조]

신경망이
퍼셉트론과
다른
점은
편향과
활성함수의
구조가
다른
점입니다.계속해서
어떻게
다른지
배워보도록
하겠습니다.

2) 활성
함수

퍼셉트론에서
배웠듯이
활성
함수, 혹은
활성화
함수(Activation
Function)는
기본적으로
개별
가중치와
입력값의
곱의
합에
편향을
더한
값을
의미합니다.퍼셉트론에서
사용한
것과
같은, [수식3]
과
같이
활성함수를
라고
할
때
처럼
입력값에
따라
0
혹은
1
로
단순하게
딱딱
나뉘는
함수를
계단
함수(Step
Function) 라고
합니다.
# 단순한계단함수def step_funtion(x):    if x > 0:        return 1    else:        return 0

# numpy array(numpy 배열를입력으로받을수있는계단함수import numpy as np


def step_function2(x):    y = x > 0


return y.astype(np.int)  # numpy array를 변환 할 땐 .astype() 메소드를 사용한다.

계단
함수를
파이썬
코드로
짜보면
위와
같습니다.이
계단
함수는
너무
간단하고
단순해서
복잡한
데이터를
입력하는
신경망에
사용할
경우
성능이
좋지
않습니다.따라서
신경망엔
여러
가지
복잡한
활성함수를
선택해서
사용하는데
크게
하이퍼볼릭
탄젠트
함수(Hyperbolic
Tangent
Function, tanh로
표현), 시그모이드
함수(Sigmoid
Function), ReLU
함수(Rectified
Linear
Unit
Function) 등을
사용합니다.현재
딥러닝에서
보편적으로
사용하고
있는
함수는
시그모이드
함수와
ReLU
함수이며
이중에서도
최신
트렌드는
ReLU
함수를
더
많이
사용합니다.ReLU
함수가
타
활성
함수보다
학습
속도와
학습
성능, 학습
효율에서
더
뛰어나기
때문입니다.
활성함수는
생물의
신경시스템에서
시냅스가
전달된
전기
신호의
최소
자극값을
초과하면
활성화되어
다음
뉴런으로
전기
신호를
전달하는
것을
모방하여
신경망에서
각각의
노드(뉴런)
는
전달
받은
데이터를
가중치를
고려해
합산한
값에
활성함수를
적용해서
다음
층에
전달한
다음
이
과정을
반복하여
출력층을
통해
결과가
출력되는
알고리즘
구조를
가집니다.신경망에
작은
값이
입력되었을
땐
활성함수는
출력값을
작은
값으로
막고
일정
값을
초과하면
출력값이
급격히
커지도록
설계되어서
마치
생물의
뉴런과
같이
반응할
수
있도록
합니다.우선
ReLU
함수가
쓰이기
전
까지
널리
쓰였던
시그모이드
함수에
대해
먼저
알아보도록
하겠습니다.

(1)
시그모이드
함수(Sigmoid
Function)

우선
시그모이드
함수를
이해하기
위해서
자연상수가
무엇인지
알아보면
자연상수
e는
자연
상수로
수학, 통계, 화학, 물리
전반에
걸쳐
널리
사용되는
상수입니다.
자연
상수는 = 2.71828182845904523536
인
비
순환소수인
무리수입니다.원주율
3.141592
를
로
쓰는
것처럼
수학자
오일러의
이름의
첫
글자인
E를
따서
기호를
써서
사용합니다.여러
자연적
현상들을
수식으로
설명할
때
이
자연
상수를
통해
대부분
설명할
수
있어서
자연
상수라고
부릅니다.자연
상수는
미적분에서
특이한
성질을
지니고
있는데
아래와
같습니다.

시그모이드
함수의
함수식은
아래와
같습니다.

[수식10.시그모이드 함수]

시그모이드
함수의
계산
결과는
항상
0
과
1
사이에
있습니다.계단함수와
시그모이드
함수를
각각
파이썬
matplotlib를
이용해서
그려본
결과는
아래와
같습니다.
# 계단함수 그래프import matplotlib.pylab as pltdef step_function(x):    return np.array(x > 0, dtype=np.int)x = np.arange(-5.0, 5.0, 0.1)y = step_function(x)plt.plot(x, y)plt.ylim(-0.1, 1.1)plt.show()








[그림18.계단 함수 그래프]

기본적인
계단함수는
0
을
기준으로
출력값이
0
과
1
로
나뉘는
것을
볼
수
있습니다.그래프
형태가
계단처럼
생겨서
계단
함수라고
부르는
것입니다.다음으로
시그모이드
함수를
파이썬
함수로
구현해보고
역시
그래프를
그려보도록
하겠습니다.

# 시그모이드함수 및 그래프import numpy as npimport matplotlib.pylab as pltdef sigmoid(x):# np.exp()는 numpy에서 지원하는 자연상수를 밑수로 가지는 지수함수
return 1 / (1 + np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

[그림19.시그모이드 함수 그래프]

시그모이드
함수에서
시그모이드란
S자
모양이라는
뜻으로
계단
함수가
계단처럼
생겼다고
계단
함수라고
부르는
것처럼
시그모이드
함수가
S자
모양이어서
시그모이드
함수라고
부릅니다.시그모이드
함수와
계단
함수를
한
좌표축에
놓고
비교해서
왜
시그모이드
함수를
사용하는지
보도록
하겠습니다.

[그림20.시그모이드와 계단 함수 그래프]

위[그림20]
에서
볼
수
있듯이
시그모이드
함수는
부드러운
곡선으로
입력에
따라
출력값이
연속적으로
부드럽게
바뀝니다.하지만
계단
함수는
0
을
기점으로
출력값이
0
에서
1
로
확
바뀝니다.또한
시그모이드
함수는
0
과
1
사이의
모든
값을
출력값으로
내보낼
수
있지만
계단
함수는
0
아니면
1
만
내보낼
수
있는
점이
시그모이드
함수와
다른점입니다.바로
이러한
점이
퍼셉트론보다
복잡하고
정교한
신경망구조에서
계단
함수를
사용하지
않는
점입니다.단층
퍼셉트론에서
선형
방정식을
활성함수로
사용지만
다층
퍼셉트론에선
비선형
방정식을
활성함수로
사용한
것과
마찬가지로
신경망도
비선형
방정식을
활성함수로
사용해야만
합니다.그렇기
때문에
시그모이드
함수와
같은
비선형
함수를
사용하는
것입니다.그렇다면
최신
트렌드에서
사용하는
ReLU
함수를
이제
알아보도록
하겠습니다.

(2)
ReLU
함수(Rectified
Linear
Unit)

ReLU
함수에서
Re는
Rectified로 ‘정류된’ 이라는
뜻입니다.이는
전기회로쪽
용어로
특정
값
이하에선
회로가
반응하지
않다가
해당
값
이상에선
값이
커질수록
크게
반응하는
것을
의미합니다.ReLU
함수의
그래프를
보시면
ReLU함수가
왜
ReLU
함수로
불리는지
알
수
있습니다.ReLU
함수의
그래프를
그리기
{
    전
ReLU
함수의
함수식을
살펴보면
수식은
아래와
같습니다.

[수식11.ReLU
함수]

ReLU함수는
0
보다
작은
입력값에
대해선
0
으로
반응하지
않다가
0
이
넘는
값이
입력되면
점점
출력값이
커지는
형태의
함수입니다.이를
파이썬으로
구현하고
matplotlib를
이용해서
그래프로
그리면
다음과
같습니다.
    # ReLU 함수와그래프import numpy as npimport matplotlib.pylab as pltdef ReLU(x):    return np.maximum(0, x)x = np.arange(-5.0, 5.0, 0.1)y = ReLU(x)plt.plot(x, y)plt.ylim(-1, 5)plt.show()

[그림21.ReLU
함수
그래프]

이
ReLU
함수
또한
0
이하의
입력값에서
아예
반응하지
않는
점이
문제가
될
때가
있어서
이를
보완한
Parametric
ReLU
함수라는
것을
사용하기도
합니다.PReLU
함수의
함수식과
그래프는
아래와
같습니다.

{

    [수식11.ReLU 함수]

    [그림22.PReLU
함수
그래프]

본격적으로
신경망에
들어가기에
앞서
현재
딥러닝에
사용하는
활성
함수들에
대해
정리해보도록
하겠습니다.

- 시그모이드
함수: 절대
사용하지
말
것
1) 값이
매우
크거나
작은
경우
함수
값이
포화되기
쉬움
2) 함수
값이
zero - centered가
아님.입력값이
항상
양인
경우
가중치에
대한
함수
미분값이
항상
양이거나
항상
음이
됨.
3) 지수함수
exp()
를
계산하는데
비용과
시간이
많이
듦

이
함수는
함수값을[0, 1]
로
제한시키며(squash)
만약
입력값이
매우
큰
양수이면
1, 매우
큰
음수이면
0
에
다가갑니다.sigmoid는
뇌의
뉴런과
유사한
형태를
보이기
때문에
과거에
많이
쓰였던
activation
함수이지만
지금은
잘
쓰이지
않습니다.그
이유는
아래와
같습니다.gradient를
죽이는
현상이
일어난다(gradient
vanishing.Gradient에
관해선
4, 5
장에
경사
감소법에서
중요하게
다루게
됩니다).sigmoid
함수의
gradient
값은
x = 0
일
때
가장
크며,
| x | 가
클수록
gradient는
0
에
수렴합니다.이는
이전의
gradient와
local
gradient를
곱해서
에러를
전파하는
back
propagation(역전파)
의
특성에
의해
그
뉴런에
흐르는
gradient가
사라져
버릴(vanishing)
위험성이
커집니다.또한
함수값의
중심이
0
이
아닙니다(not zero - centered).이
경우에
무슨
문제가
발생하는가
하면
어떤
뉴런의
입력값(xx)
이
모두
양수라
가정할
때
편미분의
체인룰에
의해
파라미터
w(가중치)
의
gradient는
다음과
같습니다.

    여기서
L은
손실
함수, a는
를
의미합니다.
    이
식에
의해
이고
결론적으로
입니다.
    파라미터의
gradient는
입력값에
의해
영향을
받으며, 만약
입력값이
모두
양수라면
파라미터의
부호는
모두
같게
됩니다.이렇게
되면
gradient
descent를
할
때
정확한
방향으로
가지
못하고, 지그재그로
수렴하는
문제가
발생한다.sigmoid
함수를
거친
출력값은
다음
레이어의
입력값이
되기
때문에
함수값이
not -centered의
특성을
가진
sigmoid는
성능에
문제가
생길
수
있습니다.RNN(Recursive
Neural
Network, 재귀
신경망)에
한해서
특정
이유로
인해
시그모이드
함수를
사용합니다.

- 하이퍼볼릭
탄젠트
함수: 절대
사용하지
말
것
1) 시그모이드
함수의
문제와
동일하다

- ReLU
함수(혹은
max(0, x)
함수): 사용을
권장한다.
1) 함수값의
포화
문제가
없음
2) 계산이
빠름
3) 수렴
속도가
시그모이드, 하이퍼볼릭
탄젠트함수보다
약
6
배
빠름
4) 입력값이
0
보다
작은
경우
함수
미분값이
0
이
되는
단점이
존재함.

    ReLU는
최근
몇
년
간
가장
인기
있는
activation
함수입니다.이
함수는
f(x) = max(0, x)
의
꼴로
표현할
수
있는데, 이는
x > 0
이면
기울기가
1
인
직선이고, x < 0
이면
출력값은
항상
0
입니다.sigmoid나
tanh
함수와
비교했을
때
SGD의
수렴속도가
매우
빠른
것으로
나타났는데
이는
함수가
saturated(함수의
출력값에
한계가
있는
것.시그모이드
함수나
계단
함수는
출력값이
무조건
0
에서
1
이기
때문에
Saturate
하다고
하고
ReLU
함수는
x값이
커질수록
출력값도
커지므로
saturate
하지
않다.)하지
않고
linear하기
때문에
나타납니다.sigmoid와
tanh는
exp()
에
의해
미분을
계산하는데
비용이
들지만, ReLU는
별다른
비용이
들지
않습니다(미분도
0
아니면
1
이다).ReLU의
큰
단점으로
네트워크를
학습할
때
뉴런들이 “죽는”(die)
경우가
발생합니다.x < 0
일
때
기울기가
0
이기
때문에
만약
입력값이
0
보다
작다면
뉴런이
죽어버릴
수
있으며, 더
이상
값이
업데이트
되지
않게
됩니다.

- PReLU
함수: 사용을
권장한다.
1) ReLU
함수의
장점을
그대로
가지고
있다.
2) ReLU
함수의
입력값이
0
보다
작을
때의
약점(4
번)을
보완했다.
3) exp()
함수를
계산해야하므로
시그모이드
함수의
3
번
약점과
비슷한
문제가
있다.

“dying
ReLU” 현상을
해결하기
위해
제시된
함수입니다.ReLU는
x < 0
인
경우
항상
함수값이
0
이지만, PReLU는
작은
기울기를
부여합니다(위[그림23]
을
참조하세요).f(x) = max(ax, x)
로
표현하며
이
때
a는
매우
작은
값입니다(0.01
등).몇몇
경우에
이
함수를
이용하여
성능
향상이
일어났다는
보고가
있지만, 모든
경우에
그렇진
않으므로
직접
경험해보고
판단해야합니다.

- ELU(Exponential
Linear
Unit) 함수: 2015
년에
나온
가장
최신
함수
1) PReLU
함수와
장점
단점이
같다.

    ReLU함수들과의
비교
그림과
공식을
보면
알겠지만
ELU는
ReLU의
임계값을 - 1
로
낮춘
함수를
exp()
로
이용하여
근사한
것
입니다.ELU의
특징은
ReLU의
장점을
모두
포함하고
역시
dying
ReLU
문제를
해결하였으며
출력값이
거의
zero - centered에
가깝습니다.하지만
ReLU, PReLU와
달리
exp()
를
계산해야하는
비용이
듭니다.

    정리하자면
공부하는
과정에선
sigmoid랑
ReLU
함수를
사용해서
공부하고
실제
신경망을
설계할
땐
ReLU와
PReLU
함수를
활성
함수로
사용하는
것이
좋을
것입니다.다만
어떤
때에
ReLU
함수를
사용하고
어떤
때에
PReLU
함수를
사용하는
가는
경험(Trial and Error)
을
통해
얻어낼
수밖에
없습니다.

    단층
신경망과
간단한
다층
신경망에선
간단하게
시그모이드
함수를
활성함수로
사용하지만
후반부로
가면
ReLU
함수를
신경망에
사용할
것입니다.

3) 다차원
배열의
계산

다차원
배열이란
숫자를
한줄(1
차원), 혹은
직사각형이나
정사각형(2
차원), 정육면체나
직육면체(3
차원), 혹은
그
이상의
N
차원으로
나열하는
것을
통틀어
다차원
배열이라고
합니다.파이썬의
Numpy
라이브러리를
이용하면
다차원
배열을
사용할
수
있습니다.아래
코드는
여태까지
사용했던
1
차원
배열입니다.

# 1차원 배열import numpy as npA = np.array([1, 2, 3, 4])print(A)# [1 2 3 4]np.ndim(A) # 배열이 몇 차원인지 확인하는 메소드1A.shape # 배열의 모양을 튜플로 반환하는 메소드(4, )A.shape[0]# 4

# 2차원배열port numpy as npA = np.array([[1, 2], [3, 4], [5, 6]])print(A)# [[1 2]#  [3 4]#  [5 6]]np.ndim(A) # 배열이 몇 차원인지 확인하는 메소드# 2A.shape # 배열의 모양을 튜플로 반환하는 메소드# (3, 2)   # 3행 2열을 의미. 2차원 배열부턴 행렬과 같다A.shape[0]# 3

1
차원
배열은
벡터(Vector)
라고
합니다.벡터는
방향과
크기의
의미를
모두
포함하는
타입으로
스칼라
값들을
순서대로
가집니다.

2
차원
배열에서
볼
수
있듯이
2
차원
배열은
행렬(Matrix)
과
동일한
형태를
가집니다.이는
2
차원
배열간
연산에
행렬
연산을
할
수
있다는
것을
의미합니다.행렬의
내적, 역행렬
등등
중요한
행렬
연산들을
numpy
array를
통해
할
수
있습니다.행렬
연산에
관련한
자세한
사항은
아래
링크로
첨부한
선형대수
강의를
참조하세요.
    https: // www.youtube.com / playlist?list = PLSN_PltQeOyjDGSghAf92VhdMBeaLZWR3

파이썬으로
행렬
내적을
하는
방법은
다음과
같습니다.
import numpy as npA = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])  # 행렬내적을 해주는 numpy 메소드 : .dot()np.dot(A,B)
# array([[19, 22],
#       [43, 50]]) 

행렬
내적을
하기
위해선
좌측
행렬의
열과
우측
행렬의
행이
같아야만
합니다.예를
들면(3, 2)
행렬과(2, 3)
행렬은
좌측
행렬의
열이
2, 우측
행렬의
행이
2
로
같으므로
행렬곱이
가능하고
결과로(3, 3)
행렬이
출력됩니다.또한(2, 3)
행렬과(3, 2)
행렬은
행렬곱을
하면(2, 2)
행렬이
출력됩니다.좌측
행렬의
열과
우측
행렬의
행이
같지
않다면.dot()
메소드를
사용했을
때
shape
가
같지
않다는
에러가
출력됩니다.
    이
규칙은
차원이
다른
배열간의
내적을
구할
때도
적용됩니다.
2
차원
배열인(3, 2)
행렬과
1
차원
배열인(2)
는
내적이
가능하며
결과로
1
차원
배열이
출력됩니다.
1
차원
배열은
위
파이썬
코드에서도
확인
했듯이(2, )
형태이기
때문입니다.이를
파이썬
코드로
확인해보면
아래와
같습니다.
import numpy as npA = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([7, 8])
A.shape  # (3, 2)B.shape# (2, )# 행렬내적을 해주는 numpy 메소드.dot()np.dot(A,B)# array([23, 53, 83])np.dot(A,B).shape# (3, )

2
차원
배열인
행렬과
1
차원
배열인
벡터간
행렬곱을
확인
할
수
있습니다.
    이러한
행렬곱이
중요한
이유는
신경망에서
입력값과
가중치의
곱의
합을
통한
활성함수
계산이
용이하기
때문입니다.이를
간단한
수식으로
확인해보면
다음과
같습니다.

[그림23.행렬곱과
신경망의
계산]

[그림23]
에서
확인할
수
있듯이
입력층과
출력층
그리고
가중치는
위의
행렬곱으로
표현할
수
있다.여기서
주의해야될
점은
각
곱의
차원의
원소수가
같아야
한다는
점입니다.이를
파이썬으로
계산해보면
아래와
같습니다.
import numpy as npX = np.array(
    [1, 2])  # 입력층= np.array([[1, 3, 5], [2, 4, 6]])    # 가중치= np.dot(X, W)    # 출력층int(Y)# [ 5 11 17]

이처럼
다차원
배열을
이용하면
아무리
신경망
구조가
복잡하더라도
행렬곱을
통해
해결할
수
있습니다.이
행렬곱
연산의
속도가
여러
딥러닝
프레임워크의
연산속도의
기반이
됩니다.텐서플로는
보통
정도의
속도를
보여주고
C + + 기반의
딥러닝
프레임워크인
시애노(Theano)
는
텐서플로의
5
배, 구글
딥마인드에서
사용하는
딥러닝
프레임워크인
토치(Torch)
는
텐서플로의
9
배
연산속도를
가진다고
알려져
있습니다.

4) 신경망
구현하기

신경망은
입력층에서
출력층으로
기본적으로
한방향으로
흐릅니다.한
싸이클이
끝나면
역전파
알고리즘을
통해
계속
학습을
진행하지만
역전파
알고리즘과
같은
고급
알고리즘은
4
장, 5
장에서
배우게
됩니다.지금
배우는
것처럼
한방향으로만
정보가
전방으로
전달되는
신경망을
피드포워드
신경망(FNN, Feed
forward
Neural
Network) 이라고
합니다.기본적으로
신경망은
입력층에서
데이터
입력을
받은
뒤
은닉층에서
데이터를
학습하고
출력층으로
결과를
내보냅니다.이를
그림으로
간단하게
확인하면
다음과
같습니다.

[그림24.신경망의
정보
전달
싸이클]

입력층의
역할은
입력
데이터를
받아들이는
것이고
이를
위해서
입력층의
노드(뉴런)
개수는
입력데이터의
특성
개수와
일치해야
합니다.은닉층은
학습을
진행하는
층으로
은닉층의
노드
수와
은닉층
층의
개수는
설계자가
경험으로
얻어낼
수
밖에
없습니다.뉴런의
수가
너무
많으면
오버피팅이
발생하고
너무
적으면
언더피팅이
발생하여
학습이
되지
않습니다.또한
은닉층의
개수가
지나치게
많은
경우
비효율적입니다.단순히
은닉층의
개수를
2
배
늘리면
연산에
걸리는
시간은
400 % 증가하지만
학습효율은
10 % 만
증가하기도
합니다.출력층은
은닉층을
거쳐서
얻어낸
결과를
해결하고자
하는
문제에
맞게
만들어
줍니다.필기체
숫자
0
부터
9
까지를
인식하는
신경망이면
출력층이
10
개가
될
것이고
개와
고양이를
분류하는
신경망이라면
3
개의
출력층이
될
것입니다.

    이제
위에서
배운
다차원
배열을
이용하여
층이
3
개인
다층
신경망을
간단하게
구현해보도록
하겠습니다.행렬곱과
각
행렬의
원소의
위치를
잘
확인하면
그리
어렵지
않습니다.구현해야할
3
층
신경망의
구조는
아래와
같습니다.

[그림25.신경망
예제]

[그림25]
를
확인해보면
3
층
신경망이
어떻게
구성되어있는지
확인할
수
있습니다.입력층(녹색
원)은
2
개이며
각
층마다
편향(파란
원)이
존재합니다.은닉층(빨간
원)은
2
개
층으로
구성되어
있고
출력층(노란
원)의
출력값은
2
개입니다.

[그림26.신경망
표기
예]
위
그림을
확인해보면, 와
같은
형식으로
표기되어있는
것을
확인
할
수
있습니다.우측
상단의(1)
은
1
층의
가중치를
의미한다.우측
하단의
1
2
에서
1
은
다음층의
뉴런
번호, 2
는
앞
층의
뉴런
번호를
의미한다.따라서
는
앞의
1
번
뉴런에서
뒤의
2
번
뉴런으로
이동하는
신경망
1
층의
가중치를
의미합니다.

[그림27.신경망
1
층의
정보
전달
과정]
예제
3
층
신경망의
구조를
찬찬히
뜯어보면
우선
입력층은
2
개로
구성되어
있고
1
층에서
편향이[그림27]
처럼
존재합니다.여기서
가중치,, , , , 에
의해
입력값은,, 에
입력됩니다.이
입력값을
수식으로
나타내면
으로
표현할
수
있습니다.이를
행렬
내적으로
표현하면
1
층의
노드를, 1
층의
가중치를, 1
층의
편향을, 입력값을
이라고
할
때
위
식은
으로
표현할
수
있습니다.

[그림28.신경망
1
층의
출력값을
2
층의
입력값으로
변환]

이를
이용해서
Numpy의
다차원
배열을
이용하면
신경망
1
층을
파이썬
코드로
짤
수
있는
것입니다.마찬가지로
1
층의
출력값을
다시
2
층의
입력값으로
넣고
똑같은
방식으로
입력노드
행렬(1
층의
출력노드
행렬), 가중치
행렬, 편향
행렬의
행렬
연산을
통해
2
층의
출력노드
행렬을
구할
수
있게
됩니다.

[그림29.신경망
2
층의
정보
전달
과정]

마찬가지로
신경망
1
층에서
행렬
연산식을
통해
출력값을
구했던
것처럼
1
층의
출력값을
2
층의
입력값으로
연결해주고
2
층의
가중치와
2
층의
편향을
더해주면
2
층의
출력값이
완성됩니다.이를
식으로
표현하면
이
되고
이에
맞는
행렬
연산을
파이썬에서
numpy를
이용해서
구현하면
됩니다.

[그림30.신경망
출력층의
정보
전달
과정]
마지막으로[그림30]
처럼
2
층의
출력값을
동일한
방법으로
출력층의
입력값으로
넣고
출력층사이의
가중치와
편향을
더해준
동일한
방법으로
식을
계산하면
최종적인
출력값이
뽑히게
됩니다.한가지
위
과정과
다른
점이
있다면
출력층의
활성함수는
풀고자
하는
문제의
성질에
맞게
정합니다.회귀가
목적인
신경망은
출력층에
항등함수를
사용하고
이중클래스
분류(0
아니면
1)에는
시그모이드
함수를, 다중클래스
분류(분류해야되는
값이
3
이상)에는
소프트맥스
함수를
일반적으로
사용합니다.그럼
출력층에
사용하는
활성
함수를
알아보도록
하겠습니다.앞서
얘기했던
것처럼
일반적으로
회귀에는
항등함수를, 분류에는
소프트맥스
함수를
보통
사용합니다.회귀는
입력데이터의
연속적인
수치를
예측
하는
것을
의미하고
분류는
각
데이터가
어떤
범주(


class )에 속하는지 나누는 것을 의미합니다.항등함수는 입력값이 그대로 출력되는 함수로 흔히 알고 있는 를 의미합니다.이를 파이썬 코드로 표현하면 아래와 같습니다.


def identity_function(x):    return x


소프트맥스
함수는
자연상수를
밑수로
하는
지수함수로
이루어진
하나의
함수입니다.우선
함수식은
아래와
같습니다.

[수식12.Softmax 함수]
여기서
n은
출력층의
뉴런의
개수, k는
그중
k번째, 는
따라서
k번째의
출력값을
의미합니다.정리하면
k번째의
출력값은
해당
k번째의
입력값의
지수함수를
전체
입력값의
지수함수로
나눈
것을
의미합니다.이러한
소프트맥스
함수가
가지는
의미는
바로
시그모이드
함수를
일반화
한
것입니다.이를
통해
각
클래스에
대한
확률을
계산
할
수도
있게
됩니다.시그모이드
함수를
일반화
해서
각
클래스에
대한
확률을
계산
할
수
있다는
것은
모든
소프트맥스
함수의
출력값을
더하면
1
이
나오게
됩니다.소프트맥스
함수의
출력값은
0
과
1
사이의
값이고
각각의
출력값은
개별
출력값에
대한
확률값이기
때문에
전체
소프트맥스
함수의
합은
항상
1
이
되는
특별한
성질을
가집니다.때문에
소프트맥스
함수를
출력층의
활성함수로
사용하면
출력결과를
확률적으로
결론낼
수
있습니다.예를
들어
y[0] = 0.018, y[1] = 0.245, y[2] = 0.737
로
결과가
출력되었다면
1.8 % 의
확률로
0
번
클래스, 24.5 % 의
확률로
1
번
클래스, 73.7 % 의
확률로
2
번
클래스일
것이므로 ‘2
번
클래스일
확률이
가장
높고
따라서
답은
2
번
클래스다’라는
결과를
도출
할
수
있는
것입니다.소프트맥스
함수를
이용해서
통계적(확률적)
으로
문제를
대응할
수
있게
되는
것입니다.소프트맥스
함수는
단조
증가
함수인
지수함수
exp()
를
기반으로
하므로
소프트맥스
함수의
출력값의
대소관계가
그대로
입력된
원소의
대소관계를
이어받습니다.따라서
역으로
소프트맥스
함수를
통해
나온
출력값의
대소관계를
입력값의
대소관계로
판단해도
됩니다.그래서
신경망
학습과정에선
출력층의
활성함수로
소프트맥스
함수를
사용하고
학습된
모델을
이용해서
추론(분류
및
회귀)하는
과정에선
소프트맥스
함수를
활성함수에서
생략해도
됩니다.이러한
소프트맥스
함수의
구현엔
주의사항이
한가지
있습니다.지수함수는
입력값이
커지면
급격하게
무한히
증가하는
경향을
가집니다.이를
오버플로(Overflow)
라고
합니다.입력값이
100
인
exp(100)
은
10
의
40
승이
넘는
수입니다.오버플로를
해결하기
위해선
해당
값을
전체
데이터셋에서의
최대값으로
뺀
값으로
치환하는
방법을
사용합니다.위
과정을
수식으로
나타내면
아래와
같습니다.

[수식13.Softmax 함수의 변형]

소프트맥스
함수의
분모
분자에
C라는
상수를
곱해줍니다.같은
상수값을
곱해주었으므로
전체
값엔
변화가
없습니다.그리고
여기에
지수함수와
로그함수의
성질
중
하나인
를
이용하여
상수
C를
exp()
함수
안으로
넣습니다.그럼
상수
C는
exp()
함수
내에서
로
변화되고
를
상수
으로
받게
되면
아래의
수식으로
변형됩니다.

[수식14.Softmax 함수의 변형 결과]

왜
이러한
변화를
거쳐야
하는지
아래의
파이썬
코드를
확인해보면
알
수
있습니다.

import numpy as npa = np.array([1010, 1000,
                                990])  # softmax 함수식에 입력np.exp(a) / np.sum(np.exp(a))   # overflow가 발생함# RuntimeWarning: overflow encountered in exp#   np.exp(a) / np.sum(np.exp(a))# RuntimeWarning: invalid value encountered in true_divide#   np.exp(a) / np.sum(np.exp(a))# 변경된 softmax 함수식에 입력c = np.max(a)print(np.exp(a-c) / np.sum(np.exp(a-c)))    # 정상적으로 계산됨# [  9.99954600e-01   4.53978686e-05   2.06106005e-09]

이처럼
같은
스케일의
변화는
아무런
결과값에
아무런
영향을
주지
않는
점을
이용해서
소프트맥스
함수의
오버플로
현상을
해결할
수
있었습니다.이를
이용하여
소프트맥스
함수를
파이썬으로
구현하면
아래와
같습니다.


def softmax(a):    c = np.max(a)


exp_a = np.exp(a - c)  # 오버플로방지    sum_exp_a = np.sum(exp_a)    y= exp_a / sum_exp_a    return y

마지막으로
출력층의
노드
개수를
정하는
방법은
간단합니다.입력한
데이터의
클래스
개수만큼
출력층의
노드
개수를
정해주면
됩니다.예를
들어
0
부터
9
까지
10
개의
숫자를
분류하고
싶으면
0
부터
9
가
적힌
데이터를
입력값으로
입력하게
되고
그렇다면
10
개의
숫자를
분류해야
하므로
10
개의
출력노드를
만들면
됩니다.다른
예로
개와
고양이를
분류하고
싶다면
개, 고양이
총
2
개의
출력노드를
만들면
됩니다.

이제
여태까지
배운
이론을
총
이용해서
은닉층이
2
개인
다층
신경망(보통
입력층을
제외한
층수로
신경망을
부릅니다.따라서
이
경우는
3
층
신경망이라고
합니다.)을
간단하게
파이썬으로
코딩해보도록
하겠습닏.이
신경망
모델은
출력층의
활성
함수로
항등함수로
정의합니다.결과적으로
위
과정을
모두
합한
전체적인
은닉층이
2
층인
다층
신경망의
파이썬
구현
코드는
아래와
같습니다.

# 관련 라이브러리 불러오기import numpy as np# 시그모이드 함수def sigmoid(x):    return 1 / (1 + np.exp(-x))# identity function. 항등함수를 사용def identity_function(x):    return x# 신경망 초기화. 여기서 가중치와 편향의 다차원배열을 선언해준다.def init_network():    network = {}    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])    network['b1'] = np.array([0.1, 0.2, 0.3])    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])    network['b2'] = np.array([0.1, 0.2])    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])    network['b3'] = np.array([0.1, 0.2])    return network
# 순전파 신경망 함수. 가중치와 편향을 입력받아 입력층과 은닉층의 활성함수는 시그모이드 함수를,# 출력층의 활성함수는 항등함수를 사용하는 3층 신경망을 함수로 구현def forward(network, x):    w1, w2, w3 = network['w1'], network['w2'], network['w3']    b1, b2, b3 = network['b1'], network['b2'], network['b3']    a1 = np.dot(x, w1) + b1    z1 = sigmoid(a1)    a2 = np.dot(z1, w2) + b2    z2 = sigmoid(a2)    a3 = np.dot(z2, w3) + b3    y = identity_function(a3)    return ynetwork = init_network()    # 신경망의 가중치와 편향값 인스턴스화x = np.array([1.0, 0.5])    # 입력값 배열 선언y = forward(network, x)     # 순전파 신경망 함수에 가중치와 편향값, 입력값을 입력print(y)# [ 0.31682708  0.69627909]

단순한
신경망을
설계하는
것은
어렵지
않습니다.다차원
배열을
잘
사용해서
가중치와
입력값과
편향을
잘
버무려
주고
활성함수를
무엇을
사용할지
정해서
구현한
다음
구현한
활성함수에
값을
잘
넣어준
다음
이전
층의
출력값을
다음
층의
출력값으로
잘
연결해서
원하는
층만큼
이어주면
됩니다.

5) 예제
데이터셋
사용해보기

신경망을
가장
손쉽게
접해보고
사용할
수
있는
예제중
하나가
바로
MNIST
데이터입니다.MNIST는
손글씨
이미지
데이터로
0
부터
9
까지의
각기
다른
손글씨
그림입니다.훈련용
이미지가
6
만장, 테스트용
이미지가
1
만장
준비되어
있습니다.일반적으로
훈련용
이미지로
모델을
학습시키고
테스트용
이미지로
성능을
평가합니다.성능
평가의
기준은
분류
정확도입니다.지금과
같은
단순한
순전파
신경망에서부터
복잡한
CNN까지
MNIST
데이터는
다양하게
사용되고
있습니다.기본적으로
MNIST는
이미지
데이터이므로
28
x
28
픽셀
크기의
그레이채널
이미지파일이며
각
픽셀은
0
부터
255
까지의
RGB
값을
가집니다.또한
각
이미지엔
그
이미지가
의미하는
실제
숫자가
라벨로
붙어있습니다.
    현재
작업공간의
상위폴더에
mnist.py를
두고
작업을
시작합니다.파이썬에는
피클(pickle)
이란
기능이
있는데
프로그램
실행
중에
특정
객체를
파일로
저장하는
기능으로
저장해둔
피클을
로드하면
실행
당시의
객체를
복원할
수
있습니다.MNIST
데이터셋은
이미지
파일로
비교적
큰
파일이기에
첫
실행에서
다운로드를
받고
그
후
실행에선
저장해둔
피클
파일을
로드해서
사용합니다.따라서
데이터를
불러들이는
시간을
줄일
수
있습니다.MNIST
데이터셋을
불러오는
파이썬
코드는
다음과
같습니다.

import sys, ossys.path.append

(os.pardir)  # 부모디렉터리의 파일을 가져올 수 있도록 설정import numpy as npfrom mnist import load_mnist    
# 작업하고 있는 파이썬파일 상위폴더에 mnist.py 파일을 놓고 import 한다(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

MNIST
데이터를
load
하는
메소드가
mnist
파일의
load_mnist
메소드입니다.불러온
데이터는(이미지, 해당이미지의
라벨)로
구성됩니다.load_mnist
메소드는
3
가지
매개변수를
받습니다.flatten, normalize, one_hot_label입니다.우선
flatten
매개변수는
입력이미지를
1
차원
배열로
만들지
정하는
옵션입니다.False로
설정하면
입력
이미지를
1
x28x28의
3
차원
배열로, True로
설정하면
784
개의
원소로
이루어진
1
차원
배열로
저장합니다.
1
차원
배열로
저장한
데이터는.reshape(픽셀값)
으로
원래
이미지로
볼
수
있습니다.예를
들어
img.reshape(28, 28)
이라고
하면
원래
이미지인
28
x
28
픽셀의
이미지를
볼
수
있습니다.normalize는
이미지의
픽셀값을
0
부터
1
사이의
값으로
정규화
할지
정하는
옵션입니다.마지막으로
one_hot
encoding이란
정답을
뜻하는
원소만
1
이고
나머진
0
으로
두는
인코딩
방법으로
이
인코딩
방법을
사용할
것인지
정하는
부분입니다.이제
불러온
데이터로
신경망을
만들어
보겠습니다.

    코드의
흐름은
다음과
같습니다.

1.
mnist
데이터를
다운로드
받거나
불러옵니다.
2.
위의
데이터로
신경망에
입력할
입력데이터를
초기화
해줍니다.
3.
예제로
작성되어
있는
피클
파일로
가중치를
불러옵니다.
4.
위에서
준비한
데이터들로
예측
신경망을
제작합니다.
    은닉층의
활성함수는
시그모이드
함수를
사용합니다.
    출력층의
활성함수는
소프트맥스
함수를
사용합니다.
5.
위
신경망을
실행해보고
소프트맥스
결과값인
확률이
가장
높은
원소의
인덱스를
얻어냅니다.이것이
결과입니다.

6.
위
단계로
예제
코드를
실행하면
Accuracy:0.9352
로
출력되는데
이는
올바르게
분류한
비율이
93.52 % 라는
의미입니다.이는
각
라벨(숫자
종류)의
확률을
predict()
함수가
차원
배열로
반환하는데
이를
np.argmax()
메소드로
해당
배열에서
값이
가장
큰
원소의
인덱스를
얻어
냅니다.이를
정답
라벨과
비교해서
정답인
개수를
count하여
count를
전체
이미지
개수로
나누어
정확도를
구하는
과정을
거칩니다.

    우선
3
단계에서
피클데이터로
저장된
가중치를
불러오는
코드는
다음과
같습니다.

import picklewith

open("sample_weight.pkl", 'rb') as f:network = pickle.load(f)
print(network)

피클
파일을
작업환경
루트에
넣고
이
코드를
파이썬에서
실행해보면
가중치를
확인할
수
있습니다.이미지는
28 * 28
픽셀, 총
784
개의
입력값을
가지므로
가중치도
총
784
개가
존재합니다.

    위
6
단계
과정을
전부
거치는
mnist
데이터
분류
신경망
모델의
파이썬
코드는
아래와
같습니다.

import sysimport

pickleimport
osimport
numpy as npsys.path.append(
    os.pardir)  # 부모디렉터리의 파일을 가져올 수 있도록 설정from mnist import load_mnist    # 파이썬 파일 상위폴더에 mnist.py 파일을 놓고 import 한다(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)# 소프트맥스함수def softmax(a):    c = np.max(a)    exp_a = np.exp(a-c) # 오버플로 방지    sum_exp_a = np.sum(exp_a)    y = exp_a / sum_exp_a    return y# 시그모이드함수    def sigmoid(x):    return 1 / (1 + np.exp(-x))def get_data():    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, 
one_hot_label = False)    return x_test, t_test  # 가중치와 편향을 초기화, 인스턴스화def init_network():    with open("sample_weight.pkl", 'rb') as f:    network = pickle.load(f)    return network# 은닉층 활성함수로 시그모이드함수, 출력층 활성함수로 소프트맥스함수를 쓴 순전파 신경망def predict(network, x):    W1, W2, W3 = network['W1'], network['W2'], network['W3']    b1, b2, b3 = network['b1'], network['b2'], network['b3']    a1 = np.dot(x, W1) + b1    z1 = sigmoid(a1)    a2 = np.dot(z1, W2) + b2    z2 = sigmoid(a2)    a3 = np.dot(z2, W3) + b3    y = softmax(a3)    return yx, t = get_data()network = init_network()accuracy_cnt = 0for i in range(len(x)):    y = predict(network, x[i])    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.    if p == t[i]:        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

위
코드를
잠시
살펴보면
load_mnist
메소드의
매개변수
옵션인
normalize
값을
True로
설정해서
정규화를
하도록
했습니다.스케일링
범위를
줄여줌으로써
계산을
더
용이하게
하는데
이러한
과정을 ‘정규화’라고
하고
정규화처럼
데이터를
손질하는
행위를 ‘전처리’라고
합니다.단순한
순전파
신경망으론
정분류율이
93 % 밖에
나오지
않지만
차후에
CNN과
같은
고급
기법들을
사용하면
99 % 까지
분류율을
향상시킬
수
있습니다.
마지막으로
이번
단원에서
살펴볼
한가지가
남아있습니다.데이터를
입력하고
처리하는
과정에
약간의
기법을
추가해서
연산
속도와
효율을
향상시키는
방법이
있습니다.이를
배치
처리라고
합니다.배치(batch)
란
여러개의
데이터를
하나로
묶은
묶음
상태의
데이터를
배치라고
합니다.mnist
데이터의
경우
이미지가
한
장
한
장
지폐
다발로
묶여있는
형태라고
생각하시면
됩니다.배치
단위를
100
으로
할
경우
이미지를
100
장씩
묶어서
학습하는
방식이라고
생각하시면
됩니다.기본적으로
이미지가
한
장이
들어갈
때의
신경망
각
층의
배열
형상의
추이는
아래와
같습니다.

[그림31.신경망 각 층의 배열 형상 추이]

784
에서
시작하여
각
은닉층을
거친
다음
0
부터
9
까지의
클래스
10
종류인
10
개로
배열이
변화하는
것을
볼
수
있습니다.이를
배치처리로
바꾸면
배열
형상
추이는
아래와
같이
변하게
됩니다.

[그림32.신경망 각 층의 배치 배열 형상 추이]
[그림32]
에서처럼
배치를
100
개
단위로
묶으면
첫
입력값은
100 * 784
가
되지만
은닉
층에서
처리되는
과정은
변함이
없고
출력
층에서만
각각의
100
개의
개별
이미지에
대한
클래스로
출력되는
것을
확인
할
수
있습니다.컴퓨터는
큰
배열을
한꺼번에
계산하는
것이
분할된
작은
배열을
여러번
계산하는
것
보다
빠릅니다.따라서
배치처리를
하게
되면
한번
실행에
100
개의
이미지를
처리하므로
작업효율이
훨씬
좋아지게
됩니다.다만
입력과
출력시
배치단위
그대로
출력되므로
지나치게
배치를
크게
하면
역으로
작업성능이
떨어지게
되는
점을
조심해야
합니다.
배치처리를
위
mnist
신경망
코드에
구현하면
다음과
같습니다.
# 소프트맥스함수def softmax(a):    c = np.max(a)    exp_a = np.exp(a-c) # 오버플로방지    sum_exp_a = np.sum(exp_a)    y = exp_a / sum_exp_a    return y# 시그모이드함수def sigmoid(x):    return 1 / (1 + np.exp(-x))import sys, ossys.path.append(os.pardir)  # 부모디렉터리의파일을가져올수있도록import numpy as npimport picklefrom mnist import load_mnistdef get_data():    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True,     one_hot_label=False)    return x_test, t_testdef init_network():    with open("sample_weight.pkl", 'rb') as f:    network = pickle.load(f)    return networkdef predict(network, x):    w1, w2, w3 = network['W1'], network['W2'], network['W3']    b1, b2, b3 = network['b1'], network['b2'], network['b3']    a1 = np.dot(x, w1) + b1    z1 = sigmoid(a1)    a2 = np.dot(z1, w2) + b2    z2 = sigmoid(a2)    a3 = np.dot(z2, w3) + b3    y = softmax(a3)    return yx, t = get_data()network = init_network()batch_size = 100 # 배치크기accuracy_cnt = 0for i in range(0, len(x), batch_size):    x_batch = x[i:i+batch_size]    y_batch = predict(network, x_batch)    p = np.argmax(y_batch, axis=1)    accuracy_cnt += np.sum(p == t[i:i+batch_size])print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

간단하게
설명을
곁들이면
아래
for I range 부분이 배치처리와 관련된 부분입니다.전체 x의 길이에 대해 배치 크기(batch_size)만큼 묶어서 묶음처리를 하게 for 루프문을 작성했습니다.입력값의 배치인 x_batch 는 I부터 I+batch_size 만큼의 x값을 묶음처리 한 것이고 출력값의 배치인 y_batch는 입력값으로 x_batch를 입력한 그 결과입니다.마찬가지로 역시 count 부분도 배치를 씌워줘야 합니다.나머지 부분은 완전 동일하다고 할 수 있습니다.

이번장에서
배운
여러
활성함수들과
다층
신경망
구조, 배치처리와
같은
부분들은
CNN과
RNN같은
고급
신경망
기법들에서도
여전히
사용합니다.따라서
지금
기초를
탄탄하게
다져놓는
것이
중요하다고
할
수
있습니다.