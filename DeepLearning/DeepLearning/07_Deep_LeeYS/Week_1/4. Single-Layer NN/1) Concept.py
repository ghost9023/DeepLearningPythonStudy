#1) 신경망의 개념


##########KEYWORD###############





################################


#사람의 뇌는 biological한 뉴런 네트워크로 이루어져있음.
#뉴런들은 여러 전기신호를 수상돌기에서 받는다. 신호들을 소마에서 처리한 후 축색 돌기를 통해 output을 다른 뉴런으로 신호를
#전달하는 원리가 인간의 신경계라고 할 수 있다.
#이 신경계의 뉴런 구조를 수학적으로 구현한 것이 인공 신경망.

#뉴런의 입력은 다수이고 출력은 하나이며 여러 신경세포로부터 전달되어 온 신호들은
#합산되어 출력된다. 합산된 값이 임계값 이상이면 출력 신호가 생기고 이하이면 출력 신호가 없는 구조.
#다수의 뉴런이 연결되어 의미 있는 작업을 하듯이 인공신경망도 노드들을 연결시켜 Layer를 만들고 연결 강도는 가중치로 처리한다.
#이처럼 사람의 인지 및 사고 구조를 수학적으로 해석해서 구현한 것이 기본적인 신경망에 대한 개념이라고 할 수 있다.

#P19 그림15

#위 그림에서 볼 수 있듯이 단층 신경망, 다층 신경망, 심층 신경망 모두 기본적으로 이러한 구조를 가진다.
#입력을 받는 입력층, 학습이나 분류가 진행되는 은닉층, 은닉층의 결과를 받아서 출력을 해주는 출력층으로 나뉜다.
#이중 은닉층은 사람의 눈으로 볼 수 없기 때문에 은닉층이라고 부른다.
#이 은닉층이 0개인 신경망 구조를 단층 신경망, 1개인 신경망을 다층 신경망, 2개 이상이면 심층 신경망이라 부름.
#구조적으로 이전에 배운 퍼셉트론과 큰 차이는 없다.

#신경망이 퍼셉트론과 다른 점은 편향과 활성함수의 구조가 다른점.