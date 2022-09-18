## loss설정

- 출력층 activation이 sigmoid 인 경우
  - binary_crossentropy


- 출력층 activation이 softmax 인 경우 
  - 원핫인코딩(O): categorical_crossentropy
  - 원핫인코딩(X): sparse_categorical_crossentropy)


- 출력층 activation이 sigmoid 인 경우
  - binary_crossentropy


- flow_from_directory
  - class_mode는 3개 이상의 클래스인 경우 'categorical', 이진 분류의 경우 binary를 지정합니다.
