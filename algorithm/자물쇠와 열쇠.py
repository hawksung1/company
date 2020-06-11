import numpy as np

def solution(key, lock):
    key = np.array(key)
    lock = np.array(lock)
    answer = True
    #  TODO: lock 이 전부가 1이 되도록 key가 있어야 함
    #       key는 회전 이동이 가능

    #  or 사용
    #  1. 4가지 key를 만든다.
    key_list = [key, np.rot90(key), np.rot90(key, 2), np.rot90(key, 3)]
    #  2. lock에 key len - 1 만큼 padding을 만든다.
    padding_lock = make_padding(lock, len(key)-1)
    #  3. kwy 별로 lock의 모든 좌표에 or 시전하여 모두 1인 lock이 나오면 true else false
    for i in key_list:
        compare(i, padding_lock)
    return answer

def make_padding(lock, i):
    result = np.ones(2 * i + len(lock))
    
    result[i:lock.shape[0]+i, i:lock.shape[1]+i] = lock
    return result

def compare(key, lock):
    pass

if __name__ == "__main__":
    solution([[0, 0, 0], [1, 0, 0], [0, 1, 1]], [[1, 1, 1], [1, 1, 0], [1, 0, 1]])