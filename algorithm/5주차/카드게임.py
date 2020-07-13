"""
    20200712
    https://programmers.co.kr/learn/courses/30/lessons/42896
"""
import copy


def solution(left, right):
    answer = 0
    game_list = []
    game_list.append(Game(left, right))
    # 1. while 카드더미 not empty
    # 2. compare left right
    # 3. if right small, pop right, add to score
    #       else pop left or pop left and right(duplicate class)
    # 4. if is_empty, append to answer
    while len(game_list) != 0:
        game = game_list.pop(0)

        if game.is_empty():
            answer += game.get_score()

        if game.is_right_small():
            game.add_right_to_score()
        else:
            new_game = copy.deepcopy(game)
            new_game.pop_both()
            game.pop_left()
            game_list.append(new_game)
            game_list.append(game)

    return answer

class Game:
    def __init__(self, left, right, score=0):
        self.score = score
        self.left = left
        self.right = right

    def pop_left(self):
        self.left.pop()
        
    def pop_right(self):
        self.score += self.right.pop(0)
        
    def pop_both(self):
        self.left.pop()
        self.right.pop()
    
    def is_right_small(self):
        if self.right[0] < self.left[0]:
            return True
        return False

    def get_left(self):
        return self.left[0]

    def get_right(self):
        return self.right[0]
    
    def get_score(self):
        return self.score

    def is_empty(self):
        if len(self.left) == 0 or len(self.right) == 0:
            return True
        return False
    
    def add_right_to_score(self):
        self.score += self.right.pop[0]

if __name__ == "__main__":
    print(solution([3, 2, 5], [2, 4, 1]))
