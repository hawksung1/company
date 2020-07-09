"""
    dynamic programming
"""
def solution(m, n, puddles):
    board = [[0]*(m+1) for i in range(n+1)]
    board[1][1] = 1
    for i in range(1, n+1):
        for j in range(1, m+1):
            if i == 1 and j == 1:
                continue
            if [j, i] in puddles:
                board[i][j] = 0
            else:
                board[i][j] = board[i-1][j] + board[i][j-1]
    answer = board[n][m]
    return answer % 1000000007

if __name__ == "__main__":
    tmp = solution(4,	3	,[[2, 2]]) # 4
    print(tmp)