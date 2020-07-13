def solution(s):
    tmp_ans = []
    for i in range(1, len(s)/2+1):
        tmp_ans.append(sub_solution(s, i))
    answer = min(tmp_ans)
    return answer

def sub_solution(s, n):
    result = ""
    count=0
    for i in range(0, len(s)-n, n):
        sub_str = s[i:i+n]
        if sub_str == s[i+n, i+2*n]:
            count += 1
        elif count != 0:
            result.append(str(count)+sub_str)
        else:
            result.append(sub_str)
    return len(result)





if __name__ == "__main__":
    solution("aabbaccc")