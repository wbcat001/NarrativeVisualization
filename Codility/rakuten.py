

def check(a1, a2, b1, b2):
    if a2[0] < b1[0] and a1[0] > b2[0]:
        return False
    # rect1 が rect2 の上または下に完全にある場合
    if a2[1] < b1[1] or a1[1] > b2[1]:
        return False
    # 衝突している
    return True

def is_overlapping( t1, b1, t2, b2):
       
        if t1[0] > b2[0] or t2[0] > b1[0] or t1[1] > b2[1] or t2[1] > b1[1]:
            return False
        return True
def solution(A):

    if not A or not A[0]:
        return 0
    
    n, m = len(A), len(A[0])

    dp = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if A[i][j]:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    rect_dict = {}

    for i in range(n):
        for j in range(m):
            size = dp[i][j]
            if size > 0:
                if size not in rect_dict:
                    rect_dict[size] = []
                
                rect_dict[size].append(((i - size + 1, j - size + 1), (i, j)))
    
    for size in sorted(rect_dict.keys(), reverse=True):
        
        rect = rect_dict[size]

        for i in range(len(rect)):
            for j in range( i + 1, len(rect)):

                if not is_overlapping(rect[i][0], rect[i][1], rect[j][0], rect[j][1]):
                    return size * size

    return 0


A = [[True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True]]
print(solution(A))