class Solution:
    def solution(self, A):
        if not A or not A[0]:
            return 0

        n, m = len(A), len(A[0])
        dp = [[0] * m for _ in range(n)]

        # Step 1: Calculate the largest square size at each cell using DP
        for i in range(n):
            for j in range(m):
                if A[i][j]:
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

        # Step 2: Collect all squares grouped by size
        squares_by_size = {}
        for i in range(n):
            for j in range(m):
                size = dp[i][j]
                if size > 0:
                    if size not in squares_by_size:
                        squares_by_size[size] = []
                    # Store top-left and bottom-right corners of the square
                    squares_by_size[size].append(((i - size + 1, j - size + 1), (i, j)))

        # Step 3: Check from largest to smallest square sizes for two disjoint squares
        for size in sorted(squares_by_size.keys(), reverse=True):
            squares = squares_by_size[size]
            for i in range(len(squares)):
                for j in range(i + 1, len(squares)):
                    if not self.is_overlapping(squares[i][0], squares[i][1], squares[j][0], squares[j][1]):
                        return size * size

        # Step 4: If no valid pair is found, return 0
        return 0

    def is_overlapping(self, t1, b1, t2, b2):
        """
        Check if two squares overlap based on their top-left (t1, t2) and bottom-right (b1, b2) corners.
        """
        # No overlap if one is completely above, below, left, or right of the other
        if t1[0] > b2[0] or t2[0] > b1[0] or t1[1] > b2[1] or t2[1] > b1[1]:
            return False
        return True


# Example Usage:
A = [[True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True]]
solution = Solution()
print(solution.solution(A))  # Expected Output: 4
