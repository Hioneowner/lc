'''
27. 移除元素
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

示例 1: 给定 nums = [3,2,2,3], val = 3, 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。 你不需要考虑数组中超出新长度后面的元素。

示例 2: 给定 nums = [0,1,2,2,3,0,4,2], val = 2, 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。

双指针法（快慢指针法）： 通过一个快指针和慢指针在一个for循环下完成两个for循环的工作。
定义快慢指针
快指针：寻找新数组的元素 ，新数组就是不含有目标元素的数组
慢指针：指向更新 新数组下标的位置
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # 快慢指针
        fast = 0  # 快指针
        slow = 0  # 慢指针
        size = len(nums)
        while fast < size:  # 不加等于是因为，a = size时，nums[a]会越界
            # slow 用来收集不等于 val 的值，如果 fast 对应值不等于 val，则把它与 slow 替换
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow




26.删除排序数组中的重复项(opens new window)
给你一个有序数组 nums ，请你原地删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。

示例 1:

给定数组 nums = [1,1,2],

函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。

你不需要考虑数组中超出新长度后面的元素。 示例 2:

给定 nums = [0,0,1,1,1,2,2,3,3,4],

函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        #第一个指针用于更改值
        first = 0
        #第二个指针用于遍历
        for second in range(len(nums)):
            #如果和之前记录的值不同
            if nums[first] != nums[second]:
                #第一个指针先加1
                first += 1
                #然后赋值
                nums[first] = nums[second]
        return first+1


283.移动零(opens new window)
概述：给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。请注意 ，必须在不复制数组的情况下原地对数组进行操作。
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]

输入: nums = [0]
输出: [0]

方法一：快慢指针
思路：核心思路就是遇到 0 时，两两交换，区间上限快指针。需要注意原数组已经是有序的。
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        n = len(nums)
        left, right = 0, 0
        while right < n:
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1



844.比较含退格的字符串
给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。

**注意：**如果对空文本输入退格字符，文本继续为空。
class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        def text_input(text):
            # 栈
            stack = []
            # 遍历字符串
            for ch in text:
                # 普通字符入栈
                if ch != "#":
                    stack.append(ch)
                # "#" 字符且栈非空时，弹出
                elif stack:
                    stack.pop()

            return ''.join(stack)
        # 判断两者是否相同
        return text_input(S) == text_input(T)



977.有序数组的平方
给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。

示例 1：
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]，排序后，数组变为 [0,1,9,16,100]

示例 2：
输入：nums = [-7,-3,2,3,11]
输出：[4,9,9,49,121]

（版本一）双指针法
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        l, r, i = 0, len(nums)-1, len(nums)-1
        res = [float('inf')] * len(nums) # 需要提前定义列表，存放结果
        while l <= r:
            if nums[l] ** 2 < nums[r] ** 2: # 左右边界进行对比，找出最大值
                res[i] = nums[r] ** 2
                r -= 1 # 右指针往左移动
            else:
                res[i] = nums[l] ** 2
                l += 1 # 左指针往右移动
            i -= 1 # 存放结果的指针需要往前平移一位
        return res




209.长度最小的子数组
给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。

示例：
输入：s = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
提示：

1 <= target <= 10^9
1 <= nums.length <= 10^5
1 <= nums[i] <= 10^5

滑动窗口
接下来就开始介绍数组操作中另一个重要的方法：滑动窗口。
所谓滑动窗口，就是不断的调节子序列的起始位置和终止位置，从而得出我们要想的结果。

（版本一）滑动窗口法
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        l = len(nums)
        left = 0
        right = 0
        min_len = float('inf')
        cur_sum = 0 #当前的累加值

        while right < l:
            cur_sum += nums[right]

            while cur_sum >= s: # 当前累加值大于目标值
                min_len = min(min_len, right - left + 1)
                cur_sum -= nums[left]
                left += 1

            right += 1

        return min_len if min_len != float('inf') else 0



59.螺旋矩阵II
给定一个正整数 n，生成一个包含 1 到 n^2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
示例:
输入: 3 输出: [ [ 1, 2, 3 ], [ 8, 9, 4 ], [ 7, 6, 5 ] ]
class Solution:
    def generateMatrix(self, n: int) -> [[int]]:
        nums = [[0] * n for _ in range(n)]

        rows = len(nums)
        cols = len(nums[0])

        top = 0
        left = 0
        bottom = rows -1
        right = cols -1
        res = []

        count = 1
        while True:
            for i in range(left, right + 1):
                nums[top][i] = count
                count += 1
            top += 1
            if top > bottom:
                break

            for i in range(top, bottom + 1):
                nums[i][right] = count
                count += 1
            right -= 1
            if left > right:
                break


            for i in range(right, left -1 , -1):
                nums[bottom][i] = count
                count += 1
            bottom -= 1
            if top > bottom:
                break

            for i in range(bottom, top - 1 , -1):
                nums[i][left] = count
                count += 1
            left += 1
            if left > right:
                break
        return nums


904.水果成篮(opens new window)
你正在探访一家农场，农场从左到右种植了一排果树。这些树用一个整数数组 fruits 表示，其中 fruits[i] 是第 i 棵树上的水果种类 。

你想要尽可能多地收集水果。然而，农场的主人设定了一些严格的规矩，你必须按照要求采摘水果：

你只有两个篮子，并且每个篮子只能装单一类型的水果。每个篮子能够装的水果总量没有限制。
你可以选择任意一棵树开始采摘，你必须从每棵树（包括开始采摘的树）上恰好摘一个水果 。采摘的水果应当符合篮子中的水果类型。每采摘一次，你将会向右移动到下一棵树，并继续采摘。
一旦你走到某棵树前，但水果不符合篮子的水果类型，那么就必须停止采摘。
给你一个整数数组 fruits ，返回你可以收集的水果的最大数目。

题目很长有点繁琐，先中译中一下：
给定一个数组 fruit[ ]，找出该数组中连续的包含两个不同数字的最大子串，返回最大子串长度

如: fruits = [3,3,3,1,2,1,1,2,3,3,4] 输出：5 （因为最大子串为 [1,2,1,1,2]）

思路：
若该数组中只有一个数字，则最大子串为其本身，返回长度为1
设置左右两个指针和一个列表s，设置result用于存储最大字串长，该列表s初始化为给定数组的第一个元素，左右指针起始为0。若右指针所指的元素与前一位不同并且该元素不在s中，则将这个元素添加到s后，此时若s的长度大于2（即包含超过两个类别的数字）时，比较之前记录的result值与当前子串长度，取最大值更新result。移动左指针，若移动后的值和上一位相同，则再次向后移动左指针。
————————————————
版权声明：本文为CSDN博主「mingchen_peng」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。

#滑动窗口最大，且滑动窗口内只有两种数字
#核心：记录滑动窗口内的种类和出现的次数，这里我们用哈希表
#能用滑动窗口的题，都是要求在一段连续的子数组中
class Solution(object):
    def totalFruit(self, fruits):
        cnt = Counter()

        left = ans = 0
        for right, x in enumerate(fruits):#这里枚举是关键
            cnt[x] += 1
            while len(cnt) > 2:#种类超过2时就应该判断
                cnt[fruits[left]] -= 1#这里，先对left这个种类的个数-1
                if cnt[fruits[left]] == 0:#然后再判断这个种类滑动窗口内数量是不是为0
                    cnt.pop(fruits[left])#如果为0了就移除这个种类
                left += 1#然后顺势向前移动滑动窗口
            ans = max(ans, right - left + 1)

        return ans






76.最小覆盖子串(opens new window)
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 “” 。

注意：
对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。
示例 1：

输入：s = “ADOBECODEBANC”, t = “ABC”
输出：“BANC”
解释：最小覆盖子串 “BANC” 包含来自字符串 t 的 ‘A’、‘B’ 和 ‘C’。
示例 2：

输入：s = “a”, t = “a”
输出：“a”
解释：整个字符串 s 是最小覆盖子串。
示例 3:

输入: s = “a”, t = “aa”
输出: “”
解释: t 中两个字符 ‘a’ 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。
提示：

m == s.length
n == t.length
1 <= m, n <= 105
s 和 t 由英文字母组成


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # 目标字符记录
        need=collections.defaultdict(int)
        for c in t:
            need[c]+=1
        # 目标字符长度
        needCnt=len(t)
        left=0
        res=(0,float('inf'))
        for r,c in enumerate(s):
            if need[c]>0:
                # 如果need[c]>0,意味着窗口搜索到目标值
                needCnt-=1
            # 目标值被搜索到，则need相应value-1,即用以记录搜索到need中字符的个数；
            # =0代表搜索到的值和need值相同；<0代表need中同一个字符被多次搜索到；
            # >0,代表need中该字符没有全部被遍历出来
            need[c]-=1

            if needCnt==0: # 滑动窗口包含了所有T元素
                # 开始窗口函数内部的搜索
                while True:  # 移动滑动窗口右边界i，排除多余元素
                    c=s[left]
                    if need[c]==0:
                        # need[c]==0，break 情况，代表着这轮的滑动窗口不符合要求
                        break
                    # 窗口中字符和need相互抵消
                    need[c]+=1
                    left+=1
                if r-left <res[1]-res[0]:   #记录结果
                    res=(left,r)
                # 窗口函数内部计算完后，向下一步迭代
                # s[left]是目标集中t的字符，这时从窗口函数清除，
                # 则意味着窗口函数需要新的字符填充，即need[s[left]]和needCnt需要加一
                need[s[left]]+=1  # left增加一个位置，寻找新的满足条件滑动窗口
                needCnt+=1
                # 新一轮的右边界
                left+=1
        return '' if res[1]>len(s) else s[res[0]:res[1]+1]    #如果res始终没被更新过，代表无满足条件的结果







54.螺旋矩阵(opens new window)
题目
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

示例 ：

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        row=len(matrix)
        col=len(matrix[0])
        left,top=0,0
        right,bottom=col-1,row-1
        out=list()
        while True:
            for i in range(left,right+1):
                out.append(matrix[top][i])
            top+=1
            if top>bottom:
                break
            for i in range(top,bottom+1):
                out.append(matrix[i][right])
            right-=1
            if left>right:
                break
            for i in range(right,left-1,-1):
                out.append(matrix[bottom][i])
            bottom-=1
            if top>bottom:
                break
            for i in range(bottom,top-1,-1):
                out.append(matrix[i][left])
            left+=1
            if left>right:
                break
        return out


'''

