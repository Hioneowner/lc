'''
165. 比较版本号
比较两个版本号 version1 和 version2。
如果 version1 > version2 返回 1，如果 version1 < version2 返回 -1， 除此之外返回 0。

你可以假设版本字符串非空，并且只包含数字和 . 字符。

. 字符不代表小数点，而是用于分隔数字序列。

例如，2.5 不是“两个半”，也不是“差一半到三”，而是第二版中的第五个小版本。

你可以假设版本号的每一级的默认修订版号为 0。例如，版本号 3.4 的第一级（大版本）和第二级（小版本）修订号分别为 3 和 4。其第三级和第四级修订号均为 0。

示例 1:

输入: version1 = “0.1”, version2 = “1.1”
输出: -1
示例 2:

输入: version1 = “1.0.1”, version2 = “1”
输出: 1
示例 3:

输入: version1 = “7.5.2.4”, version2 = “7.5.3”
输出: -1
示例 4：

输入：version1 = “1.01”, version2 = “1.001”
输出：0
解释：忽略前导零，“01” 和 “001” 表示相同的数字 “1”。
示例 5：

输入：version1 = “1.0”, version2 = “1.0.0”
输出：0
解释：version1 没有第三级修订号，这意味着它的第三级修订号默认为 “0”。

提示：

版本字符串由以点 （.） 分隔的数字字符串组成。这个数字字符串可能有前导零。
版本字符串不以点开始或结束，并且其中不会有两个连续的点。
思路
如果version1与version2长度相同的话，依次比较每一级修订号，如果有一级不相同的话则可以直接出结果，
如果version1比version2长得话，比较version2与version1前半部分，如果有一级不不相同的话直接出结果，如果相同的话观察version1的后半部分，不全为0的话比version1较新



class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        str1 = version1.split('.')
        str2 = version2.split('.')
        m = len(str1)
        n = len(str2)
        i = j = 0
        while i < m or j < n:
            x = int(str1[i]) if i < m else 0
            y = int(str2[j]) if j < n else 0
            if x == y:
                i += 1
                j += 1
                continue
            elif x > y:
                return 1
            else:
                return -1
            i += 1
            j += 1
        return 0



41. 缺失的第一个正数
给定一个未排序的整数数组，找出其中没有出现的最小的正整数。

示例 1:

输入: [1,2,0]
输出: 3
示例 2:

输入: [3,4,-1,1]
输出: 2
示例 3:

输入: [7,8,9,11,12]
输出: 1

思路
原地修改数组，把每个元素放到相应位置，比如把1放到第1个位置，把2放到第2个位置，放好后顺序遍历数组，看哪一个位置的数不对，返回那个位置就对了
移动的方法可以通过遍历数组，将数组中当前元素 与应该放到的位置的元素互换，但这样会有一个问题，
如[3,4,-1,1]
在遍历到第二个元素4时，会将4与1交换，随然4被换到了合适的位置，但1并没有，所以也要将换过来的元素放到合适的位置
如果数组中每个元素都放到了合适的位置，那返回数组长度+1就行了
 def firstMissingPositive(self, nums):
        if nums==[]:return 1
        for i in range(len(nums)):

            while nums[i]>0 and nums[i]<=len(nums) and nums[i]!=i+1 and nums[nums[i]-1]!=nums[i]:
                w=nums[i]
                nums[w-1],nums[i]=nums[i],nums[w-1]

        for i in  range(len(nums)):
            if nums[i]!=i+1:return i+1
        return len(nums)+1




32. 最长有效括号
给定一个只包含 ‘(’ 和 ‘)’ 的字符串，找出最长的包含有效括号的子串的长度。

示例 1:

输入: “(()”
输出: 2
解释: 最长有效括号子串为 “()”
示例 2:

输入: “)()())”
输出: 4
解释: 最长有效括号子串为 “()()”
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        if not s:return 0
        stack = [-1]
        length = len(s)
        res = 0
        for l in range(length):
            if s[l] == '(':
                stack.append(l)
            else:
                stack.pop()
                if not stack:
                    stack.append(l)
                else:
                    res = max(res,l-stack[-1])
        return res




43. 字符串相乘
给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

示例 1:

输入: num1 = “2”, num2 = “3”
输出: “6”
示例 2:

输入: num1 = “123”, num2 = “456”
输出: “56088”
说明：

num1 和 num2 的长度小于110。
num1 和 num2 只包含数字 0-9。
num1 和 num2 均不以零开头，除非是数字 0 本身。
不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理。


class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        ans = 0
        f1 = 1
        for i in range(len(num1)-1, -1, -1):
            n1 = int(num1[i]) * f1

            f2 = 1
            for j in range(len(num2)-1, -1, -1):
                n2 = int(num2[j]) * f2
                f2 *= 10
                ans += n1 * n2

            f1 *= 10

        return str(ans)




155. 最小栈
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack:
            self.min_stack.append(val)
        elif val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        val = self.stack[-1]
        self.stack.pop()

        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]






151. 翻转字符串里的单词
给定一个字符串，逐个翻转字符串中的每个单词。

示例 1：
输入: “the sky is blue”
输出: “blue is sky the”

示例 2：
输入: " hello world! "
输出: “world! hello”
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。

示例 3：
输入: “a good example”
输出: “example good a”
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        arr = s.split(' ')
        return ' '.join([s for s in arr[::-1] if s != ""])





129. 求根到叶子节点数字之和
给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。

例如，从根到叶子节点路径 1->2->3 代表数字 123。

计算从根到叶子节点生成的所有数字之和。

说明: 叶子节点是指没有子节点的节点。

示例 1:

输入: [1,2,3]
    1
   / \
  2   3

输出: 25
解释:
从根到叶子节点路径 1->2 代表数字 12.
从根到叶子节点路径 1->3 代表数字 13.
因此，数字总和 = 12 + 13 = 25.
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return self.helper(root, 0)


    def helper(self, root, t):

        if root.left is None and root.right is None:
            return t + root.val
        l = 0
        r = 0
        if root.left:
            l = self.helper(root.left, 10*(t + root.val))

        if root.right:
            r = self.helper(root.right, 10*(t + root.val))


        return l + r



543. 二叉树的直径
题目描述：
给你一棵二叉树的根节点，返回该树的 直径 。

二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
两节点之间路径的 长度 由它们之间边数表示。
      1
     / \
    2   3
   / \
  4   5

示例 1：

输入：root = [1,2,3,4,5]
输出：3
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。

示例 2：
输入：root = [1,2]
输出：1

思路：
对于该节点的左儿子向下遍历经过最多的节点数 L （即以左儿子为根的子树的深度） 和其右儿子向下遍历经过最多的节点数 R （即以右儿子为根的子树的深度），那么以该节点为起点的路径经过节点数的最大值即为 L+R+1。

算法流程：
定义一个递归函数 depth(node) 计算 node，函数返回该节点为根的子树的深度。先递归调用左儿子和右儿子求得它们为根的子树的深度 L 和 R ，则该节点为根的子树的深度即为max(L,R)+1，该节点的node值为L+R+1。

递归搜索每个节点并设一个全局变量 ans记录node的最大值，最后返回 ans-1 即为树的直径。
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def depth(self, root):
        if root is None:
            return 0

        l = self.depth(root.left)
        r = self.depth(root.right)

        self.ans = max(l+r + 1, self.ans)
        return max(l, r) + 1

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.ans = 1
        self.depth(root)

        return self.ans - 1



补充题：手撕归并排序
1、原理
　　归并排序是分治法的典型应用。分治法（Divide-and-Conquer）：将原问题划分成 n 个规模较小而结构与原问题相似的子问题；递归地解决这些问题，然后再合并其结果，就得到原问题的解。从上图看分解后的数列很像一个二叉树。

归并排序采用分而治之的原理：

将一个序列从中间位置分成两个序列；
在将这两个子序列按照第一步继续二分下去；
直到所有子序列的长度都为1，也就是不可以再二分截止。这时候再两两合并成一个有序序列即可。

2、举例
对以下数组进行归并排序：　
[11, 99, 33 , 69, 77, 88, 55, 11, 33, 36,39, 66, 44, 22]
首先，进行数组分组，即

[11, 99, 33 , 69, 77, 88, 55], [ 11, 33, 36,39, 66, 44, 22]
[11, 99, 33] , [69, 77, 88, 55], [ 11, 33, 36], [39, 66, 44, 22]
[11], [99, 33] , [69, 77], [88, 55],[ 11], [33, 36],[39, 66], [44, 22]
直到所有子序列的长度都为1，也就是不可以再二分截止。
[11], [99], [33] , [69], [77], [88], [55],[ 11], [33], [36],[39], [66], [44], [22]
这时候再两两合并成一个有序序列即可。

[11],[33,99],[69,77],[55,88],[11],[33,36],[39,66],[22,44]
[11,33,99],[55,69,77,88],[11,33,36],[22,39,44,66]
[11,33,55,69,77,88,99],[11,22,33,36,39,44,66]
4、最终排序
[11, 11, 22, 33, 33, 36, 39, 44, 55, 66, 69, 77, 88, 99]
二、代码
def merge_sort(arr):
    """归并排序"""
    if len(arr) == 1:
        return arr
    # 使用二分法将数列分两个
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    # 使用递归运算
    return marge(merge_sort(left), merge_sort(right))


def marge(left, right):
    """排序合并两个数列"""
    result = []
    # 两个数列都有值
    while len(left) > 0 and len(right) > 0:
        # 左右两个数列第一个最小放前面
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    # 只有一个数列中还有值，直接添加
    result += left
    result += right
    return result

merge_sort([11, 99, 33 , 69, 77, 88, 55, 11, 33, 36,39, 66, 44, 22])

# 返回结果[11, 11, 22, 33, 33, 36, 39, 44, 55, 66, 69, 77, 88, 99]

'''
