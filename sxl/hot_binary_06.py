'''
242.有效的字母异位词
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
示例 1: 输入: s = “anagram”, t = “nagaram” 输出: true
示例 2: 输入: s = “rat”, t = “car” 输出: false

说明: 你可以假设字符串只包含小写字母。
class Solution(object):
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter
        a_count = Counter(s)
        b_count = Counter(t)
        return a_count == b_count


两个数组的交集
题意：给定两个数组，编写一个函数来计算它们的交集
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    # 使用哈希表存储一个数组中的所有元素
        table = {}
        for num in nums1:
            table[num] = table.get(num, 0) + 1
        # 使用集合存储结果
        res = set()
        for num in nums2:
            if num in table:
                res.add(num)
                del table[num]
        return list(res)


class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))



第202题. 快乐数
编写一个算法来判断一个数 n 是不是快乐数。
「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果 可以变为 1，那么这个数就是快乐数。
如果 n 是快乐数就返回 True ；不是，则返回 False 。

示例：
输入：19
输出：true
解释：
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1

#思路
这道题目看上去貌似一道数学问题，其实并不是！
题目中说了会 无限循环，那么也就是说求和的过程中，sum会重复出现，这对解题很重要！
正如：关于哈希表，你该了解这些！ (opens new window)中所说，当我们遇到了要快速判断一个元素是否出现集合里的时候，就要考虑哈希法了。
所以这道题目使用哈希法，来判断这个sum是否重复出现，如果重复了就是return false， 否则一直找到sum为1为止。
判断sum是否重复出现就可以使用unordered_set。
还有一个难点就是求和的过程，如果对取数值各个位上的单数操作不熟悉的话，做这道题也会比较艰难。
class Solution:
    def isHappy(self, n: int) -> bool:
        record = set()

        while True:
            n = self.get_sum(n)
            if n == 1:
                return True
            # 如果中间结果重复出现，说明陷入死循环了，该数不是快乐数
            if n in record:
                return False
            else:
                record.add(n)

    def get_sum(self,n: int) -> int:
        new_num = 0
        while n:
            n, r = divmod(n, 10)
            new_num += r ** 2
        return new_num




1. 两数之和
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
示例:
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        records = dict()

        for index, value in enumerate(nums):
            if target - value in records:   # 遍历当前元素，并在map中寻找是否有匹配的key
                return [records[target- value], index]
            records[value] = index    # 如果没找到匹配对，就把访问过的元素和下标加入到map中
        return []

49.字母异位词分组
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

示例 1:
输入: strs =
[“eat”, “tea”, “tan”, “ate”, “nat”, “bat”]
输出: [[“bat”],[“nat”,“tan”],[“ate”,“eat”,“tea”]]

示例 2:
输入: strs =
[“”]
输出: [[“”]]

示例 3:
输入: strs =
[“a”]
输出: [[“a”]]

提示：
1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] 仅包含小写字母
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
table = {}
for s in strs:
    s_ = "".join(sorted(s))
    if s_ not in table:
        table[s_] = [s]
    else:
        table[s_].append(s)
print(list(table.values()))




438.找到字符串中所有字母异位词
给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。
字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。

说明：
字母异位词指字母相同，但排列不同的字符串。
不考虑答案输出的顺序。

示例 1:
输入:
s: “cbaebabacd” p: “abc”
输出:
[0, 6]
解释:
起始索引等于 0 的子串是 “cba”, 它是 “abc” 的字母异位词。
起始索引等于 6 的子串是 “bac”, 它是 “abc” 的字母异位词。

示例 2:
输入:
s: “abab” p: “ab”
输出:
[0, 1, 2]
解释:
起始索引等于 0 的子串是 “ab”, 它是 “ab” 的字母异位词。
起始索引等于 1 的子串是 “ba”, 它是 “ab” 的字母异位词。
起始索引等于 2 的子串是 “ab”, 它是 “ab” 的字母异位词。


class Solution(object):
    def findAnagrams(self, s, p):
        lenp = len(p)
        poslist = []
        p_sorted = sorted(p)
        for i in range(0,len(s)-len(p)+1):
            s_part = s[i:i+lenp]
            s_part_sorted = sorted(s_part)
            if s_part_sorted==p_sorted:
                poslist.append(i)
        return poslist


LeetCode：350—— 两个数组的交集 II
概述：给你两个整数数组 nums1 和 nums2 ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]

# 排序+双指针
# 先排序，然后指针分别指向两个列表进行判断，如果重复跳过即可。
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        length_1, length_2 = len(nums1), len(nums2)
        intersection = []
        index_1, index_2 = 0, 0
        while index_1 < length_1 and index_2 < length_2:
            if nums1[index_1] < nums2[index_2]:
                index_1 += 1
            elif nums1[index_1] > nums2[index_2]:
                index_2 += 1
            else:
                intersection.append(nums1[index_1])
                index_1 += 1
                index_2 += 1
        return intersection



'''