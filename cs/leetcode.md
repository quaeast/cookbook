## python函数速记

[python buildin functions](https://www.programiz.com/python-programming/list)

```
list.append(item)
list.pop(index)
list.remove(item)
list.insert(index, item)
list.index(item)
list.clear()

set.add(item)
set.remove(item)

dict.remove
```

## 单调栈

下一个最大元素的问题，构造一个从栈头到栈底单调递增的单调栈即可。压入栈的信息通常是(index, value)元组队

LC739、LC496

##  单调队列

## 优先队列

```python
from heapq import *

heappush(heap, item)
heappop(heap) // 弹出最小值，即heap[0]
heappushpop(heap, item)
heapify(x) // 原地构建堆
heapreplace(heap, item)
```

## 股票问题

[中文链接](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/solutions/8610/yi-ge-fang-fa-tuan-mie-6-dao-gu-piao-wen-ti-by-lab/)

### 普遍形式，K不固定

`dp[i][k][s]`中`i`表示第`i`天结束，`k`表示还能进行`k`次交易，`s`表示持有状态`0`表示未持有`1`表示持有。

```python
base case：
dp[-1][k][0] = dp[i][0][0] = 0
dp[-1][k][1] = dp[i][0][1] = -infinity
dp[i][0][0] = 0
dp[i][0][1] = -infinity

状态转移方程：
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
```

```python
def dp(i, k, s):
    if i<0 or k<=0:
        if s==0:
            return 0
        elif s==1:
            return float("-inf")
    if s==0:
        return max(dp(i-1, k, 0), dp(i-1,k, 1)+prices[i])
    elif s==1:
        return max(dp(i-1, k, 1), dp(i-1, k-1, 0)-prices[i])
```

### 当k=1时，可以简化为

```python
dp[i][1][0] = max(dp[i-1][1][0], dp[i-1][1][1] + prices[i])
dp[i][1][1] = max(dp[i-1][1][1], dp[i-1][0][0] - prices[i]) 
            = max(dp[i-1][1][1], -prices[i])
解释：k = 0 的 base case，所以 dp[i-1][0][0] = 0。

现在发现 k 都是 1，不会改变，即 k 对状态转移已经没有影响了。
可以进行进一步化简去掉所有 k：
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], -prices[i])
```

### 当k=+infinity时，可以简化未

```python
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
            = max(dp[i-1][k][1], dp[i-1][k][0] - prices[i])

我们发现数组中的 k 已经不会改变了，也就是说不需要记录 k 这个状态了：
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
```

## 背包问题

### 01背包问题

例题：
LC416. 分割等和子集、LC494. 目标和

`dp[i][j]`表示从下标`[0-i]`中取物品，背包的容量为`j`

初始化条件：
$$
dp[i][0]=0\\
dp[0][j]=value[0]\ if\ j\ge weight[0]
$$


状态转移方程：
$$
dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])\ (j\ge weight[i])
$$
优化状态转移方程（这里似乎有一些问题）：
$$
dp[j]=max(dp[j],dp[j-weight[i]]+value[i])
$$

```
from  functools import cache

@cache
def dp(i, j):
    if j==0:
        return 0
    if j<0:
        return -float('inf')
    if i==0:
        return value[0] if j>=weight[0] else 0
    return max(dp(i-1, j),dp(i-1, j-weight[i])+value[i])
```

## 二叉堆



## 二分

LC34.在排序数组中查找元素的第一个和最后一个位置

**寻找下界**

如果我们在二分查找的过程中，**不断右移 `left`，左移 `right`，使得所有「小于」`target` 的元素都在 `left` 左侧，所有「大于等于」`target` 的元素都在 `right` 右侧，那么当区间为空时，`left` 就是要查找的下界**：

`while` 中的 `left<=right` 表示“区间不为空”，其中`if`中的条件就表示`left`和`right`的具体含义，下面代码使用两侧闭区间的方式实现。

```js
//下界
func LowerBound(nums []int, target int) int {
    left, right := 0, len(nums)-1
    for left <= right {
        mid := left + (right-left) >> 1
        if nums[mid] >= target { // >=target的都在right右侧
            right = mid - 1
        } else { // <target的都在left左侧
            left = mid + 1
        }
    }
    return left // 返回下界的下标
}
```

**寻找上界**

定义满足 `x ≤ target` 的**最后一个元素**为「上界」。给定一个 `target`，要求返回升序数组中上界的下标。比如：对于数组 `[0,1,2,3,4]`，当 `target=3` 时，返回下标 `2`；当 `target=5` 时，返回下标 `4`。

根据上界和下界的定义，我们可以发现：**上界和「互补的」下界是相邻的，并且 `上界 = 下界 - 1`**。比如 `x ≤ target` 的上界和 `x > target` 的下界相邻。因此，**所有找上界的问题，都可以转换为「互补的」找下界的问题。**

```js
//上界
func LowerBound(nums []int, target int) int {
    left, right := 0, len(nums)-1
    for left <= right {
        mid := left + (right-left) >> 1
        if nums[mid] > target {
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    return left // 或者返回 left-1
}
```

## 回溯

LC78. 子集、46. 全排列

## 组合

LC39. 组合总和、LC322. 零钱兑换

## 并查集

LC399. 除法求值[题解](https://leetcode.cn/problems/evaluate-division/solutions/1367493/by-flourish-rt-omgz/)

`find`函数的基本思路是对于某个$x$，向前寻找他的最小不可分的$f_x$，并得到他们之间的倍数关系$w_x$：

$$
x = w_x \times f_x\\
y = w_y \times f_y\\
$$

`union`函数的基本思路是存储$f_y$和$f_x$的倍数关系：

$$
y = vx\\
f_y = \frac{vw_xf_x}{w_y}
$$

并查集存储的数据结构为`{'x':(w, 'y')}`

整体的代码为：

```python
class UnionFind:
    def __init__(self):
        # {'node_name_i': (w_ij, node_name_j)}
        self.nodes = {}
    
    # return x = w_x * f_x
    def find(self, x):
        if x not in self.nodes:
            self.nodes[x] = (1.0, x)
        cur_x = x
        w = 1.0
        while cur_x != self.nodes[cur_x][1]:
            w *= self.nodes[cur_x][0]
            cur_x = self.nodes[cur_x][1]
        self.nodes[x] = (w, cur_x)
        return self.nodes[x]

    # y = wx
    def union(self, y, x, w):
        y_node = self.find(y)
        x_node = self.find(x)
        self.nodes[y_node[1]] = (w*x_node[0]/y_node[0], x_node[1])
        

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        unif = UnionFind()
        for n, w in zip(equations, values):
            unif.union(n[0], n[1], w)
        res = []
        for y, x in queries:
            if x not in unif.nodes or y not in unif.nodes:
                res.append(-1.0)
                continue
            wy, fy = unif.find(y)
            wx, fx = unif.find(x)
            if fx==fy:
                res.append(wy/wx)
            else:
                res.append(-1.0)
        return res

```

