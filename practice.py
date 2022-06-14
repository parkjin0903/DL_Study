from math import gcd
from math import sqrt

input_list = list()
minus_list = list()
gcd_list = list()
gcd_num = 1

total_number = int(input()) # 3

for _ in range(total_number):
  input_num = int(input())
  input_list.append(input_num) # [6 34 38]

input_list.sort()

for index_num in range(total_number-1):
  minus_list.append(abs(input_list[index_num]-input_list[index_num+1])) # [28 4]

minus_list.sort() # [4, 28]

gcd_num = minus_list[0]
 
for index_minus in range(1, len(minus_list)): # 0
   gcd_num = gcd(gcd_num, minus_list[index_minus]) # gcd_num = 4

gcd_list.append(gcd_num)

for index_gcd in range(2, int(sqrt(gcd_num))+1): # 0 1 2
  if gcd_num % index_gcd == 0 :
    gcd_list.append(index_gcd)
    gcd_list.append(gcd_num // index_gcd)

answer = list(set(gcd_list))
answer.sort()

print(*answer)