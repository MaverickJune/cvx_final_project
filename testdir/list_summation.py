list_1 = [1,2,3,4,5]
list_2 = [6,7,8,9,10]

list_3 = [sum(item) for item in zip(list_1, list_2)]
print(list_3)