n = float(input("Enter a sequence \n"))
numbers_entered = []
while n != 0:
    n = float(input())
    if len(str(int(n))) == 2:
        numbers_entered.append(n)
if len(numbers_entered) > 0:
    print(round(sum(numbers_entered) / len(numbers_entered), 1))
else:
    print("NO")
