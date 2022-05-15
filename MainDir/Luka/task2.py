sequence_length = int(input("Sequence length\n"))
sequence = []
for i in range(sequence_length):
    number = input()
    if number[-1] == "8":
        sequence.append(int(number))

print(sum(sequence))
