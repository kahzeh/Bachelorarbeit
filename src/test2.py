with open('uniquenames.txt', 'r') as f:
    lines = f.readlines()

lines = [x.strip() for x in lines]

lines.sort()

dictionary = {}
i = 0
j = 2000

while j < 3000:
    dictionary[str(j)] = i
    j += 1
    i+=1

print(dictionary)