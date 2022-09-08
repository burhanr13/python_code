# a = input()

# b = ""

# c = "wordle"

# for ind,i in enumerate(a):
#     b += chr((ord(i) - ord('a') - (ord(c[ind%6]) - ord('a')))%26 + ord('a'))

# print(b)

word = "knoll"

while True:
    guess = input()
    outstr = ""
    for ind, i in enumerate(guess):
        if word[ind] == i:
            outstr += "🟩"
        elif i in word:
            outstr += "🟨"
        else:
            outstr += "⬛"
    print(outstr)
    if guess == word:
        print('win')
        break
