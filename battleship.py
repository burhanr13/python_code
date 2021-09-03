import random as r


def createBoard(size):
    board = [["O"]*size]*size
    return board


def printBoard(board):
    for i in board:
        print(*i, end="\n")


def createShips(board, numofships):
    ships = []
    for i in range(numofships):
        x = []
        x = [r.randint(0, len(board)-1), r.randint(0, len(board)-1)]
        while x in ships:
            x = [r.randint(0, len(board)-1), r.randint(0, len(board)-1)]
        ships.append(x)
    return ships


def guess(board, ship, sunk, guesslist):
    guessx = int(input("guess horiz value: ")) - 1
    guessy = int(input("guess vert value: ")) - 1
    if (guessx in range(0, len(board))) & (guessy in range(0, len(board))):
        if [guessx, guessy] in guesslist:
            print("You guessed that already")
        else:
            if [guessx, guessy] in ship:
                print("Hit!")
                board[guessy][guessx] = "X"
                sunk[ship.index([guessx, guessy])] = 1
                guesslist.append([guessx, guessy])
            else:
                print("Miss")
                board[guessy][guessx] = "M"
                guesslist.append([guessx, guessy])
    else:
        print("Not in range")
    return board, sunk, guesslist


def runGame(boardSize, numOfShips):
    board = createBoard(boardSize)
    ships = createShips(board, numOfShips)
    shipsSunk = [0]*len(ships)
    guesslist = []
    printBoard(board)
    while shipsSunk != [1]*len(ships):
        board, shipsSunk, guesslist = guess(board, ships, shipsSunk, guesslist)
        printBoard(board)
    print("You win!")


runGame(5, 5)
