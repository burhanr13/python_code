import socket

s = socket.socket()
print('Socket Created')

s.bind(('localhost',9999))

s.listen(3)
print('Waiting for connections')

while True:
    c, address = s.accept()

    name = c.recv(1024).decode()

    print(f"Connected with {name}")
    
    c.send(bytes('nice','utf-8'))
    c.close()
