import socket


def read_data_adc():
    print("flag")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Création d'un socket TCP
    client.connect(("192.168.33.30", 4098))
    print("flag")
    file = open("data.txt", "wb")
    i = 0
    while i != 100:
        print("data : ")
        print(client.recv(1024))
        file.write(client.recv(16))
        i += 1
    client.close()


if __name__ == '__main__':
    read_data_adc()