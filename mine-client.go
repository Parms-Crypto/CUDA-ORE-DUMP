package main

import (
	"fmt"
	"net"
)

func main() {
	conn, err := net.Dial("tcp", "127.0.0.1:8080")
	if err != nil {
		fmt.Printf("Error connecting to server: %s\n", err)
		return
	}
	defer conn.Close()

	fmt.Println("Connected to server")

	// Send data to the server
	nonce := "123456"
	_, err = conn.Write([]byte(nonce))
	if err != nil {
		fmt.Printf("Error writing to server: %s\n", err)
		return
	}

	// Read hash from the server
	buffer := make([]byte, 1024)
	n, err := conn.Read(buffer)
	if err != nil {
		fmt.Printf("Error reading from server: %s\n", err)
		return
	}
	hash := string(buffer[:n])
	fmt.Printf("Received hash from server: %s\n", hash)
}
