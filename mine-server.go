package main

import (
	"fmt"
	"net"
)

// Mockup of a function to simulate Keccak hashing
func simulateKeccakHash(input string) string {
	// Simulated hash function implementation
	return "0000" + "mockedhash" // Simulate 256-bit hash output with leading zeros
}

func handleClient(conn net.Conn) {
	defer conn.Close()
	fmt.Printf("Client connected: %s\n", conn.RemoteAddr())

	// Read data from the client
	buffer := make([]byte, 1024)
	n, err := conn.Read(buffer)
	if err != nil {
		fmt.Printf("Error reading from client: %s\n", err)
		return
	}
	data := string(buffer[:n])
	fmt.Printf("Received data from client: %s\n", data)

	// Simulate Keccak hashing
	hash := simulateKeccakHash(data)

	// Echo the hash back to the client
	_, err = conn.Write([]byte(hash))
	if err != nil {
		fmt.Printf("Error writing to client: %s\n", err)
	}
}

func main() {
	listener, err := net.Listen("tcp", "127.0.0.1:8080")
	if err != nil {
		fmt.Printf("Error starting server: %s\n", err)
		return
	}
	defer listener.Close()

	fmt.Println("Server listening on port 8080")

	for {
		conn, err := listener.Accept()
		if err != nil {
			fmt.Printf("Error accepting connection: %s\n", err)
			continue
		}
		go handleClient(conn)
	}
}
