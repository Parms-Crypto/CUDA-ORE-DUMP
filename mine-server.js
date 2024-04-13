const net = require('net');

// Mockup of a function to simulate Keccak hashing
function simulateKeccakHash(input) {
    // Simulated hash function implementation
    return "0000" + Math.random().toString(16).slice(2, 66); // Simulate 256-bit hash output with leading zeros
}

// Create a TCP server
const server = net.createServer((socket) => {
    console.log('Client connected:', socket.remoteAddress, socket.remotePort);

    // Handle data from the client
    socket.on('data', (data) => {
        console.log('Received data from client:', data.toString());

        // Simulate Keccak hashing
        const hash = simulateKeccakHash(data.toString());

        // Echo the hash back to the client
        socket.write(hash);
    });

    // Handle client disconnection
    socket.on('end', () => {
        console.log('Client disconnected:', socket.remoteAddress, socket.remotePort);
    });

    // Handle errors
    socket.on('error', (err) => {
        console.error('Socket error:', err);
    });
});

// Start the server and listen on port 8080
server.listen(8080, () => {
    console.log('Server listening on port 8080');
});
