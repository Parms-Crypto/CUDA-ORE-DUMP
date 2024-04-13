const net = require('net');

// Connect to the server
const client = net.createConnection({ port: 8080, host: '127.0.0.1' }, () => {
    console.log('Connected to server');

    // Send data to the server (in this case, a nonce to hash)
    const nonce = '123456';
    client.write(nonce);
});

// Handle data from the server
client.on('data', (data) => {
    console.log('Received hash from server:', data.toString());

    // Process the received hash as needed
});

// Handle server disconnection
client.on('end', () => {
    console.log('Disconnected from server');
});

// Handle errors
client.on('error', (err) => {
    console.error('Client error:', err);
});
