use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio::stream::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// Mockup of a function to simulate Keccak hashing
fn simulate_keccak_hash(input: &str) -> String {
    // Simulated hash function implementation
    format!("0000{}", rand::random::<u32>().to_string())
}

async fn handle_client(mut stream: TcpStream) {
    println!("Client connected: {:?}", stream.peer_addr().unwrap());

    // Read data from the client
    let mut buffer = [0; 1024];
    let bytes_read = stream.read(&mut buffer).await.unwrap();
    let data = String::from_utf8_lossy(&buffer[..bytes_read]);
    println!("Received data from client: {:?}", data);

    // Simulate Keccak hashing
    let hash = simulate_keccak_hash(&data);

    // Echo the hash back to the client
    stream.write_all(hash.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:8080".parse::<SocketAddr>()?;

    let listener = TcpListener::bind(&addr).await?;
    println!("Server listening on {}", addr);

    while let Ok((stream, _)) = listener.accept().await {
        tokio::spawn(handle_client(stream));
    }

    Ok(())
}
