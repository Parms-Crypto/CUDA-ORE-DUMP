use tokio::net::TcpStream;
use std::io::{self, Read, Write};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:8080".parse::<SocketAddr>()?;
    let mut stream = TcpStream::connect(&addr).await?;
    println!("Connected to server");

    // Send data to the server
    let nonce = "123456";
    stream.write_all(nonce.as_bytes()).await?;
    stream.flush().await?;

    // Read hash from the server
    let mut buffer = [0; 1024];
    let bytes_read = stream.read(&mut buffer).await?;
    let hash = String::from_utf8_lossy(&buffer[..bytes_read]);
    println!("Received hash from server: {:?}", hash);

    Ok(())
}
