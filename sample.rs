extern "C" {
    fn keccakHash(input: *const u8, output: *mut u8, digestSize: u32);
}

fn main() {
    let input = b"Hello, CUDA!";
    let digest_size = 256; // Choose the digest size (in bits)
    let mut output = vec![0; digest_size / 8];

    // Call the CUDA function
    unsafe {
        keccakHash(input.as_ptr(), output.as_mut_ptr(), digest_size);
    }

    // Print hash
    print!("Keccak Hash: ");
    for byte in output {
        print!("{:02x}", byte);
    }
    println!();
}
