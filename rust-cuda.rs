extern crate rust_cuda;
use rust_cuda::launch;

fn main() {
    let n = 1024;
    let mut data = vec![1.0f32; n];

    // Launch CUDA kernel
    unsafe {
        launch!(my_kernel<<<1, 1024>>>(data.as_mut_ptr(), n as i32));
    }

    // Check results
    println!("{:?}", data);
}
