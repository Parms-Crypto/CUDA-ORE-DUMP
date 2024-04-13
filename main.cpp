#include <stdio.h>

// Forward declaration of CUDA kernel
extern "C" void keccakHash(unsigned char *input, unsigned char *output, unsigned int digestSize);

int main() {
    unsigned char input[] = "Hello, CUDA!";
    unsigned int digestSize = 256; // Choose the digest size (in bits)
    unsigned char *d_input, *d_output;

    // Allocate memory on device
    cudaMalloc((void**)&d_input, sizeof(input));
    cudaMalloc((void**)&d_output, digestSize / 8);

    // Copy input data to device
    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);

    // Launch kernel
    keccakHash<<<1, 1>>>(d_input, d_output, digestSize);

    // Copy output data back to host
    unsigned char output[digestSize / 8];
    cudaMemcpy(output, d_output, digestSize / 8, cudaMemcpyDeviceToHost);

    // Print hash
    printf("Keccak Hash: ");
    for (int i = 0; i < digestSize / 8; ++i) {
        printf("%02x", output[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
