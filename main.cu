#include <iostream>


//Each thread should handle one pixel
/*Params:
- Buffer, storing Pixel colors
- dimx, the height of the image in pixels
- dimy, the wdith of the image in pixels
*/
__global__
void render(float* buffer, int dimx, int dimy)
{
    int id = threadIdx.x; //should be in range(0,255)
    for(int i =0;i< dimy ; ++i) //i represents row number
    {
        auto pixel_id = dimx*3*i + 3*id; //Get index of the pixel into the buffer. ---> [0_r,0_g,0_b,1_r,1_b,1_g,...]
        buffer[pixel_id + 0] = float(id) / (dimx-1); //RGB values are offsets from pixel index
        buffer[pixel_id + 1] = 0.f;
        buffer[pixel_id + 2] = 0.f;
    }
}

int main()
{
    constexpr int image_width = 256;
    constexpr int image_height= 256;

    constexpr int num_pixels = image_height*image_width; //Total number of pixels
    auto buffer_size = 3*sizeof(float)* num_pixels; //Size of the cache. Multiply by 3 because each pixel holds an RGB value

    //Allocate cache
    float* buffer;
    cudaMallocManaged(&buffer, buffer_size); //Allocates on unified memory

    int block_size = 256;
    int num_blocks = 1;
    render<<<num_blocks,block_size>>>(buffer, image_height, image_width); //Render on the GPU
    cudaDeviceSynchronize();


    //Transfer from buffer to output stream
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height-1; j >= 0; --j) {
        std::cerr<<"\rScanlines Remaining: "<<j<<'\n';
        for (int i = 0; i < image_width; ++i) {
            auto pixel_id = 3*j*image_height + 3*i; //Get index of the pixel into the buffer. ---> [0_r,0_g,0_b,1_r,1_b,1_g,...]
            auto r = buffer[pixel_id + 0]; //RGB values are offsets from pixel index
            auto g = buffer[pixel_id + 1];
            auto b = buffer[pixel_id + 2];

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    cudaFree(buffer);//Deallocate unified memory
}
