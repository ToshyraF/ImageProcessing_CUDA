//
//  imgproc_main.cpp
//  
//
//  Created by Nathaniel Lewis on 3/8/12.
//  Copyright (c) 2012 E1FTW Games. All rights reserved.
//

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define Blur 5
// GPU constant memory to hold our kernels (extremely fast access time)
__constant__ float convolutionKernelStore[1024];

/**
 * Convolution function for cuda.  Destination is expected to have the same width/height as source, but there will be a border
 * of floor(kWidth/2) pixels left and right and floor(kHeight/2) pixels top and bottom
 *
 * @param source      Source image host pinned memory pointer
 * @param width       Source image width
 * @param height      Source image height
 * @param paddingX    source image padding along x 
 * @param paddingY    source image padding along y
 * @param kOffset     offset into kernel store constant memory 
 * @param kWidth      kernel width
 * @param kHeight     kernel height
 * @param destination Destination image host pinned memory pointer
 */
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kWidth, int kHeight, unsigned char *destination)
{
    // Calculate our pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float sum = 0.0;
    int   pWidth = kWidth/2;
    int   pHeight = kHeight/2;

    // Only execute for valid pixels
    if(x >= pWidth+paddingX &&
       y >= pHeight+paddingY &&
       x < (blockDim.x * gridDim.x)-pWidth-paddingX &&
       y < (blockDim.y * gridDim.y)-pHeight-paddingY)
    {
        for(int j = -pHeight; j <= pHeight; j++)
        {
            for(int i = -pWidth; i <= pWidth; i++)
            {
                // Sample the weight for this location
                int ki = (i+pWidth);
                int kj = (j+pHeight);
                float w  = convolutionKernelStore[(kj * kWidth) + ki + kOffset];

        
                sum += w * float(source[((y+j) * width) + (x+i)]);
            }
        }
    }
    
    // Average the sum
    destination[(y * width) + x] = (unsigned char) sum;
}

__global__ void Dilate(unsigned char *source, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kWidth, int kHeight, unsigned char *destination)
{
   // Calculate our pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    bool check = false ; 
    float sum = 0.0;
    int   pWidth = kWidth/2;
    int   pHeight = kHeight/2;

    // Only execute for valid pixels
    if(x >= pWidth+paddingX &&
       y >= pHeight+paddingY &&
       x < (blockDim.x * gridDim.x)-pWidth-paddingX &&
       y < (blockDim.y * gridDim.y)-pHeight-paddingY)
    {
        for(int j = -pHeight; j <= pHeight; j++)
        {
            for(int i = -pWidth; i <= pWidth; i++)
            {           
                if(float(source[((y+j) * width) + (x+i)]) != 0){
                    sum = 255;
                    break;
                } 
                else{
                    float w  = convolutionKernelStore[(j * kWidth) + i + kOffset];
                    sum += w*float(source[((y+j) * width) + (x+i)]);  
                }
                
            }
            if(check == true){
                check = false ;
                break;
            }
        }
    }
    if(sum != 0){
        sum = 255;
    }
    // Average the sum
    destination[(y * width) + x] = (unsigned char) sum;
}
__global__ void Erode(unsigned char *source, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kWidth, int kHeight, unsigned char *destination)
{
   // Calculate our pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    bool check = false ;   
    float sum = 0.0;
    int   pWidth = kWidth/2;
    int   pHeight = kHeight/2;

    // Only execute for valid pixels
    if(x >= pWidth+paddingX &&
       y >= pHeight+paddingY &&
       x < (blockDim.x * gridDim.x)-pWidth-paddingX &&
       y < (blockDim.y * gridDim.y)-pHeight-paddingY)
    {
        for(int j = -pHeight; j <= pHeight; j++)
        {
            for(int i = -pWidth; i <= pWidth; i++)
            {   
                if(float(source[((y+j) * width) + (x+i)]) == 0){
                    sum = 0;
                    check = true;
                    break ;
                }
                else{
                    float w  = convolutionKernelStore[(j * kWidth) + i + kOffset];
                    sum += w*float(source[((y+j) * width) + (x+i)]);   
                }        
                   
            }
            if(check == true){
                check = false ;
                break;

            }
        }
    }
    if(sum != 0){
        sum = 255;
    }
    // Average the sum
    destination[(y * width) + x] = (unsigned char) sum;
}
// converts the pythagoran theorem along a vector on the GPU
__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float af = float(a[idx]);
    float bf = float(b[idx]);

    c[idx] = (unsigned char) sqrtf(af*af + bf*bf);
    // c[idx] = (unsigned char) abs(af*af - bf*bf);
}

__global__ void theshould(unsigned char *a, unsigned char *c, int min)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float af = float(a[idx]);
    if(af > min ){
        c[idx] = 255;
    }
    else{
        c[idx] = 0;
    }
    // c[idx] = (unsigned char) abs(af*af - bf*bf);
}

// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr)
{   
    unsigned char *ptr = NULL;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(devicePtr, ptr, 0);
    return ptr;
}

int main (int argc, char** argv)
{
    // Open a webcamera
    //uchar3 test ;
    cv::VideoCapture camera(1);
    cv::Mat          frame;
    if(!camera.isOpened()) 
        return -1;
    
    // Create the capture windows
    cv::namedWindow("Source");
    cv::namedWindow("Greyscale");
    cv::namedWindow("Blurred");
    cv::namedWindow("Sobel");

    // Create the cuda event timers 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create the gaussian kernel (sum = 159)
    // const float gaussianKernel5x5[25] = 
    // {
    //     2.f/159.f,  4.f/159.f,  5.f/159.f,  4.f/159.f, 2.f/159.f,   
    //     4.f/159.f,  9.f/159.f, 12.f/159.f,  9.f/159.f, 4.f/159.f,   
    //     5.f/159.f, 12.f/159.f, 15.f/159.f, 12.f/159.f, 5.f/159.f,   
    //     4.f/159.f,  9.f/159.f, 12.f/159.f,  9.f/159.f, 4.f/159.f,   
    //     2.f/159.f,  4.f/159.f,  5.f/159.f,  4.f/159.f, 2.f/159.f,   
    // };
    const float gaussianKernel5x5[Blur*Blur] = 
    {
        1.f/25.f,  1.f/25.f, 1.f/25.f,  1.f/25.f, 1.f/25.f,
        1.f/25.f,  1.f/25.f, 1.f/25.f,  1.f/25.f, 1.f/25.f,
        1.f/25.f,  1.f/25.f, 1.f/25.f,  1.f/25.f, 1.f/25.f,
        1.f/25.f,  1.f/25.f, 1.f/25.f,  1.f/25.f, 1.f/25.f,
        1.f/25.f,  1.f/25.f, 1.f/25.f,  1.f/25.f, 1.f/25.f,
    };

    cudaMemcpyToSymbol(convolutionKernelStore, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);
    const ssize_t gaussianKernel5x5Offset = 0;

    // Sobel gradient kernels
    const float sobelGradientX[9] =
    {
        0.f, 1.f, 0.f,
        1.f, -4.f, 1.f,
        0.f, 1.f, 0.f,

    };
    const float sobelGradientY[9] =
    {
        1.f, 1.f, 1.f,
        1.f, -8.f, 1.f,
        1.f, 1.f, 1.f,
    };
    //  const float sobelGradientX[9] =
    // {
    //     0.f, -1.f, 0.f,
    //     -1.f, 4.f, -1.f,
    //     0.f, -1.f, 0.f,
    // };
    // const float sobelGradientY[9] =
    // {
    //     -1.f, -1.f, -1.f,
    //     -1.f, 8.f, -1.f,
    //     -1.f, -1.f, -1.f,
    // };
    //  const float sobelGradientX[9] =
    // {
    //     0.f, -8.f, 0.f,
    //     -8.f, 4.f, -8.f,
    //     0.f, -8.f, 0.f,
    // };
    // const float sobelGradientY[9] =
    // {
    //     -2.f, 0.f, -2.f,
    //     -2.f, 7.f, -2.f,
    //     -2.f, 0.f, -2.f,
    // };
    // const float sobelGradientX[25] =
    // {
    //     2.f, 2.f, 4.f, 2.f, 2.f,
    //     1.f, 1.f, 2.f, 1.f, 1.f,
    //     0.f, 0.f, 0.f, 0.f, 0.f,
    //     -1.f, -1.f, -2.f, -1.f, -1.f,
    //     -2.f, -2.f, -4.f, -2.f, -2.f,
    // };
    // const float sobelGradientY[25] =
    // {
    //     2.f, 1.f, 0.f, -1.f, -2.f,
    //     2.f, 1.f, 0.f, -1.f, -2.f,
    //     4.f, 2.f, 0.f, -2.f, -4.f,
    //     2.f, 1.f, 0.f, -1.f, -2.f,
    //     2.f, 1.f, 0.f, -1.f, -2.f,
    // };
    // const float sobelGradientX[9] =
    // {
    //     5.f, 5.f, 5.f,
    //     -3.f, 0.f, -3.f,
    //     -3.f, -3.f, -3.f,
    // };
    // const float sobelGradientY[9] =
    // {
    //     5.f, -3.f, -3.f,
    //     5.f, 0.f, -3.f,
    //     5.f, -3.f, -3.f,
    // };
    const float dilate[9] =
    {
        1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 
        1.f, 1.f, 1.f
     
    };
    const float erode[9] =
    {
        1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 
        1.f, 1.f, 1.f
       
    };

    cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientX, sizeof(sobelGradientX), sizeof(gaussianKernel5x5));
    cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientY, sizeof(sobelGradientY), sizeof(gaussianKernel5x5) + sizeof(sobelGradientX));

    const ssize_t sobelGradientXOffset = sizeof(gaussianKernel5x5)/sizeof(float);
    const ssize_t sobelGradientYOffset = sizeof(sobelGradientX)/sizeof(float) + sobelGradientXOffset;
    const ssize_t dilateKernel = sizeof(dilate)/sizeof(float);
    const ssize_t erodeKernel = sizeof(erode)/sizeof(float);

    const ssize_t erodeKernel2 = sizeof(erodeKernel)/sizeof(float);
    // Create CPU/GPU shared images - one for the initial and one for the result
    camera >> frame;
    unsigned char *sourceDataDevice, *blurredDataDevice, *edgesDataDevice, *thresholdDataDevice, *dilateDataDevice, *erodeDataDevice, *erodeDataDevice2, *dilateDataDevic2;
    cv::Mat source  (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sourceDataDevice));
    cv::Mat blurred (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &blurredDataDevice));
    cv::Mat edges   (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice));
    cv::Mat thes   (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &thresholdDataDevice));
    cv::Mat dilates   (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &dilateDataDevice));
    cv::Mat dilates2   (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &dilateDataDevic2));
    cv::Mat erodes   (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &erodeDataDevice));
    cv::Mat erodes2   (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &erodeDataDevice2));
    // Create two temporary images (for holding sobel gradients)
    unsigned char *deviceGradientX, *deviceGradientY;
    cudaMalloc(&deviceGradientX, frame.size().width * frame.size().height);
    cudaMalloc(&deviceGradientY, frame.size().width * frame.size().height);
    
    // Loop while capturing images
    while(1)
    {
        // Capture the image and store a gray conversion to the gpu
        camera >> frame;
        cv::cvtColor(frame, source, CV_BGR2GRAY);
        
        // Record the time it takes to process
        cudaEventRecord(start);
        {
            // convolution kernel launch parameters
            dim3 cblocks (frame.size().width / 32, frame.size().height / 32);
            dim3 cthreads(32, 32);

            // pythagoran kernel launch paramters
            dim3 pblocks (frame.size().width * frame.size().height / 1024);
            dim3 pthreads(1024, 1);

            // convolve<<<cblocks,cthreads>>>(sourceDataDevice, frame.size().width, frame.size().height, 0, 0, gaussianKernel5x5Offset, Blur, Blur, blurredDataDevice);
            // Perform the sobel gradient convolutions (x&y padding is now 2 because there is a border of 2 around a 5x5 gaussian filtered image)
            convolve<<<cblocks,cthreads>>>(sourceDataDevice, frame.size().width, frame.size().height, 2, 2, sobelGradientXOffset, 3, 3, deviceGradientX);
            convolve<<<cblocks,cthreads>>>(sourceDataDevice, frame.size().width, frame.size().height, 2, 2, sobelGradientYOffset, 3, 3, deviceGradientY);
            pythagoras<<<pblocks,pthreads>>>(deviceGradientX, deviceGradientY, edgesDataDevice);

            // Perform the gaussian blur (first kernel in store @ 0)
            convolve<<<cblocks,cthreads>>>(edgesDataDevice, frame.size().width, frame.size().height, 0, 0, gaussianKernel5x5Offset, Blur, Blur, blurredDataDevice);

            theshould<<<pblocks,pthreads>>>(blurredDataDevice,thresholdDataDevice,60);
            Dilate<<<cblocks,cthreads>>>(thresholdDataDevice, frame.size().width, frame.size().height, 2, 2, dilateKernel, 4, 4, dilateDataDevice);
            Erode<<<cblocks,cthreads>>>(dilateDataDevice, frame.size().width, frame.size().height, 2, 2, erodeKernel, 7, 7, erodeDataDevice);
            // // 
            // Erode<<<cblocks,cthreads>>>(dilateDataDevice, frame.size().width, frame.size().height, 2, 2, erodeKernl, 4, 4, erodeDataDevice);
            
            // Dilate<<<cblocks,cthreads>>>(erodeDataDevice, frame.size().width, frame.size().height, 2, 2, dilateKernel, 4, 4, dilateDataDevic2);
            // Erode<<<cblocks,cthreads>>>(dilateDataDevic2, frame.size().width, frame.size().height, 2, 2, erodeKernel2, 4, 4, erodeDataDevice2);
            
            cudaThreadSynchronize();
        }
        cudaEventRecord(stop);
        
        // Display the elapsed time
        float ms = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;

        // Show the results
        cv::imshow("Source", frame);
        cv::imshow("Greyscale", source);
        cv::imshow("Sobel", edges);
        cv::imshow("Blurred", blurred); 
        cv::imshow("Theshould", thes);
        cv::imshow("Erode", erodes);
        cv::imshow("Erode2", erodes2);
        cv::imshow("Dilate", dilates);
        cv::imshow("Dilate2", dilates2);
        
        // Spin
        if(cv::waitKey(1) == 27) break;
    }
    
    // Exit
    cudaFreeHost(source.data);
    cudaFreeHost(blurred.data);
    cudaFreeHost(edges.data);
    cudaFree(deviceGradientX);
    cudaFree(deviceGradientY);

    return 0;
}
