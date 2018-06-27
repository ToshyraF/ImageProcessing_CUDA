#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

__constant__ float convolutionKernelStore[1024];

unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr)
{   
    unsigned char *ptr = NULL;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(devicePtr, ptr, 0);
    return ptr;
}
__global__ void bg(unsigned char *source1,unsigned char *source2, unsigned char *destination)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float af = float(source1[idx]);
    float bf = float(source2[idx]);

    destination[idx] = af-bf;
}

__global__ void theshould(unsigned char *source, unsigned char *destination, int min)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float af = float(source[idx]);
    if(af > min ){
        destination[idx] = 255;
    }
    else{
        destination[idx] = 0;
    }
    // c[idx] = (unsigned char) abs(af*af - bf*bf);
}


int main (int argc, char** argv)
{
    // Open a webcamera
    cv::VideoCapture camera("768x576.avi");
    // cv::VideoCapture camera(0);
    cv::Mat          frame;
    if(!camera.isOpened()) 
        return -1;
    camera >> frame;

    unsigned char *frameStartDataDevice;

    cv::Mat frameStart (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &frameStartDataDevice));

    cv::cvtColor(frame, frameStart, CV_BGR2GRAY);

    unsigned char *sourceDataDevice, *thresDataDevice, *bgDataDevice;
    cv::Mat source  (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sourceDataDevice));
    cv::Mat bg_subtrack (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &bgDataDevice));
    cv::Mat thres (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &thresDataDevice));



    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

     while(1)
    {
        
        camera >> frame;
        cv::cvtColor(frame, source, CV_BGR2GRAY);
        cudaEventRecord(start);
        {   
            dim3 pblocks (frame.size().width * frame.size().height / 1024);
            dim3 pthreads(1024, 1);

            bg<<<pblocks,pthreads>>>(frameStartDataDevice, sourceDataDevice,bgDataDevice);
            theshould<<<pblocks,pthreads>>>(bgDataDevice,thresDataDevice,30);
            
            cudaThreadSynchronize();
        }
        cudaEventRecord(stop);

        float ms = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;

        cv::imshow("FrameStart", frameStart);
        cv::imshow("Frame", frame);
        cv::imshow("Source", source);
        cv::imshow("Thres", thres);
        cv::imshow("Bg", bg_subtrack);

        std::cout << camera.get(CV_CAP_PROP_POS_FRAMES) << std::endl;
        if((camera.get(CV_CAP_PROP_POS_FRAMES)+1) < camera.get(CV_CAP_PROP_FRAME_COUNT)){
        
        }
        else{
            camera.set(CV_CAP_PROP_POS_FRAMES, 0);
        }
        if(cv::waitKey(1) == 27) break;
    }

   cudaFreeHost(source.data);
   cudaFreeHost(bg_subtrack.data);
   cudaFreeHost(thres.data);
   return 0;
}
