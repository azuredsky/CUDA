//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <time.h>
#include <cuda.h>

const int MIN_SIZE=1280; 
const int MAX_SIZE=10000;
const int STEP_SIZE=256;
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float* a;
float* b;
float* c;
float* a1;
float* b1;
float* c1;
float* a2;
float* b2;
float* c2;
float* c3;
float* c4;
int n;

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
__global__ void kernelFunc(float* ad, float* bd, float* cd, int n) {
    __shared__ float as[32][32];
    __shared__ float bs[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
  int x = (blockIdx.x * blockDim.x) + tx;
	int y = (blockIdx.y * blockDim.y) + ty;

    if ((x<n)&(y<n) )
	{

    float v = 0.0f;
    
    int yn = y * n;
    int s = n / 32;
    for(int m=0; m<s; m++) {
        int m32 = m * 32;
        as[ty][tx] = ad[yn + (m32 + tx)];
        bs[ty][tx] = bd[(m32 + ty) * n + x];
        
        __syncthreads();
        
        for(int i=0; i<32; i++) {
            v += as[ty][i] * bs[i][tx];
        }
        
        __syncthreads();
    }
    
    cd[yn + x] = v;   
	}
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matrixMultiply(float* ad,float* bd,float* cd,float *H2DBandWidthInMBs,float *D2HBandWidthInMbs) {
	

    cudaMalloc((void**)&ad, n * n * sizeof(float));
    cudaMalloc((void**)&bd, n * n * sizeof(float));
    cudaMalloc((void**)&cd, n * n * sizeof(float));

    cudaEvent_t start, stop;

	float time;

	cudaEventCreate(&start);

	cudaEventCreate(&stop);

	cudaEventRecord( start, 0 );


    cudaMemcpy(ad, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, n * n * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord( stop, 0 );

	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );

	cudaEventDestroy( start );

	cudaEventDestroy( stop );

	*H2DBandWidthInMBs = (1e3f * 2*n*n* sizeof(float)) /
                     (time* (float)(1 << 20));
	//printf("HostToDevice bandwidthInMBs %f\n",H2DBandWidthInMBs );
	//printf("cudaMemcpyHostToDevice(ms) %f\n", 1000*(tc1 - tc0) / (float) CLOCKS_PER_SEC);
    dim3 block(32, 32);           
    dim3 grid((n+31)/32, (n+31)/32);
    
    kernelFunc<<<grid, block>>>(ad, bd, cd, n);

	float time1;

	cudaEventCreate(&start);

	cudaEventCreate(&stop);

	cudaEventRecord( start, 0 );
	
    cudaMemcpy(c, cd, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord( stop, 0 );

	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time1, start, stop );

	cudaEventDestroy( start );

	cudaEventDestroy( stop );

	*D2HBandWidthInMbs = (1e3f * n*n* sizeof(float)) /
                     (time1* (float)(1 << 20));
	//printf("DeviceToHostbandwidthInMBs %f\n",D2HBandWidthInMbs );
	//printf("cudaMemcpyDeviceToHost(ms) %f\n", 1000*(tg1 - tg0) / (float) CLOCKS_PER_SEC);
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void fill(float* data, int size) {
    for (int i=0; i<size; ++i)
        data[i] = rand() / (float) RAND_MAX;
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void save(char* name, int n, float H2DBandWidthInMBs,float D2HBandWidthInMbs) {
	char fname[128];
	FILE* f;
/*
	sprintf(fname, "results/result-mm-%s_%d.txt", name, n);
	f = fopen(fname, "w");
	for (int i=0; i<n * n; ++i)
		fprintf(f, "%f\n", c[i]);
	fclose(f);
*/	
	sprintf(fname, "runtime-mm-%s.txt", name);
	f = fopen(fname, "a");

	fprintf(f, "size of matrix%d : H2DBandWidthInMBs     %f D2HBandWidthInMbs    %f \n",  
		n, H2DBandWidthInMBs,D2HBandWidthInMbs);
	fclose(f);
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
int main(int argc, char** argv) {
	remove("runtime-mm-gpu-optimized.txt");
	int devicenum,dev;
	struct cudaDeviceProp p;
	cudaGetDeviceCount(&devicenum);
	for (dev=0;dev<devicenum;dev++){
		cudaGetDeviceProperties(&p, dev);
	printf("\nDevice %d: \%s\"\n",dev,p.name);
	}
    cudaGetDeviceProperties(&p, 0);
    printf("maxGridSize: %d %d %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    printf("maxThreadsDim: %d %d %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("maxThreadsPerBlock: %d\n", p.maxThreadsPerBlock);
    printf("warpSize: %d\n", p.warpSize);
    printf("totalGlobalMem(MB): %d\n", p.totalGlobalMem/1024/1024);
    printf("sharedMemPerBlock: %d\n", p.sharedMemPerBlock);
	float H2DBandWidthInMBs,D2HBandWidthInMbs;
	for(n=MIN_SIZE; n<=MAX_SIZE; n+=STEP_SIZE) {

		a = (float*)malloc(n * n * sizeof(float));
		b = (float*)malloc(n * n * sizeof(float));
		c = (float*)malloc(n * n * sizeof(float));
		
		if(n*n*sizeof(float)*8<=p.totalGlobalMem){
		srand(0);
		fill(a, n * n);
		fill(b, n * n);
		printf("gpu N*N=%d\n",n);
		//clock_t t0 = clock(); 
		matrixMultiply(a,b,c,&H2DBandWidthInMBs,&D2HBandWidthInMbs);
		//clock_t t1 = clock(); 
		//printf("gpu time(ms)%f\n",(1000*(t1-t0)/(float) CLOCKS_PER_SEC));
		save("gpu-optimized", n, H2DBandWidthInMBs,D2HBandWidthInMbs);
		}

		if (n*n*sizeof(float)*8>p.totalGlobalMem){
		a1 = (float*)malloc(n/2 * n * sizeof(float));a2 = (float*)malloc(n/2 * n * sizeof(float));
		b1 = (float*)malloc(n* n/2 * sizeof(float));b2 = (float*)malloc(n * n/2 * sizeof(float));
		c1 = (float*)malloc(n/2 * n/2 * sizeof(float));c2 = (float*)malloc(n/2 * n/2 * sizeof(float));
		c3 = (float*)malloc(n/2 * n/2 * sizeof(float));c4 = (float*)malloc(n/2 * n/2 * sizeof(float));
		if (n/2*n*sizeof(float)*8> (p.totalGlobalMem-50*1024*1024))
			break;
		srand(0);
		fill(a1, n/2 * n);fill(a2, n/2 * n);
		fill(b1, n * n/2);fill(b2, n * n/2);
		printf("gpu N*N %d\n",n);

		//clock_t Mt0 = clock(); 
			matrixMultiply(a1,b1,c1,&H2DBandWidthInMBs,&D2HBandWidthInMbs);
			save("gpu-optimized", n, H2DBandWidthInMBs,D2HBandWidthInMbs);
			matrixMultiply(a1,b2,c2,&H2DBandWidthInMBs,&D2HBandWidthInMbs);
			save("gpu-optimized", n, H2DBandWidthInMBs,D2HBandWidthInMbs);
			matrixMultiply(a2,b1,c3,&H2DBandWidthInMBs,&D2HBandWidthInMbs);
			save("gpu-optimized", n, H2DBandWidthInMBs,D2HBandWidthInMbs);
			matrixMultiply(a2,b2,c4,&H2DBandWidthInMBs,&D2HBandWidthInMbs);
			save("gpu-optimized", n, H2DBandWidthInMBs,D2HBandWidthInMbs);
		//clock_t Mt1 = clock(); 
		//printf("gpu Matrix time(ms)%f\n",(1000*(Mt1-Mt0)/(float) CLOCKS_PER_SEC));		

		}
		
		free(a);free(a2);free(a1);
		free(b);free(b2);free(b1);
		free(c);free(c2);free(c1);
		free(c4);free(c3);
		
	}
    
	cudaThreadExit();
		return 0;
}
