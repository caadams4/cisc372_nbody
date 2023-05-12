#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void accelEffectsInitialize(vector3 *values, vector3 **accels) {
        int i = threadIdx.x + (blockIdx.x*blockDim.x);
                printf("%d\n",i);

        if (i<NUMENTITIES) accels[i] = &values[NUMENTITIES*i];
}

__global__ void computePairwiseAccels(double *d_hMass,vector3* accels_sum,vector3 **accels,vector3 *d_hPos) {
        printf("%d\t%d\n",d_hPos[0]);
        //for (i=0;i<NUMENTITIES;i++){
        int i = threadIdx.x + (blockIdx.x*blockDim.x);
                //for (j=0;j<NUMENTITIES;j++){
                        int j = threadIdx.y;
                        if (i==j) {
                                FILL_VECTOR(accels[i][j],0,0,0);
                        }else{
                                vector3 distance;
                                for (int k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
                                        double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                                double magnitude=sqrt(magnitude_sq);
                                double accelmag=-1*GRAV_CONSTANT*d_hMass[j]/magnitude_sq;
                                FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
                        }
}                                                                               //}
                                                                                                        //}


__global__ void sumAccels(vector3 *accels_sum, vector3 **accels, vector3 *d_hPos, vector3 *d_hVel) {
                //sum up the rows of our matrix to get effect on each entity, then update velocity and position.
                int i = threadIdx.x + (blockIdx.x*blockDim.x);
                printf("%d\n",i);
                int k;
                //for (i;i<numentities;i++){
                        //vector3 accel_sum={0,0,0};
                        for (int j=0;j<NUMENTITIES;j++){
                                for (k=0;k<3;k++);
                                        accels_sum[i][k] += accels[i][j][k];
                                        printf("d\n",accels_sum[i][k]);
                        }
                        //compute the new velocity based on the acceleration and time interval
                        //compute the new position based on the velocity and time interval
                        for (k=0;k<3;k++){
                                d_hVel[i][k]+=1;//accels_sum[k]*INTERVAL;
                                d_hPos[i][k]=d_hVel[i][k]*INTERVAL;
                        }
                //}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
        vector3 *d_hVel,  *values, *accels_sum, **accels;
        double *d_hMass;

        int szBlocks = 256; // 3d matrix
        int blocks = 1;

        cudaMalloc((void**)&values,sizeof(vector3)*NUMENTITIES*NUMENTITIES);
        cudaMalloc((void**)&accels_sum,sizeof(vector3)*NUMENTITIES);
        cudaMalloc((void***)&accels,sizeof(vector3*)*NUMENTITIES);
        cudaMalloc((void**)&d_hPos,sizeof(vector3)*NUMENTITIES);
        cudaMalloc((void**)&d_hMass,sizeof(double)*NUMENTITIES);
        cudaMalloc((void**)&d_hVel,sizeof(vector3)*NUMENTITIES);

        cudaMemcpy(d_hVel,hVel,sizeof(NUMENTITIES)*sizeof(vector3),cudaMemcpyHostToDevice);
        cudaMemcpy(d_hPos,hPos,sizeof(NUMENTITIES)*sizeof(vector3),cudaMemcpyHostToDevice);
        cudaMemcpy(d_hMass,mass,sizeof(double)*sizeof(vector3),cudaMemcpyHostToDevice);

        /*=============================================================================*/
        // Step 1, Construct a NUMELEMENTS x NUMELEMENTS matrix to hold the pairwise acceleration effects between any 2 objects
        /*=============================================================================*/

        accelEffectsInitialize<<<blocks,szBlocks>>>(values,accels);

        /*=============================================================================*/
        // Step 2, Compute the acceleration matrix
        /*=============================================================================*/

        computePairwiseAccels<<<blocks,szBlocks>>>(d_hMass,accels_sum,accels,d_hPos);

        /*=============================================================================*/
        // Step 3, Sum up the columns to get a single acceleration effect on that object
        /*=============================================================================*/

        sumAccels<<<blocks,szBlocks>>>(accels_sum,accels,d_hPos,d_hVel);

        // Wrap up program, send cuda data to host
        cudaMemcpy(hVel,d_hVel,sizeof(NUMENTITIES)*sizeof(vector3),cudaMemcpyDeviceToHost);
        cudaMemcpy(hPos,d_hPos,sizeof(NUMENTITIES)*sizeof(vector3),cudaMemcpyDeviceToHost);
        cudaMemcpy(mass,d_hMass,sizeof(NUMENTITIES)*sizeof(vector3),cudaMemcpyDeviceToHost);

        // free cuda memory
        cudaFree(d_hVel);
        cudaFree(d_hPos);
        cudaFree(d_hMass);
        cudaFree(accels);
        cudaFree(accels_sum);
        cudaFree(values);

}
