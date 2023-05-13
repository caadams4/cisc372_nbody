#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void accelEffectsInitialize(vector3 *values, vector3 **accels) {
        int i = threadIdx.x + (blockIdx.x*blockDim.x);
        if (i<NUMENTITIES) accels[i] = &values[NUMENTITIES*i];
}

__global__ void computePairwiseAccels(double *d_hMass,vector3* accels_sum,vector3 **accels,vector3 *d_hPos) {
        int i = threadIdx.x + (blockIdx.x*blockDim.x);

        if (i<NUMENTITIES) {
                for (int j=0;j<NUMENTITIES;j++){
                        //int j = threadIdx.y;
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
                }
        }                                                                               //}

}

__global__ void sumAccels(vector3 *accels_sum, vector3 **accels,vector3 *d_hPos, vector3*d_hVel) {
        int i = threadIdx.x + (blockIdx.x*blockDim.x);
        if (i<NUMENTITIES) {
             for (int j=0;j<NUMENTITIES;j++){
                for (int k=0;k<3;k++)
                        accels_sum[i][j]+=accels[i][j][k];
             }
                for (int k=0;k<3;k++){

                        d_hVel[i][k]+=accels_sum[i][k]*INTERVAL;
                        d_hPos[i][k]=d_hVel[i][k]*INTERVAL;

                }

        }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
        vector3 *d_hVel,  *values, *accels_sum, **accels;
        double *d_hMass;
        int i, j ,k;

        int szBlocks = 256; // 3d matrix
        int blocks =  1;

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

        cudaFree(d_hVel);
        cudaFree(d_hPos);
        cudaFree(d_hMass);
        cudaFree(accels);
        cudaFree(accels_sum);
        cudaFree(values);

}
