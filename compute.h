void compute();
__global__ void accelEffectsInitialize(vector3 *values, vector3 **accels);

__global__ void computePairwiseAccels(double d_mass,vector3* accels_sum,vector3 **accels,vector3 *d_hPos);

__global__ void sumAccels(vector3 *d_hPos, vector3 *d_hVel);
