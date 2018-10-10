__kernel void reduce_add_6(__global float* A, __local float* scratch) {
	int id = get_global_id(0); //global id
	int lid = get_local_id(0); //local id
	int N = get_local_size(0); // number of elements
	const uint group_id = get_group_id(0);//get global work item id

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) { // strides
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i]; //add neighbouring element to current value

		barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish adding neighbouring elements
	}

	//copy the sum of work group to output array in position of work item id
	if (lid == 0) A[group_id] = scratch[0];
}

__kernel void reduce_max(__global float* A, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	const uint group_id = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) { //strides
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid] < scratch[lid + i]) //check neighbour is bigger
				scratch[lid] = scratch[lid + i]; //set current value as neighbours value as it is bigger
		}
		barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish
	}

	//copy the cache to output array in position of work item id
	if (!lid)  A[group_id] = scratch[0];

}

__kernel void reduce_min(__global float* A, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	const uint group_id = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) { //strides
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid] > scratch[lid + i]) //check neighbour is smaller
				scratch[lid] = scratch[lid + i]; //set current value as neighbours value as it is bigger
		}
		barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish
	}

	//copy the cache to output array in position of work item id
	if (!lid)  A[group_id] = scratch[0];

}

int bin_index(const float val, const float min, const int nr_bins, const float bin_width)
{
	int index = ((val - min) / bin_width); // returns index of given value in bin H vector
	if (index >= nr_bins || index < 0) // check bin index is within H vector
	{
		index = nr_bins; // return index outside of H so it isn't counted
	}
	return index; //return calculated index
}
// hist kernal
__kernel void hist_atomic(__global const float* A, __global const int* nr_bins, __global const float* bin_width, __global const float* min, __global int* H) {
	int id = get_global_id(0);

	// atomically increment Historgram vector from bin id returned from bin_index function
	atomic_inc(&H[bin_index(A[id], *min, *nr_bins, *bin_width)]);
}