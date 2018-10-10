#pragma once

#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>

#include <CL/cl.hpp>
#include "Utils.h"

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

typedef float mytype;

//function to find mean of data using opencl kernels
double parallelMean(cl::Context& context, cl::Program & program, cl::CommandQueue& queue, vector<mytype> A)
{
	//create kernel for reduction
	cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_6");
	size_t start_nr_elements = A.size(); // save the number of elements for later

	//get device and get the max number of work group size recommended
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t local_size = kernel_1.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

	//pad input vector to be multiple of workgroup size as it is more efficient
	size_t padding_size = A.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<float> A_ext(local_size - padding_size, 0);
		//append that extra vector to our input
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}

	size_t input_elements = A.size();//number of input elements after padding
	size_t input_size = A.size() * sizeof(mytype);//size in bytes of input after padding

	//host - output
	vector<mytype> B(input_elements);
	size_t output_size = B.size() * sizeof(mytype);//size in bytes of output

	//device - buffers
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, input_size, &A);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size, NULL);

	//Copy vector A to device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);

	//Setup kernal arguments
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, cl::Local(local_size * sizeof(mytype)));//local memory size

	//call all kernels in a sequence
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(A.size()), cl::NDRange(local_size));

	//keep calling reduction kernel until input is smaller than workgroup size
	// this means that the first element will be the sum of the vector
	while (input_elements / local_size > local_size) {
		input_elements = input_elements / local_size;

		size_t padding_size2 = input_elements % local_size;

		//if the input vector is not a multiple of the local_size
		//then add to input size and make the added element 0
		if (padding_size2) {
			//read the buffer back into A
			queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &A[0]); // read out buffer_A and then pad
			size_t neededElements = local_size - padding_size2;

			for (auto i = A.begin() + input_elements; i < (A.begin() + input_elements + neededElements); i++) {
				*i = 0; // use zero as neutral element as it does not affect addition
			}

			input_elements = input_elements + neededElements; // add the pad elements

			//write bac to buffer
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]); // write vector A into device
		}

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
	}

	//do final reduction, this time first element will be total sum
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &A[0]);
	size_t start = input_elements / local_size;
	for (auto i = A.begin() + start; i < (A.begin() + local_size); i++) {
		*i = 0; // use zero as neutral element as it does not affect addition
	}

	input_elements = local_size;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

	//read the buffer A from device to host vector B
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &B[0]);
	
	//return mean using sum total divided by number of elements before padding
	return B[0]/start_nr_elements;
}

//function to find max of vector using reduction in parallel
float parallelMax(cl::Context& context, cl::Program & program, cl::CommandQueue& queue, vector<mytype> A)
{
	//create kernel for max reduction
	cl::Kernel kernel_1 = cl::Kernel(program, "reduce_max");

	//get device and get the max number of work group size recommended
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t local_size = kernel_1.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

	//pad input vector to be multiple of workgroup size as it is more efficient
	size_t padding_size = A.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (-INFINITY for max) so that the top value will be max
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<float> A_ext(local_size - padding_size, -INFINITY);
		//append that extra vector to our input
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}

	size_t input_elements = A.size();//number of input elements after padding
	size_t input_size = A.size() * sizeof(mytype);//size in bytes after padding

	//host - output
	vector<mytype> B(input_elements);
	size_t output_size = B.size() * sizeof(mytype);//size in bytes of output
    
    //Copy vector A to device memory and initialize output
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, input_size, &A);

	//Copy vector A to device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);

	//Setup and execute all kernels (i.e. device code)
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, cl::Local(local_size * sizeof(mytype)));//local memory size

	//call all kernels in a sequence
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(A.size()), cl::NDRange(local_size));

	//keep calling reduction kernel until input is smaller than workgroup size
	//this means that the first element will be the max of the vector
	while (input_elements / local_size > local_size) {
		input_elements = input_elements / local_size;

		size_t padding_size2 = input_elements % local_size;

		//if the input vector is not a multiple of the local_size
		//then add to input size and make the added element 0
		if (padding_size2) {
			//read the buffer back into A
			queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &A[0]);
			size_t neededElements = local_size - padding_size2;

			//set netural values for current input size
			for (auto i = A.begin() + input_elements; i != (A.begin() + input_elements + neededElements); i++) {
				*i = -INFINITY; //neutral value for max is -INFINITY
			}

			input_elements = input_elements + neededElements;

			//write A back to buffer
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		}
		//execute kernal on device
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
	}

	//do final reduction, this time first element will be max value
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &A[0]);
	size_t start = input_elements / local_size;
	for (auto i = A.begin() + start; i != (A.begin() + local_size); i++) {
		*i = -INFINITY; //neutral value for max is -INFINITY
	}
	input_elements = local_size;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

	//read the buffer A from device to host vector B
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &B[0]);

	return B[0];
}

//function to find min of vector using reduction in parallel
float parallelMin(cl::Context& context, cl::Program & program, cl::CommandQueue& queue, vector<mytype> A)
{
	//create kernel for min reduction
	cl::Kernel kernel_1 = cl::Kernel(program, "reduce_min");

	//get device and get the max number of work group size recommended
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t local_size = kernel_1.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

	//pad input vector to be multiple of workgroup size as it is more efficient
	size_t padding_size = A.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (INFINITY for min) so that the top value will be min
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<float> A_ext(local_size - padding_size, INFINITY);
		//append that extra vector to our input
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}

	size_t input_elements = A.size();//number of input elements after padding
	size_t input_size = A.size() * sizeof(mytype);//size in bytes after padding

	//host - output
	vector<mytype> B(input_elements);
	size_t output_size = B.size() * sizeof(mytype);//size in bytes of output
	
	//Copy vector A to device memory and initialize output
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, input_size, &A);

	//Copy vector A to device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);

	//Setup and execute all kernels (i.e. device code)
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, cl::Local(local_size * sizeof(mytype)));//local memory size

	//call all kernels in a sequence
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(A.size()), cl::NDRange(local_size));

	//keep calling reduction kernel until input is smaller than workgroup size
	//this means that the first element will be the max of the vector
	while (input_elements / local_size > local_size) {
		input_elements = input_elements / local_size;

		size_t padding_size2 = input_elements % local_size;

		//if the input vector is not a multiple of the local_size
		//then add to input size and make the added element 0
		if (padding_size2) {
			//read the buffer back into A
			queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &A[0]);
			size_t neededElements = local_size - padding_size2;

			//set netural values for current input size
			for (auto i = A.begin() + input_elements; i != (A.begin() + input_elements + neededElements); i++) {
				*i = INFINITY; //neutral value for min is INFINITY
			}

			input_elements = input_elements + neededElements;

			//write A back to buffer
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		}
		//execute kernal on device
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
	}

	//do final reduction, this time first element will be max value
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &A[0]);
	size_t start = input_elements / local_size;
	for (auto i = A.begin() + start; i != (A.begin() + local_size); i++) {
		*i = INFINITY; //neutral value for min is INFINITY
	}
	input_elements = local_size;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

	//read the buffer A from device to host vector B
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &B[0]);

	return B[0];
}

//function to create histogram using number of bins in parallel
void parallelHistogram(cl::Context& context, cl::Program & program, cl::CommandQueue& queue, vector<mytype> A, int & nr_bins)
{

	float min = floor(parallelMin(context, program, queue, A)); //find min value and round down
	float max = (ceil(parallelMax(context, program, queue, A))) + 1; // find max value and round up then add 1 so all value are counted

	float range = max - min; // find range of data set
	float bin_width = range / nr_bins; // find width of each bin by dividing range by number of bins wanted

	//create kernel for parallel histogram
	cl::Kernel kernel_1 = cl::Kernel(program, "hist_atomic");

	//get device and get the max work group size recommended
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	size_t local_size = kernel_1.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);


	size_t padding_size = A.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (max+1 for histogram as it will be outside the bins) so that the histogram will not be affected
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<float> A_ext(local_size - padding_size, max + 1);
		//append that extra vector to our input
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}

	size_t input_elements = A.size();//number of input elements
	size_t input_size = A.size() * sizeof(mytype);//size in bytes

	vector<int> H(nr_bins); // create out put host vector for histogram

	//device - buffers
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer buffer_bins(context, CL_MEM_READ_WRITE, sizeof(int));
	cl::Buffer buffer_width(context, CL_MEM_READ_WRITE, sizeof(float));
	cl::Buffer buffer_min(context, CL_MEM_READ_WRITE, sizeof(float));
	cl::Buffer buffer_H(context, CL_MEM_READ_WRITE, sizeof(int)*(nr_bins));

	//Write data to buffer and initialize output buffer
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
	queue.enqueueWriteBuffer(buffer_bins, CL_TRUE, 0, sizeof(int), &nr_bins);
	queue.enqueueWriteBuffer(buffer_width, CL_TRUE, 0, sizeof(float), &bin_width);
	queue.enqueueWriteBuffer(buffer_min, CL_TRUE, 0, sizeof(float), &min);
	queue.enqueueFillBuffer(buffer_H, 0, 0, sizeof(int)*(nr_bins));//zero B buffer on device memory

	//Setup and execute all kernels (i.e. device code)
	kernel_1.setArg(0, buffer_A);
	kernel_1.setArg(1, buffer_bins);
	kernel_1.setArg(2, buffer_width);
	kernel_1.setArg(3, buffer_min);
	kernel_1.setArg(4, buffer_H);

	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(A.size()), cl::NDRange(local_size));

	//read buffer_H into host code vector H
	queue.enqueueReadBuffer(buffer_H, CL_TRUE, 0, sizeof(int)*(nr_bins), &H[0]);

	//display output in console using vector H
	std::cout << "--------------------------------------------------------------" << std::endl;
	std::cout << "Full Data Histogram" << std::endl;
	std::cout << "--------------------------------------------------------------" << std::endl;
	cout << "Number of Bins: " << nr_bins << endl;
	std::cout << "--------------------------------------------------------------" << std::endl;
	for (int i = 0; i < H.size(); i++) {
		cout << "Bin " << i+1 << " [" << ((i*bin_width) + min) << " to " << (((i+1)*bin_width) + min) << "]  " << H[i] << endl;
	}
	std::cout << "--------------------------------------------------------------" << std::endl;

}

/*void normalHist(cl::Context& context, cl::Program & program, cl::CommandQueue& queue, vector<mytype> A, int & nr_bins)
{
	float min = floor(parallelMin(context, program, queue, A)); //find min value and round
	float max = (ceil(parallelMax(context, program, queue, A))) + 1; // find max value and round up then add 1 so value is counted

	float range = max - min; // find range of data set
	float bin_width = range / nr_bins; // find width of each bin by dividing range by number of bins wanted
	vector<int> H(nr_bins);

	for (int i = 0; i < A.size(); i++) {
		int index = ((A[i] - min) / bin_width);
		if (index >= nr_bins || index < 0)
		{
			index = nr_bins;
		}
		H[index]++;
	}

	for (int c = 0; c < H.size(); c++) {
		cout << "Bin " << c + 1 << " [" << ((c*bin_width) + min) << " to " << (((c + 1)*bin_width) + min) << "]  " << H[c] << endl;
	}
}*/

//check function for mean in seqential programming
double normalMean(cl::Context& context, cl::Program & program, cl::CommandQueue& queue, vector<mytype> A)
{

	double sum = 0;
	for (int i = 0; i < A.size(); i++) {
		sum += A[i];
	}

	return sum/A.size() ;
}