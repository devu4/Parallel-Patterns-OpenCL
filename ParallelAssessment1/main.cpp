#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <future>


#include <CL/cl.hpp>
#include "Utils.h"
#include "Functions.h" // file with all host code functions

using namespace std;

typedef float mytype;
vector<mytype> A; // input data vector
vector<vector<mytype>> months(12); // input data split to months

void print_help() {
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

//function to read in data from text file and set to vectors
void populate_data() {
	   ifstream file("../temp_lincolnshire.txt");
	   if (file.fail()) { //check file exists
		   cout << endl << "File does not exist!" << endl;
		   system("pause");
		   exit(EXIT_FAILURE);
	   }

	   string   line;

	   while (getline(file, line))
	   {
		   istringstream linestream(line);
		   float val; int monthID; string tmp;
		   linestream >> tmp >> tmp >> monthID >> tmp >> tmp >> val;
		   A.push_back(val);
		   months[monthID - 1].push_back(val);
	   }
}

int main(int argc, char **argv)
{
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	cl::Context context; cl::CommandQueue queue; cl::Program program;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//Part 1.1 - Load the data on a different thread so menu can be shown when data is loading...
	std::future<void> result = async(launch::async, populate_data);
	std::cout << "        *----------------------* David's Parallel Temp Stats *----------------------*" << endl;

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		context = GetContext(platform_id, device_id);

		//display the selected device

		//create a queue to which we will push commands for the device
		queue = cl::CommandQueue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		program = cl::Program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	std::cout << "        *-----------------* Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << " *------------------*" << std::endl << endl;
	
	//show main menu and input from user
	int menuInput = 1;
	bool hasMenuInput = false;
	while (!hasMenuInput)
	{
		cout << "----------------------------------------" << endl;
		cout << "Please select a menu item!" << endl;
		cout << "----------------------------------------" << endl;
		cout << "1. View Full Data Summaries" << endl;
		cout << "2. View Monthly Summaries" << endl;
		cout << "3. View Full Data Histogram" << endl;
		cin >> menuInput;

		if ((menuInput <= 3) && (menuInput > 0))
			hasMenuInput = true;
		else
			cout << "Invalid value given, please choose a number from 1-3!" << endl;
	}
	//show full data results
	if (menuInput == 1)
	{
		result.get(); // make sure different thread data load is done

		std::cout << "-----------------------------------" << std::endl;
		std::cout << "Full Data Summaries" << std::endl;
		std::cout << "-----------------------------------" << std::endl;
		std::cout << "Min Value = " << parallelMin(context, program, queue, A) << std::endl;
		std::cout << "Mean Value = " << parallelMean(context, program, queue, A) << std::endl;
		std::cout << "Max Value = " << parallelMax(context, program, queue, A) << std::endl;
		std::cout << "-----------------------------------" << std::endl;
	}
	else if (menuInput == 2)
	{
		//ask user for which month to do 
		int monthChosen = 0;
		bool hasMonth = false;
		while (!hasMonth)
		{
			std::cout << "--------------------------------------------------------------" << std::endl;
			std::cout << "Monthly Data Summaries" << std::endl;
			std::cout << "--------------------------------------------------------------" << std::endl;
			cout << "Which month would you like to see summaries of? (1-12)" << endl;
			std::cout << "--------------------------------------------------------------" << std::endl;
			cin >> monthChosen;

			if ((monthChosen <= 12) && (monthChosen > 0))
			{
				hasMonth = true;
			}
			else
			{
				cout << "Invalid value given, please choose a number from 1-12!" << endl << endl;
			}
		
		}

		result.get();// make sure different thread data load is done

		//show chosen months min/avg/max
		std::cout << "-----------------------------------" << std::endl;
		std::cout << "Month " << monthChosen << " Data Summaries" << std::endl;
		std::cout << "-----------------------------------" << std::endl;
		std::cout << "Min Value = " << parallelMin(context, program, queue, months[monthChosen - 1]) << std::endl;
		std::cout << "Mean Value = " << parallelMean(context, program, queue, months[monthChosen - 1]) << std::endl;
		std::cout << "Max Value = " << parallelMax(context, program, queue, months[monthChosen - 1]) << std::endl;
		std::cout << "-----------------------------------" << std::endl;


	}
	//show histogram menu
	else
	{
		//Ask user for number of bins wanted
		int binsChosen = 0;
		bool hasBins = false;
		while (!hasBins)
		{
			std::cout << "--------------------------------------------------------------" << std::endl;
			std::cout << "Full Data Histogram" << std::endl;
			std::cout << "--------------------------------------------------------------" << std::endl;
			cout << "How many bins would you like for this histogram?" << endl;
			std::cout << "--------------------------------------------------------------" << std::endl;
			cin >> binsChosen;

			if ((binsChosen < 1) || cin.fail())
			{
				cout << "Invalid value given, please choose a integer greater than 0!" << endl << endl;
			}
			else
			{
				hasBins = true;
			}

		}
		result.get();// make sure different thread data load is done
		//create histogram using nr of bins chosen by user (this is in functions.h)
		parallelHistogram(context, program, queue, A, binsChosen);
	}

	system("pause");
	return 0;
}