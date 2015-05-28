/* Demonstrate platform behavior with events for failed API calls */

#include <string.h>
#include <stdio.h>
#include <CL/cl.h>

typedef int bool;
#define false 0
#define true (!false)

#define CHECK_ERROR(what) do { \
	if (error != CL_SUCCESS) { \
		fprintf(stderr, "%s:%u: %s : error %d\n", \
			__func__, __LINE__, what, error);\
		goto out; \
	} \
} while (0);

cl_uint np; // number of platforms
cl_platform_id *platform; // list of platforms ids

// context property: field 1 (the platform) will be set at runtime
cl_context_properties ctx_prop[] = { CL_CONTEXT_PLATFORM, 0, 0, 0 };
cl_context ctx; // context

// generic string retrieval buffer. quick'n'dirty, hence fixed-size
#define BUFSZ 1024
char strbuf[BUFSZ];

cl_int test_device(cl_platform_id p, cl_device_id d)
{
	static const cl_event invalid_evt = (cl_event)(-1);
	cl_event event = invalid_evt;

	unsigned int ocl_major, ocl_minor;
	cl_command_queue q = NULL;


	cl_int error = clGetDeviceInfo(d, CL_DEVICE_NAME, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting device name");
	printf("Device: %s\n", strbuf);

	error = clGetDeviceInfo(d, CL_DEVICE_VERSION, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting device version");

	if (sscanf(strbuf, "OpenCL %u.%u ", &ocl_major, &ocl_minor) != 2) {
		error = CL_INVALID_VALUE;
		CHECK_ERROR("getting OpenCL version");
	}

	ctx = clCreateContext(ctx_prop, 1, &d, NULL, NULL, &error);
	CHECK_ERROR("create context");

	q = clCreateCommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE, &error);
	CHECK_ERROR("create command queue");

	// let's fire an invalid command
	error = clEnqueueReadBuffer(q, NULL, CL_FALSE, 0, 0, NULL, 0, NULL, &event);

	if (error == CL_SUCCESS) {
		error = CL_INVALID_VALUE;
		CHECK_ERROR("getting clEnqueueReadBuffer error");
	}

	printf("\t" "error %d, event %p (was: %p)\n", error, event, invalid_evt);


out:
	if (q)
		clReleaseCommandQueue(q);

	if (ctx) {
		clReleaseContext(ctx);
		ctx = NULL;
	}
	return error;

}

cl_int test_platform(cl_platform_id p)
{
	cl_uint nd = 0; // number of devices
	cl_device_id *device = NULL; // list of device ids

	unsigned int ocl_major, ocl_minor;

	cl_int error = clGetPlatformInfo(p, CL_PLATFORM_NAME, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting platform name");
	printf("Platform: %s\n", strbuf);

	error = clGetPlatformInfo(p, CL_PLATFORM_VERSION, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting platform version");

	if (sscanf(strbuf, "OpenCL %u.%u ", &ocl_major, &ocl_minor) != 2) {
		error = CL_INVALID_VALUE;
		CHECK_ERROR("getting OpenCL version");
	}

	error = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
	if (error == CL_DEVICE_NOT_FOUND) {
		printf("platform has no valid device, skipping");
		goto out;
	} else {
		CHECK_ERROR("getting amount of device IDs");
	}

	device = calloc(nd, sizeof(*device));
	error = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, device, NULL);

	ctx_prop[1] = (cl_context_properties)p;
	for (cl_uint d = 0; d < nd; ++d) {
		error = test_device(p, device[d]);
		puts("");
	}

out:
	free(device);

	return error;
}

int main(int argc, char *argv[])
{
	cl_int error = CL_SUCCESS;

	error = clGetPlatformIDs(0, NULL, &np);
	CHECK_ERROR("getting amount of platform IDs");
	platform = calloc(np, sizeof(*platform));
	error = clGetPlatformIDs(np, platform, NULL);
	CHECK_ERROR("getting platform IDs");

	// choose platform
	for (cl_uint p = 0; p < np; ++p)
	{
		error = test_platform(platform[p]);
		puts("");
	}

out:
	return error;
}
