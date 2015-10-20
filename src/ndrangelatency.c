/* Measure kernel launch latency */

#include <string.h>
#include <stdio.h>
#include <limits.h>
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

static const char *src[] = {
	"kernel void nop() { return; }\n"
};

// collect statistics over LOOPS runs
#define LOOPS 5
#define MAXWG (1024*1024)

int compare_ulong(const void * restrict _a, const void * restrict _b)
{
	const cl_ulong *a = (const cl_ulong *)_a;
	const cl_ulong *b = (const cl_ulong *)_b;
	if (*a > *b)
		return 1;
	if (*a < *b)
		return -1;
	return 0;
}

cl_int test_device(cl_platform_id p, cl_device_id d)
{
	cl_command_queue q = NULL;
	cl_program pg = NULL;
	cl_kernel nop = NULL;

	// + 1: avg
	cl_ulong submit_time[LOOPS + 1] = {0}; // SUBMIT - QUEUE
	cl_ulong launch_time[LOOPS + 1] = {0}; // START - SUBMIT
	cl_ulong end_time[LOOPS + 1] = {0};    // END - START

	cl_int error = clGetDeviceInfo(d, CL_DEVICE_NAME, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting device name");
	printf("Device: %s\n", strbuf);

	// create context
	ctx_prop[1] = (cl_context_properties)p;
	ctx = clCreateContext(ctx_prop, 1, &d, NULL, NULL, &error);
	CHECK_ERROR("creating context");

	// create queue
	q = clCreateCommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE, &error);
	CHECK_ERROR("creating queue");

	// create program
	pg = clCreateProgramWithSource(ctx, sizeof(src)/sizeof(*src), src, NULL, &error);
	CHECK_ERROR("creating program");

	// build program
	error = clBuildProgram(pg, 1, &d, NULL, NULL, NULL);
#if 1
	if (error == CL_BUILD_PROGRAM_FAILURE) {
		error = clGetProgramBuildInfo(pg, d, CL_PROGRAM_BUILD_LOG,
			BUFSZ, strbuf, NULL);
		CHECK_ERROR("get program build info");
		printf("=== BUILD LOG ===\n%s\n=========\n", strbuf);
	}
#endif
	CHECK_ERROR("building program");

	// get kernels
	nop = clCreateKernel(pg, "nop", &error);
	CHECK_ERROR("creating kernel nop");

	for (size_t gws = 1; gws <= MAXWG; gws *= 1024) {
		memset(submit_time, 0, sizeof(submit_time));
		memset(launch_time, 0, sizeof(launch_time));
		memset(end_time, 0, sizeof(end_time));

		for (int loop = 0; loop < LOOPS; ++loop) {
			cl_event evt;
			cl_ulong queued;
			error = clEnqueueNDRangeKernel(q, nop, 1, NULL, &gws, NULL,
				0, NULL, &evt);
			CHECK_ERROR("enqueue");
			error = clFinish(q);
			CHECK_ERROR("finish");

			error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_QUEUED,
				sizeof(cl_ulong), &queued, NULL);
			CHECK_ERROR("QUEUED");
			error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_SUBMIT,
				sizeof(cl_ulong), submit_time + loop, NULL);
			CHECK_ERROR("SUBMIT");
			error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
				sizeof(cl_ulong), launch_time + loop, NULL);
			CHECK_ERROR("START");
			error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
				sizeof(cl_ulong), end_time + loop, NULL);
			CHECK_ERROR("END");

			end_time[loop] -= launch_time[loop];
			launch_time[loop] -= submit_time[loop];
			submit_time[loop] -= queued;

			submit_time[LOOPS] += submit_time[loop];
			launch_time[LOOPS] += launch_time[loop];
			end_time[LOOPS] += end_time[loop];
		}
		submit_time[LOOPS] /= LOOPS;
		launch_time[LOOPS] /= LOOPS;
		end_time[LOOPS] /= LOOPS;

		qsort(submit_time, LOOPS, sizeof(cl_ulong), compare_ulong);
		qsort(launch_time, LOOPS, sizeof(cl_ulong), compare_ulong);
		qsort(end_time, LOOPS, sizeof(cl_ulong), compare_ulong);

		printf("== %zu work-items ==\n", gws);
		puts("latency in ns\t:\tmin\tmed\tavg\tmax");
		printf("submit\t\t:\t%lu\t%lu\t%lu\t%lu\n",
			submit_time[0], submit_time[LOOPS/2],
			submit_time[LOOPS], submit_time[LOOPS-1]);
		printf("launch\t\t:\t%lu\t%lu\t%lu\t%lu\n",
			launch_time[0], launch_time[LOOPS/2],
			launch_time[LOOPS], launch_time[LOOPS-1]);
		printf("end\t\t:\t%lu\t%lu\t%lu\t%lu\n",
			end_time[0], end_time[LOOPS/2],
			end_time[LOOPS], end_time[LOOPS-1]);
	}

out:
	if (nop)
		clReleaseKernel(nop);
	if (pg)
		clReleaseProgram(pg);
	if (q) {
		clFinish(q);
		clReleaseCommandQueue(q);
	}
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

	cl_int error = clGetPlatformInfo(p, CL_PLATFORM_NAME, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting platform name");
	printf("Platform: %s\n", strbuf);

	error = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
	CHECK_ERROR("getting amount of device IDs");
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
