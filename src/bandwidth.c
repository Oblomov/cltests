/* Demonstrate OpenCL overallocation and buffer juggling */

#include <string.h>
#include <stdlib.h>
#include <CL/cl.h>

#include "error.h"

cl_uint np; // number of platforms
cl_platform_id *platform; // list of platforms ids
cl_platform_id p; // selected platform

cl_uint nd; // number of devices in the selected platform
cl_device_id *device; // list of device ids
cl_device_id d; // selected device

// context property: field 1 (the platform) will be set at runtime
cl_context_properties ctx_prop[] = { CL_CONTEXT_PLATFORM, 0, 0, 0 };
cl_context ctx; // context
cl_command_queue q; // command queue

// generic string retrieval buffer. quick'n'dirty, hence fixed-size
#define BUFSZ 1024
char strbuf[BUFSZ];

size_t gmem; // device global memory size
size_t alloc_max; // max single-buffer-size on device

cl_uint nbuf; // number of buffers to allocate
cl_mem *buf; // array of allocated buffers

cl_uint nels; // number of elements that fit in the allocated arrays
cl_uint e; // index to iterate over buffer elements on CPU
float **hbuf; // host buffer pointers

// kernel to force usage of the buffer
const char *src[] = {
"kernel void set(global float * restrict dst, global float * restrict src, uint n) {\n",
"	uint i = get_global_id(0);\n",
"	if (i < n) { dst[i] = 0; src[i] = i; }\n",
"}\n"
"kernel void add(global float * restrict dst, global const float * restrict src, uint n) {\n",
"	uint i = get_global_id(0);\n",
"	if (i < n) dst[i] += src[i];\n",
"}"
};

// expected result at a given timestep
float expected;

cl_program pg; // program
cl_kernel k_set, k_add; // actual kernels
size_t gws ; // global work size
size_t wgm ; // preferred workgroup size multiple (will be used as local size too)

// sync events for mem/launch ops
cl_event set_event, add_event, map_event;

// macro to round size to the next multiple of base
#define ROUND_MUL(size, base) \
	((size + base - 1)/base)*base

/* print the event runtime in ms, bandwidth in GB/s assuming
 * nbytes total gmem access (read + write), return runtime
 * in ms
 */
double event_perf(cl_event evt, size_t nbytes, const char *name)
{
	cl_ulong start, end;
	error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
		sizeof(start), &start, NULL);
	CHECK_ERROR("get start");
	error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
		sizeof(end), &end, NULL);
	CHECK_ERROR("get end");
	double time_ms = (end - start)*1.0e-6;
	double bandwidth = (double)(nbytes)/(end - start);
	printf("%s runtime: %gms, B/W: %gGB/s\n", name, time_ms, bandwidth);
	return time_ms;
}

int compare_double(const void *_a, const void *_b)
{
	const double *a = (const double*)_a;
	const double *b = (const double*)_b;
	return *a - *b;
}

int main(int argc, char *argv[])
{
	// selected platform and device number
	cl_uint pn = 0, dn = 0;

	// OpenCL error
	cl_int error;

	// generic iterator
	cl_uint i;

	// set platform/device num from command line
	if (argc > 1)
		pn = atoi(argv[1]);
	if (argc > 2)
		dn = atoi(argv[2]);

	error = clGetPlatformIDs(0, NULL, &np);
	CHECK_ERROR("getting amount of platform IDs");
	printf("%u platforms found\n", np);
	if (pn >= np) {
		fprintf(stderr, "there is no platform #%u\n" , pn);
		exit(1);
	}
	// only allocate for IDs up to the intended one
	platform = calloc(pn+1,sizeof(*platform));
	// if allocation failed, next call will bomb. rely on this
	error = clGetPlatformIDs(pn+1, platform, NULL);
	CHECK_ERROR("getting platform IDs");

	// choose platform
	p = platform[pn];

	error = clGetPlatformInfo(p, CL_PLATFORM_NAME, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting platform name");
	printf("using platform %u: %s\n", pn, strbuf);

	error = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
	CHECK_ERROR("getting amount of device IDs");
	printf("%u devices found\n", nd);
	if (dn >= nd) {
		fprintf(stderr, "there is no device #%u\n", dn);
		exit(1);
	}
	// only allocate for IDs up to the intended one
	device = calloc(dn+1,sizeof(*device));
	// if allocation failed, next call will bomb. rely on this
	error = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, dn+1, device, NULL);
	CHECK_ERROR("getting device IDs");

	// choose device
	d = device[dn];
	error = clGetDeviceInfo(d, CL_DEVICE_NAME, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting device name");
	printf("using device %u: %s\n", dn, strbuf);

	error = clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE,
			sizeof(gmem), &gmem, NULL);
	CHECK_ERROR("getting device global memory size");
	error = clGetDeviceInfo(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
			sizeof(alloc_max), &alloc_max, NULL);
	CHECK_ERROR("getting device max memory allocation size");

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
#if 0
	if (error == CL_BUILD_PROGRAM_FAILURE) {
		error = clGetProgramBuildInfo(pg, d, CL_PROGRAM_BUILD_LOG,
			BUFSZ, strbuf, NULL);
		CHECK_ERROR("get program build info");
		printf("=== BUILD LOG ===\n%s\n=========\n", strbuf);
	}
#endif
	CHECK_ERROR("building program");

	// get kernels
	k_set = clCreateKernel(pg, "set", &error);
	CHECK_ERROR("creating kernel set");

	k_add = clCreateKernel(pg, "add", &error);
	CHECK_ERROR("creating kernel add");


	error = clGetKernelWorkGroupInfo(k_add, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
			sizeof(wgm), &wgm, NULL);
	CHECK_ERROR("getting preferred workgroup size multiple");

	// number of elements on which kernel will be launched. it's ok if we don't
	// cover every byte of the buffers
	nels = alloc_max/sizeof(cl_float);

	gws = ROUND_MUL(nels, wgm);

	printf("will use %zu workitems to process %u elements\n",
			gws, nels);

	// we allocate two buffers
	nbuf = 2;

#define MB (1024*1024.0)

	printf("will try allocating %u buffers of %gMB each\n", nbuf, alloc_max/MB);

	buf = calloc(nbuf, sizeof(cl_mem));

	if (!buf) {
		fprintf(stderr, "could not prepare support for %u buffers\n", nbuf);
		exit(1);
	}

	// we try multiple configurations: no HOST_PTR flags, USE_HOST_PTR and ALLOC_HOST_PTR
	const cl_mem_flags buf_flags[] = {
		CL_MEM_READ_WRITE,
		CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
		CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
		CL_MEM_READ_WRITE,
	};

	const size_t nturns = sizeof(buf_flags)/sizeof(*buf_flags);
	const size_t nloops = 5; // number of loops for each turn, for stats
	const size_t gmem_bytes_rw = sizeof(float)*2*nels;

	const char * const flag_names[] = {
		"(none)", "USE_HOST_PTR", "ALLOC_HOST_PTR", "(none)"
	};

	double runtimes[nturns][3][nloops]; /* set, add, map */
	memset(runtimes, 0, nturns*sizeof(*runtimes));

	hbuf = calloc(nbuf, sizeof(*hbuf));
	if (!hbuf) {
		fputs("couldn't allocate host buffer array\n", stderr);
		exit(1);
	}

	for (size_t turn = 0; turn < sizeof(buf_flags)/sizeof(*buf_flags); ++turn) {
		for (i = 0; i < nbuf; ++i) {
			if (buf_flags[turn] & CL_MEM_USE_HOST_PTR) {
				hbuf[i] = calloc(alloc_max, 1);
				if (!hbuf[i]) {
					fputs("couldn't allocate host buffer array\n", stderr);
					exit(1);
				}
			}
			buf[i] = clCreateBuffer(ctx, buf_flags[turn], alloc_max,
				hbuf[i], &error);
			CHECK_ERROR("allocating buffer");
			printf("buffer %u allocated\n", i);
		}

		for (size_t loop = 0; loop < nloops; ++loop) {
			clSetKernelArg(k_set, 0, sizeof(buf[0]), buf);
			clSetKernelArg(k_set, 1, sizeof(buf[1]), buf + 1);
			clSetKernelArg(k_set, 2, sizeof(nels), &nels);
			error = clEnqueueNDRangeKernel(q, k_set, 1, NULL, &gws, NULL,
					0, NULL, &set_event);
			CHECK_ERROR("enqueueing kernel set");

			clSetKernelArg(k_add, 0, sizeof(buf[0]), buf);
			clSetKernelArg(k_add, 1, sizeof(buf[1]), buf + 1);
			clSetKernelArg(k_add, 2, sizeof(nels), &nels);
			error = clEnqueueNDRangeKernel(q, k_add, 1, NULL, &gws, NULL,
					1, &set_event, &add_event);
			CHECK_ERROR("enqueueing kernel add");

			float *hmap = clEnqueueMapBuffer(q, buf[0], CL_TRUE,
				CL_MAP_READ, 0, alloc_max, 1, &add_event, &map_event, &error);
			CHECK_ERROR("map");

			printf("Turn %zu, loop %zu: %s\n", turn, loop, flag_names[turn]);
			runtimes[turn][0][loop] = event_perf(set_event, gmem_bytes_rw, "set");
			runtimes[turn][1][loop] = event_perf(add_event, gmem_bytes_rw, "add");
			runtimes[turn][2][loop] = event_perf(map_event, alloc_max, "map");

			clEnqueueUnmapMemObject(q, buf[0], hmap, 0, NULL, NULL);

			clFinish(q);

			// release the events
			clReleaseEvent(set_event);
			clReleaseEvent(add_event);
			clReleaseEvent(map_event);
		}

		// release the buffers
		for (i = 0; i < nbuf; ++i) {
			if (buf_flags[turn] & CL_MEM_USE_HOST_PTR) {
				free(hbuf[i]);
				hbuf[i] = NULL;
			}
			clReleaseMemObject(buf[i]);
		}

	}

	puts("Summary/stats:");

	for (size_t turn = 0; turn < nturns; ++turn) {
		double avg[3] = {0};

		/* I'm lazy, so sort with qsort and then compute average,
		 * otherwise we could just compute min, max, avg and median together */
		qsort(runtimes[turn][0], nloops, sizeof(double), compare_double);
		qsort(runtimes[turn][1], nloops, sizeof(double), compare_double);
		qsort(runtimes[turn][2], nloops, sizeof(double), compare_double);
		for (size_t loop = 0; loop < nloops; ++loop) {
			avg[0] += runtimes[turn][0][loop];
			avg[1] += runtimes[turn][1][loop];
			avg[2] += runtimes[turn][2][loop];
		}
		avg[0] /= nloops;
		avg[1] /= nloops;
		avg[2] /= nloops;

		printf("Turn %zu: %s\n", turn, flag_names[turn]);
		printf("set\ttime (ms): min: %8g, median: %8g, max: %8g, avg: %8g\n",
			runtimes[turn][0][0],
			runtimes[turn][0][(nloops + 1)/2],
			runtimes[turn][0][nloops - 1],
			avg[0]);
		printf("\tBW (GB/s): min: %8g, median: %8g, max: %8g, avg: %8g\n",
			gmem_bytes_rw/runtimes[turn][0][0]*1.0e-6,
			gmem_bytes_rw/runtimes[turn][0][(nloops + 1)/2]*1.0e-6,
			gmem_bytes_rw/runtimes[turn][0][nloops - 1]*1.0e-6,
			gmem_bytes_rw/avg[0]*1.0e-6);
		printf("add\ttime (ms): min: %8g,, median: %8g max: %8g, avg: %8g\n",
			runtimes[turn][1][0],
			runtimes[turn][1][(nloops + 1)/2],
			runtimes[turn][1][nloops - 1],
			avg[1]);
		printf("\tBW (GB/s): min: %8g, median: %8g, max: %8g, avg: %8g\n",
			gmem_bytes_rw/runtimes[turn][1][0]*1.0e-6,
			gmem_bytes_rw/runtimes[turn][1][(nloops + 1)/2]*1.0e-6,
			gmem_bytes_rw/runtimes[turn][1][nloops - 1]*1.0e-6,
			gmem_bytes_rw/avg[1]*1.0e-6);
		printf("map\ttime (ms): min: %8g, median: %8g, max: %8g, avg: %8g\n",
			runtimes[turn][2][0],
			runtimes[turn][2][(nloops + 1)/2],
			runtimes[turn][2][nloops - 1],
			avg[2]);
		printf("\tBW (GB/s): min: %8g, median: %8g, max: %8g, avg: %8g\n",
			alloc_max/runtimes[turn][2][0]*1.0e-6,
			alloc_max/runtimes[turn][2][(nloops + 1)/2]*1.0e-6,
			alloc_max/runtimes[turn][2][nloops - 1]*1.0e-6,
			alloc_max/avg[2]*1.0e-6);

	}


	return 0;
}
