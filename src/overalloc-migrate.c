/* Demonstrate OpenCL overallocation and buffer juggling */

#include <string.h>
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
float *hbuf; // host buffer pointer

// kernel to force usage of the buffer
const char *src[] = {
"kernel void add(global float *dst, global const float *src, uint n) {\n",
"	uint i = get_global_id(0);\n",
"	if (i < n) dst[i] += src[i];\n",
"}"
};

// expected result at a given timestep
float expected;

cl_program pg; // program
cl_kernel k; // actual kernel
size_t gws ; // global work size
size_t wgm ; // preferred workgroup size multiple (will be used as local size too)

// sync events for mem/launch ops
cl_event mem_evt, krn_evt;

// macro to round size to the next multiple of base
#define ROUND_MUL(size, base) \
	((size + base - 1)/base)*base

int main(int argc, char *argv[])
{
	// selected platform and device number
	cl_uint pn = 0, dn = 0;

	// OpenCL error
	cl_int error;

	// generic iterator
	cl_uint i;

	// major/minor version of the platform OpenCL version
	uint ocl_major, ocl_minor;

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

	error = clGetPlatformInfo(p, CL_PLATFORM_VERSION, BUFSZ, strbuf, NULL);
	CHECK_ERROR("getting platform version");
	// we need 1.2 at least
	i = sscanf(strbuf, "OpenCL %u.%u ", &ocl_major, &ocl_minor);
	if (i != 2) {
		fprintf(stderr, "%s:%u: unable to determine platform OpenCL version\n",
			__func__, __LINE__);
		exit(1);
	}
	if (ocl_major == 1 && ocl_minor < 2) {
		fprintf(stderr, "%s:%u: Platform version %s is not at least 1.2\n",
			__func__, __LINE__, strbuf);
		exit(1);
	}

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
	CHECK_ERROR("building program");

	// get kernel
	k = clCreateKernel(pg, "add", &error);
	CHECK_ERROR("creating kernel");

	error = clGetKernelWorkGroupInfo(k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
			sizeof(wgm), &wgm, NULL);
	CHECK_ERROR("getting preferred workgroup size multiple");

	// number of elements on which kernel will be launched. it's ok if we don't
	// cover every byte of the buffers
	nels = alloc_max/sizeof(cl_float);

	gws = ROUND_MUL(nels, wgm);

	printf("will use %zu workitems grouped by %zu to process %u elements\n",
			gws, wgm, nels);

	// we will try and allocate at least one buffer more than needed to fill
	// the device memory, and no less than 3 anyway
	nbuf = gmem/alloc_max + 1;
	if (nbuf < 3)
		nbuf = 3;

#define MB (1024*1024.0)

	printf("will try allocating %u buffers of %gMB each to overcommit %gMB\n",
			nbuf, alloc_max/MB, gmem/MB);

	buf = calloc(nbuf, sizeof(cl_mem));

	if (!buf) {
		fprintf(stderr, "could not prepare support for %u buffers\n", nbuf);
		exit(1);
	}

	for (i = 0; i < nbuf; ++i) {
		buf[i] = clCreateBuffer(ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, alloc_max,
				NULL, &error);
		CHECK_ERROR("allocating buffer");
		printf("buffer %u allocated\n", i);
	}

	// memset the first buffer
	hbuf = clEnqueueMapBuffer(q, buf[0], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
			0, alloc_max, 0, NULL, NULL, &error);
	CHECK_ERROR("mapping buffer 0");
	memset(hbuf, 0, alloc_max);
	error = clEnqueueUnmapMemObject(q, buf[0], hbuf, 0, NULL, NULL);
	CHECK_ERROR("unmapping buffer 0");
	hbuf = NULL;

	// use the buffers
	for (i = 1; i < nbuf; ++i) {
		printf("testing buffer %u\n", i);

		// for each buffer, we do a setup on CPU and then use it as second
		// argument for the kernel
		hbuf = clEnqueueMapBuffer(q, buf[i], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
				0, alloc_max, 0, NULL, NULL, &error);
		CHECK_ERROR("mapping buffer");
		for (e = 0; e < nels; ++e)
			hbuf[e] = i;
		error = clEnqueueUnmapMemObject(q, buf[i], hbuf, 0, NULL, NULL);
		CHECK_ERROR("unmapping buffer");
		hbuf = NULL;

		// migrate previous buffer out of the GPU
		if (i > 1) {
			error = clEnqueueMigrateMemObjects(q, 1, buf + i-1,
					CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
					0, NULL, NULL);
			CHECK_ERROR("migrating previous buffer to host");
		}
		// make sure all pending actions are completed
		error =	clFinish(q);
		CHECK_ERROR("settling down");

		clSetKernelArg(k, 0, sizeof(buf[0]), buf);
		clSetKernelArg(k, 1, sizeof(buf[i]), buf + i);
		clSetKernelArg(k, 2, sizeof(nels), &nels);
		error = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &wgm,
				0, NULL, &krn_evt);
		CHECK_ERROR("enqueueing kernel");

		expected = i*(i+1)/2.0f;
		hbuf = clEnqueueMapBuffer(q, buf[0], CL_TRUE, CL_MAP_READ,
				0, alloc_max, 1, &krn_evt, NULL, &error);
		CHECK_ERROR("mapping buffer 0");
		for (e = 0; e < nels; ++e)
			if (hbuf[e] != expected) {
				fprintf(stderr, "mismatch @ %u: %g instead of %g\n",
						e, hbuf[e], expected);
				exit(1);
			}
		error = clEnqueueUnmapMemObject(q, buf[0], hbuf, 0, NULL, NULL);
		CHECK_ERROR("unmapping buffer 0");
		hbuf = NULL;
		clReleaseEvent(krn_evt); // free up the kernel event
	}

	for (i = 1; i <= nbuf; ++i) {
		clReleaseMemObject(buf[nbuf - i]);
		printf("buffer %u freed\n", nbuf  - i);
	}

	return 0;
}
