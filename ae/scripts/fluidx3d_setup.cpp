// New D.5-style AE driver. The archived historical patch is preserved separately.
#include "setup.hpp"
#include "info.hpp"
#include <cassert>
#include <cstdlib>
#include <iomanip>

void main_setup() {
    assert(main_arguments.size() == 1u); // Explicit OpenCL ordinal, not a CUDA ordinal.
    const uint n = static_cast<uint>(std::stoul(std::getenv("AE_N")));
    const ulong batch = static_cast<ulong>(std::stoull(std::getenv("AE_BATCH")));
    assert(n >= 3u && batch > 0ull);
    const auto devices = get_devices(false);
    const uint selected = to_uint(main_arguments[0]);
    assert(selected < devices.size());
    println("AE_OPENCL_DEVICE " + devices[selected].name);
    assert(contains(devices[selected].cl_device.getInfo<CL_DEVICE_EXTENSIONS>(), "cl_khr_device_uuid"));
    const auto uuid = devices[selected].cl_device.getInfo<CL_DEVICE_UUID_KHR>();
    const char* hex = "0123456789abcdef";
    string identity = "GPU-";
    for (uint index = 0u; index < CL_UUID_SIZE_KHR; ++index) {
        if (index == 4u || index == 6u || index == 8u || index == 10u) identity += '-';
        identity += hex[(uuid[index] >> 4u) & 15u];
        identity += hex[uuid[index] & 15u];
    }
    println("AE_OPENCL_UUID " + identity);
    LBM lbm(n, n, n, 0.5f); // omega=1/(3*nu+1/2)=0.5; FP32/FP32.
    parallel_for(lbm.get_N(), [&](ulong index) {
        uint x = 0u, y = 0u, z = 0u;
        lbm.coordinates(index, x, y, z);
        if (x == 0u || x + 1u == n || y == 0u || y + 1u == n || z == 0u || z + 1u == n)
            lbm.flags[index] = TYPE_S;
    });
    for (uint window = 0u; window < 6u; ++window) {
        const double before = info.runtime_lbm;
        const ulong step = lbm.get_t();
        lbm.run(batch, 6ull * batch);
        const double seconds = info.runtime_lbm - before;
        assert(lbm.get_t() - step == batch && seconds > 0.0);
        std::cout << "\nAE_WINDOW " << std::setprecision(12)
                  << double(lbm.get_N()) * double(batch) * 1.0e-6 / seconds << '\n';
    }
}
