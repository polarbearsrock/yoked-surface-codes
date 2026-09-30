// Compile with the completed experiment's source_snapshot on the include path.
// This reuses its exact BP arithmetic, graph projection, and UF implementation.
#include "experiments/bp_uf_single_patch_d7/kernel.cc"

namespace fixed_bp_extension {
constexpr int F = 15;
void decode(const evidence::Model& m, const uint8_t* packed, int budget,
            double* out, uint32_t* flags) {
    using namespace evidence;
    std::fill(out, out + F, 0.);
    if (flags) std::fill(flags, flags + m.g.edges.size(), 0);
    auto syndrome = unpack(m.g, packed);
    auto start = Clock::now();
    BP bp(m, syndrome);
    out[12] = seconds(start);
    start = Clock::now();
    for (int k = 0; k < budget; ++k) bp.step(DAMPING);
    out[13] = seconds(start);
    start = Clock::now();
    auto weights = m.project(bp.posterior);
    out[14] = seconds(start);
    auto selected = solve(m.g, weights, syndrome, out, nullptr, 0);
    if (flags) for (int e : selected) flags[e] = 1;
    out[9] = out[12] + out[13] + out[14];
    out[11] = budget;
}
}  // namespace fixed_bp_extension

extern "C" int fixed_bp_decode(void* handle, int shots, int stride,
        const uint8_t* packed, double* out, int threads, int budget,
        uint32_t* flags, int* team_size) {
    if (shots < 0 || threads < 1 || threads > 32 || (budget != 5 && budget != 10))
        return -1;
    auto& m = *static_cast<evidence::Model*>(handle);
    int failures = 0;
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(threads) reduction(+:failures)
    {
        #pragma omp single
        *team_size = omp_get_num_threads();
        #pragma omp for schedule(dynamic,1)
        for (int i = 0; i < shots; ++i) {
            try {
                fixed_bp_extension::decode(m, packed+i*stride, budget,
                    out+i*fixed_bp_extension::F,
                    flags ? flags+i*m.g.edges.size() : nullptr);
            } catch (...) { failures++; }
        }
    }
    return failures;
}
