// The include path is the new run's frozen source_snapshot, copied from d=7.
#include "experiments/bp_uf_single_patch_d7/kernel.cc"

namespace distance_sweep {
constexpr int NV = 6, F = 15;
void decode(const evidence::Model& m, const uint8_t* packed, double* out,
            uint32_t* flags) {
    using namespace evidence;
    std::fill(out, out + NV*F, 0.);
    if (flags) std::fill(flags, flags + m.g.edges.size(), 0);
    const auto& g = m.g;
    auto syndrome = unpack(g, packed);
    std::vector<double> original;
    for (auto e : g.edges) original.push_back(e.weight);
    auto first = solve(g, original, syndrome, out, nullptr, 0);
    single_patch::mark(first, flags, 0);
    auto start = Clock::now();
    auto adjusted = original;
    for (int e : first) for (auto rule : g.rules[e])
        adjusted[rule.first] = std::min(adjusted[rule.first], rule.second);
    double correlation_seconds = seconds(start);
    if (adjusted == original) {
        std::copy(out, out + F, out + F);
        out[F+9] = correlation_seconds;
        single_patch::mark(first, flags, 1);
    } else {
        auto second = solve(g, adjusted, syndrome, out + F, nullptr, 0);
        single_patch::mark(second, flags, 1);
        out[F+9] = out[10] + correlation_seconds;
    }
    start = Clock::now();
    BP bp(m, syndrome);
    double initialization_seconds = seconds(start), iteration_seconds = 0.;
    const int budgets[4] = {1, 2, 5, 10};
    int previous = 0;
    for (int b = 0; b < 4; ++b) {
        start = Clock::now();
        for (int k = previous; k < budgets[b]; ++k) bp.step(DAMPING);
        iteration_seconds += seconds(start);
        start = Clock::now();
        auto weights = m.project(bp.posterior);
        double projection_seconds = seconds(start);
        double* row = out + (b+2)*F;
        auto selected = solve(g, weights, syndrome, row, nullptr, 0);
        single_patch::mark(selected, flags, b+2);
        row[9] = initialization_seconds + iteration_seconds + projection_seconds;
        row[11] = budgets[b];
        row[12] = initialization_seconds;
        row[13] = iteration_seconds;
        row[14] = projection_seconds;
        previous = budgets[b];
    }
}
}  // namespace distance_sweep

extern "C" int distance_sweep_decode(void* handle, int shots, int stride,
        const uint8_t* packed, double* out, int threads,
        uint32_t* flags, int* team_size) {
    if (threads < 1 || threads > 32 || shots < 0) return -1;
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
                distance_sweep::decode(m, packed+i*stride,
                    out+i*distance_sweep::NV*distance_sweep::F,
                    flags ? flags+i*m.g.edges.size() : nullptr);
            } catch (...) { failures++; }
        }
    }
    return failures;
}
