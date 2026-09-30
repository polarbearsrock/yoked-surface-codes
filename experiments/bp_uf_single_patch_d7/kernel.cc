// Reuse the previously verified BP arithmetic and weighted UF implementation.
#if __has_include("../../repos/yoked-surface-codes/docs/results/uf_evidence_predecoder_d7_d13/kernel.cc")
#include "../../repos/yoked-surface-codes/docs/results/uf_evidence_predecoder_d7_d13/kernel.cc"
#else
#include "../../docs/results/uf_evidence_predecoder_d7_d13/kernel.cc"
#endif
#include <omp.h>

namespace single_patch {
constexpr int NV = 5, F = 12;
// Fields 0..10 match evidence::solve; field 11 records fixed BP iterations.
void mark(const std::vector<int>& edges, uint32_t* flags, int variant) {
    if (flags) for (int e : edges) flags[e] |= uint32_t(1) << variant;
}

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
    mark(first, flags, 0);
    auto start = Clock::now();
    auto adjusted = original;
    for (int e : first) for (auto rule : g.rules[e])
        adjusted[rule.first] = std::min(adjusted[rule.first], rule.second);
    double correlation_seconds = seconds(start);
    if (adjusted == original) {
        std::copy(out, out + F, out + F);
        out[F+9] = correlation_seconds;
        mark(first, flags, 1);
    } else {
        auto second = solve(g, adjusted, syndrome, out + F, nullptr, 0);
        mark(second, flags, 1);
        out[F+9] = out[10] + correlation_seconds;
    }
    start = Clock::now();
    BP bp(m, syndrome);
    double inference_seconds = seconds(start);
    const int budgets[3] = {1, 2, 5};
    int previous = 0;
    for (int b = 0; b < 3; ++b) {
        start = Clock::now();
        for (int iteration = previous; iteration < budgets[b]; ++iteration)
            bp.step(DAMPING);
        inference_seconds += seconds(start);
        start = Clock::now();
        auto weights = m.project(bp.posterior);
        double projection_seconds = seconds(start);
        double* row = out + (b+2)*F;
        auto selected = solve(g, weights, syndrome, row, nullptr, 0);
        mark(selected, flags, b+2);
        row[9] = inference_seconds + projection_seconds;
        row[11] = budgets[b];
        previous = budgets[b];
    }
}
}  // namespace single_patch

extern "C" int single_patch_decode(void* handle, int shots, int stride,
                                   const uint8_t* packed, double* out,
                                   int threads,
                                   uint32_t* flags, int* team_size) {
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
                single_patch::decode(m, packed+i*stride,
                    out+i*single_patch::NV*single_patch::F,
                    flags ? flags+i*m.g.edges.size() : nullptr);
            } catch (...) { failures++; }
        }
    }
    return failures;
}
