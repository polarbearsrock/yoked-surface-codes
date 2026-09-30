// The prior implementation supplies the unchanged BP, UF, and gap routines.
#include "../yoked_fourway_d7/kernel.cc"

namespace weighted_followup {
void decode(const fourway::Model& topology, const uint8_t* packed,
            double* row, uint8_t* flags, double* costs) {
    using evidence::Clock;
    const auto& m = topology.evidence_model;
    const auto& g = m.g;
    std::fill(row, row + fourway::NF, 0.);
    if (flags) std::fill(flags, flags + g.edges.size(), 0);
    auto syndrome = evidence::unpack(g, packed);
    auto start = Clock::now();
    evidence::BP bp(m, syndrome);
    for (int k = 0; k < 5; ++k) bp.step(evidence::DAMPING);
    auto weights = m.project(bp.posterior);
    row[5] = 5;
    row[10] = evidence::seconds(start);
    start = Clock::now();
    // Exactly one weighted UF pass; the correlation rules are never consulted.
    auto solution = fourway::solve(g, weights, syndrome);
    row[6] = 1;
    row[11] = evidence::seconds(start);
    row[0] = solution.mask;
    row[8] = std::count(weights.begin(), weights.end(), 0.);
    row[9] = solution.max_cluster;
    row[13] = solution.selected.size();
    start = Clock::now();
    for (int sector = 0; sector < 2; ++sector) {
        auto [value, settled] = topology.gap(solution.remaining, sector);
        row[1 + sector] = value;
        row[3 + sector] = settled;
    }
    row[12] = evidence::seconds(start);
    if (flags) for (int e : solution.selected) flags[e] = 1;
    if (costs) std::copy(solution.remaining.begin(), solution.remaining.end(), costs);
}
}

extern "C" int bp5_weighted_decode(void* handle, int shots, int stride,
        const uint8_t* packed, double* out, int threads,
        uint8_t* flags, double* costs, int* team_size) {
    const auto& model = *static_cast<fourway::Model*>(handle);
    size_t ne = model.evidence_model.g.edges.size();
    int failures = 0;
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(threads) reduction(+:failures)
    {
        #pragma omp single
        *team_size = omp_get_num_threads();
        #pragma omp for schedule(dynamic,1)
        for (int i = 0; i < shots; ++i) {
            try {
                weighted_followup::decode(model, packed + i*stride, out + i*fourway::NF,
                    flags ? flags + i*ne : nullptr, costs ? costs + i*ne : nullptr);
            } catch (...) { ++failures; }
        }
    }
    return failures;
}
