#include "../yoked_bp5_weighted_d7/kernel.cc"

namespace sweep {
void correlated(const fourway::Model& topology, const uint8_t* packed,
                double* row, uint8_t* flags, double* costs) {
    using evidence::Clock;
    const auto& g = topology.evidence_model.g;
    std::fill(row, row + fourway::NF, 0.);
    if (flags) std::fill(flags, flags + g.edges.size(), 0);
    auto syndrome = evidence::unpack(g, packed);
    auto start = Clock::now();
    std::vector<double> weights;
    for (const auto& edge : g.edges) weights.push_back(edge.weight);
    row[10] = evidence::seconds(start);
    start = Clock::now();
    auto solution = fourway::solve(g, weights, syndrome);
    row[6] = 1;
    auto adjusted = weights;
    for (int edge : solution.selected)
        for (auto [target, implied] : g.rules[edge])
            adjusted[target] = std::min(adjusted[target], implied);
    for (size_t e = 0; e < weights.size(); ++e) row[7] += adjusted[e] < weights[e];
    if (row[7] != 0) {
        solution = fourway::solve(g, adjusted, syndrome);
        row[6] = 2;
    }
    row[11] = evidence::seconds(start);
    row[0] = solution.mask;
    row[8] = std::count(adjusted.begin(), adjusted.end(), 0.);
    row[9] = solution.max_cluster;
    row[13] = solution.selected.size();
    start = Clock::now();
    for (int sector = 0; sector < 2; ++sector) {
        auto [value, settled] = topology.gap(solution.remaining, sector);
        row[1+sector] = value;
        row[3+sector] = settled;
    }
    row[12] = evidence::seconds(start);
    if (flags) for (int e : solution.selected) flags[e] = 1;
    if (costs) std::copy(solution.remaining.begin(), solution.remaining.end(), costs);
}
}

// variant=-1 evaluates both, 0 evaluates cUF, 1 evaluates BP5+weighted UF.
extern "C" int sweep_decode(void* handle, int shots, int stride,
        const uint8_t* packed, double* out, int threads, int variant,
        uint8_t* flags, double* costs, int* team_size) {
    if (variant < -1 || variant > 1) return 1;
    const auto& model = *static_cast<fourway::Model*>(handle);
    const size_t ne = model.evidence_model.g.edges.size();
    const int nv = variant == -1 ? 2 : 1;
    int failures = 0;
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(threads) reduction(+:failures)
    {
        #pragma omp single
        *team_size = omp_get_num_threads();
        #pragma omp for schedule(dynamic,1)
        for (int i = 0; i < shots; ++i) {
            try {
                for (int k = 0; k < nv; ++k) {
                    int which = variant == -1 ? k : variant;
                    auto function = which == 0 ? sweep::correlated : weighted_followup::decode;
                    function(model, packed + i*stride, out + (i*nv+k)*fourway::NF,
                        flags ? flags + (i*nv+k)*ne : nullptr,
                        costs ? costs + (i*nv+k)*ne : nullptr);
                }
            } catch (...) { ++failures; }
        }
    }
    return failures;
}
