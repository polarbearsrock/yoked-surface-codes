// Reuse the verified BP arithmetic and weighted UF implementation unchanged.
#if __has_include("../../repos/yoked-surface-codes/docs/results/uf_evidence_predecoder_d7_d13/kernel.cc")
#include "../../repos/yoked-surface-codes/docs/results/uf_evidence_predecoder_d7_d13/kernel.cc"
#else
#include "../../docs/results/uf_evidence_predecoder_d7_d13/kernel.cc"
#endif
#include <omp.h>

namespace fourway {
constexpr int NV = 2, NF = 14;
// mask, gap0, gap1, states0, states1, BP iterations, UF passes,
// discounted edges, zero edges, max cluster, evidence s, UF s, gap s, correction edges.
struct Solution {
    std::vector<int> selected;
    std::vector<double> remaining;
    uint32_t mask = 0;
    int max_cluster = 0;
};

Solution solve(const Graph& g, const std::vector<double>& weights,
               const std::vector<uint8_t>& syndrome) {
    Growth growth(g, weights, syndrome, false);
    growth.run();
    Tree tree(g, weights, syndrome, growth.forest);
    Solution result;
    result.selected = tree.peel();
    result.max_cluster = growth.max_cluster;
    auto residual = syndrome;
    for (int e : result.selected) {
        const auto edge = g.edges[e];
        result.mask ^= edge.mask;
        residual[edge.u] ^= 1;
        residual[edge.v] ^= 1;
    }
    for (int v = 0; v < g.nd; ++v)
        if (residual[v]) throw std::runtime_error("Invalid physical correction");
    result.remaining.resize(weights.size());
    for (int e = 0; e < int(weights.size()); ++e) {
        const auto edge = g.edges[e];
        result.remaining[e] = growth.find(edge.u) == growth.find(edge.v)
            ? 0. : std::max(0., weights[e] - growth.grown[e]);
    }
    return result;
}

struct Model {
    const evidence::Model& evidence_model;
    std::vector<std::vector<std::pair<int,int>>> adjacency[2];
    explicit Model(const evidence::Model& m) : evidence_model(m) {
        const auto& g = m.g;
        DSU components(g.nd);
        for (const auto& e : g.edges)
            if (e.u < g.nd && e.v < g.nd) components.join(e.u, e.v);
        for (int k = 0; k < 2; ++k) {
            int component = -1;
            for (const auto& e : g.edges) if ((e.mask >> k) & 1) {
                int label = components.find(e.u);
                if (component >= 0 && label != component)
                    throw std::runtime_error("Observable spans multiple components");
                component = label;
            }
            // Synthetic graphs may carry just one observable.
            adjacency[k].resize(g.nd + 1);
            if (component < 0) continue;
            for (int i = 0; i < int(g.edges.size()); ++i) {
                const auto& e = g.edges[i];
                if (components.find(e.u) != component) continue;
                int u = e.u < g.nd ? e.u : g.nd;
                int v = e.v < g.nd ? e.v : g.nd;
                adjacency[k][u].emplace_back(i, v);
                adjacency[k][v].emplace_back(i, u);
            }
        }
    }

    std::pair<double,int> gap(const std::vector<double>& costs, int sector) const {
        const auto& g = evidence_model.g;
        if (adjacency[sector][g.nd].empty()) return {0., 0};
        int start = 2*g.nd, target = start+1;
        std::vector<double> distances(2*(g.nd+1), INF);
        using Item = std::pair<double,int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
        distances[start] = 0.;
        queue.emplace(0., start);
        int settled = 0;
        while (!queue.empty()) {
            auto [distance, state] = queue.top(); queue.pop();
            if (distance > distances[state]) continue;
            ++settled;
            if (state == target) return {distance, settled};
            int vertex = state/2, parity = state%2;
            for (auto [edge, other] : adjacency[sector][vertex]) {
                int next = 2*other + (parity ^ ((g.edges[edge].mask >> sector) & 1));
                double candidate = distance + costs[edge];
                if (candidate < distances[next]) {
                    distances[next] = candidate;
                    queue.emplace(candidate, next);
                }
            }
        }
        throw std::runtime_error("No logical boundary walk");
    }

    void decode(const uint8_t* packed, double* out, uint8_t* flags, double* costs) const {
        using evidence::Clock;
        const auto& m = evidence_model;
        const auto& g = m.g;
        std::fill(out, out + NV*NF, 0.);
        if (flags) std::fill(flags, flags+g.edges.size(), 0);
        auto syndrome = evidence::unpack(g, packed);
        for (int variant = 0; variant < NV; ++variant) {
            double* row = out + variant*NF;
            auto start = Clock::now();
            std::vector<double> weights;
            if (variant == 0) {
                for (const auto& edge : g.edges) weights.push_back(edge.weight);
            } else {
                evidence::BP bp(m, syndrome);
                for (int iteration = 0; iteration < 5; ++iteration) bp.step(evidence::DAMPING);
                weights = m.project(bp.posterior);
                row[5] = 5;
            }
            row[10] = evidence::seconds(start);
            start = Clock::now();
            auto solution = solve(g, weights, syndrome);
            row[6] = 1;
            auto adjusted = weights;
            for (int edge : solution.selected)
                for (auto [target, implied] : g.rules[edge])
                    adjusted[target] = std::min(adjusted[target], implied);
            for (size_t e = 0; e < weights.size(); ++e) row[7] += adjusted[e] < weights[e];
            if (row[7] != 0) {
                solution = solve(g, adjusted, syndrome);
                row[6] = 2;
            }
            row[11] = evidence::seconds(start);
            row[0] = solution.mask;
            row[8] = std::count(adjusted.begin(), adjusted.end(), 0.);
            row[9] = solution.max_cluster;
            row[13] = solution.selected.size();
            start = Clock::now();
            for (int sector = 0; sector < 2; ++sector) {
                auto [value, settled] = gap(solution.remaining, sector);
                row[1+sector] = value;
                row[3+sector] = settled;
            }
            row[12] = evidence::seconds(start);
            if (flags) for (int edge : solution.selected) flags[edge] |= 1 << variant;
            if (costs) std::copy(solution.remaining.begin(), solution.remaining.end(),
                                 costs + variant*g.edges.size());
        }
    }
};
}  // namespace fourway

extern "C" {
void* fourway_create(void* evidence_handle) {
    try { return new fourway::Model(*static_cast<evidence::Model*>(evidence_handle)); }
    catch (...) { return nullptr; }
}
void fourway_destroy(void* handle) { delete static_cast<fourway::Model*>(handle); }
int fourway_decode(void* handle, int shots, int stride, const uint8_t* packed,
                   double* out, int threads, uint8_t* flags, double* costs, int* team_size) {
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
                model.decode(packed+i*stride, out+i*fourway::NV*fourway::NF,
                    flags ? flags+i*ne : nullptr,
                    costs ? costs+i*fourway::NV*ne : nullptr);
            } catch (...) { ++failures; }
        }
    }
    return failures;
}
}
