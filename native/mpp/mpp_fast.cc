// User-supplied MPP cluster-score adapter, integrated against PyMatching 2.4.0.
//
// This preserves the supplied live-graph, radius-gathering, bucket-queue and
// incremental algorithms. Integration adds input/exception checks and an optional
// independent full-scan + heap-Dijkstra oracle, outside normal decoding.
//
// A score is the shortest path between opposite logical boundaries with length
// max(0, w - r(u) - r(v)), using final local radii before blossoms are shattered.
// This is a cluster-gap approximation, NOT an exact complementary matching gap.
// Correlations change the engine's own weights; no copied graph needs syncing.
// The default bucket width may exceed the lightest edge to limit bucket count.
// Re-scanning each bucket until stable makes the search exact at any width.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <bit>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"

// These upstream driver functions have external linkage but are not declared
// in the public header. Keeping the engine revision pinned is therefore required.
void process_timeline_until_completion(pm::Mwpm&, const std::vector<uint64_t>&);
pm::MatchingResult shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(
    pm::Mwpm&, const std::vector<uint64_t>&);

namespace py = pybind11;
namespace {
constexpr int64_t INF = std::numeric_limits<int64_t>::max();
}

class FastSoftDecoder {
    pm::Mwpm mwpm;
    size_t n;
    double scale;
    bool correlated;
    std::string correlation;
    std::vector<int> sector, sector_obs;

    // Compact adjacency omits boundaries. The weight pointers refer directly
    // to the engine's stable arrays, reweighted in place for each shot.
    std::vector<uint32_t> offset, target;
    std::vector<uint8_t> skip;
    std::vector<const pm::weight_int*> weights;
    std::vector<uint32_t> sources[2];
    std::vector<uint8_t> is_target;

    // Only flooded nodes have nonzero radii; clear only the nodes we touched.
    std::vector<uint32_t> flooded;
    std::vector<pm::GraphFillRegion*> tops;

    // Shortening edges cannot increase a shortest path. Static scores are thus
    // upper bounds; static distances initialize the incremental algorithm.
    bool incremental;
    std::vector<int64_t> static_dist[2], unreached;
    int64_t static_score[2];
    std::vector<uint32_t> first_rule, rule_offset;
    std::vector<std::pair<uint32_t, uint32_t>> rule_edge;
    std::vector<int64_t> matched_edges;
    std::vector<std::pair<uint32_t, uint32_t>> lowered[2];
    std::vector<uint8_t> basis;

    struct Slot {
        int64_t dist;
        uint32_t stamp;
        int32_t radius;
    };
    std::vector<Slot> slot;
    const int64_t* init = nullptr;
    uint32_t search_id = 0;
    std::vector<std::vector<uint64_t>> buckets;
    size_t last_bucket = 0;
    std::vector<uint64_t> staged;
    int bucket_shift = 0;
    int node_bits;
    uint64_t node_mask;

    void validate_shots(const py::array_t<uint8_t>& shots) const {
        if (shots.ndim() != 2 || size_t(shots.shape(1)) != n)
            throw std::invalid_argument("Expected shots with shape (shots, detectors)");
        auto s = shots.unchecked<2>();
        for (py::ssize_t i = 0; i < shots.shape(0); ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (s(i, j) > 1) throw std::invalid_argument("Shots must be binary");
                if (s(i, j) && sector[j] < 0)
                    throw std::invalid_argument("Nonzero isolated detector");
            }
        }
    }

    void gather_radii(const std::vector<uint64_t>& events) {
        auto* base = mwpm.flooder.graph.nodes.data();
        tops.clear();
        for (auto e : events) {
            if (auto* top = base[e].region_that_arrived_top) tops.push_back(top);
        }
        std::sort(tops.begin(), tops.end(), std::less<pm::GraphFillRegion*>{});
        tops.erase(std::unique(tops.begin(), tops.end()), tops.end());
        for (auto* top : tops) {
            top->do_op_for_each_node_in_total_area([&](pm::DetectorNode* node) {
                uint32_t v = uint32_t(node - base);
                int64_t r = int64_t(node->region_that_arrived_top->radius.y_intercept())
                            + node->wrapped_radius_cached;
                if (r < 0 || r > std::numeric_limits<int32_t>::max())
                    throw std::logic_error("Cluster radius out of range");
                // Record first, so cleanup also works if allocation throws.
                flooded.push_back(v);
                slot[v].radius = int32_t(r);
            });
        }
    }

    void clear_radii() {
        for (uint32_t v : flooded) slot[v].radius = 0;
        flooded.clear();
    }

    void recover_after_exception() {
        clear_radii();
        for (auto& bucket : buckets) bucket.clear();
        last_bucket = 0;
        if (correlated) mwpm.flooder.graph.undo_reweights();
        mwpm.reset();
    }

    int64_t boundary_length(uint32_t u) const {
        return std::max<int64_t>(0, int64_t(weights[u][0]) - slot[u].radius);
    }

    int64_t edge_length(uint32_t u, uint32_t v, const pm::weight_int& w) const {
        return std::max<int64_t>(0, int64_t(w) - slot[u].radius - slot[v].radius);
    }

    void push(int64_t d, uint64_t key) {
        size_t k = size_t(d) >> bucket_shift;
        if (k >= buckets.size()) buckets.resize(k + 1);
        buckets[k].push_back(key);
        last_bucket = std::max(last_bucket, k);
    }

    void relax(uint32_t v, int64_t d) {
        Slot& s = slot[v];
        if (d < (s.stamp == search_id ? s.dist : init[v])) {
            s.stamp = search_id;
            s.dist = d;
            push(d, (uint64_t(d) << node_bits) | v);
        }
    }

    // Offer both directions of a shortened edge. v == n denotes a boundary.
    void seed(int b, uint32_t u, uint32_t v, int64_t length, int64_t& best) {
        const int64_t* a = static_dist[b].data();
        if (a[u] == INF) return;
        if (v == n) {
            if (is_target[u] & (1 << b)) best = std::min(best, a[u] + length);
            else if (length < best) relax(u, length);
            return;
        }
        if (a[u] + length < best) relax(v, a[u] + length);
        if (a[v] != INF && a[v] + length < best) relax(u, a[v] + length);
    }

    // Collect every potentially shortened edge: flooded endpoints or a rule
    // triggered by the first-pass matching. Duplicate rule targets are harmless.
    void collect_lowered_edges() {
        lowered[0].clear();
        lowered[1].clear();
        for (uint32_t x : flooded) {
            auto& list = lowered[basis[x]];
            if (skip[x]) list.push_back({x, 0});
            for (uint32_t e = offset[x], k = skip[x]; e < offset[x + 1]; ++e, ++k) {
                uint32_t y = target[e];
                if (slot[y].radius == 0 || x < y) list.push_back({x, k});
            }
        }
        auto& nodes = mwpm.flooder.graph.nodes;
        for (size_t i = 0; i + 1 < matched_edges.size(); i += 2) {
            int64_t u = matched_edges[i], v = matched_edges[i + 1];
            size_t r = first_rule[u] + nodes[u].index_of_neighbor(v == -1 ? nullptr : &nodes[v]);
            for (size_t j = rule_offset[r]; j < rule_offset[r + 1]; ++j)
                lowered[basis[rule_edge[j].first]].push_back(rule_edge[j]);
        }
    }

    void seed_lowered_edges(int b, int64_t& best) {
        for (auto [x, k] : lowered[b]) {
            if (skip[x] && k == 0) seed(b, x, uint32_t(n), boundary_length(x), best);
            else {
                uint32_t y = target[offset[x] + k - skip[x]];
                seed(b, x, y, edge_length(x, y, weights[x][k]), best);
            }
        }
    }

    // Multi-source ordinary boundary -> multi-target logical boundary. Full
    // searches build the static distance fields; shot searches prune at best.
    template <bool Incremental>
    int64_t search(int b, bool full = false) {
        if (++search_id == 0) {
            for (auto& s : slot) s.stamp = 0;
            search_id = 1;
        }
        int64_t best = full ? INF : static_score[b];
        if (Incremental) {
            init = static_dist[b].data();
            seed_lowered_edges(b, best);
        } else {
            init = unreached.data();
            for (uint32_t u : sources[b]) {
                if (boundary_length(u) < best) relax(u, boundary_length(u));
            }
        }
        const int64_t* start = init;
        const uint32_t id = search_id;
        int64_t prune = full ? INF : best;
        size_t k = 0;
        for (; k <= last_bucket && int64_t(k << bucket_shift) < prune; ++k) {
            // Re-index after each push: growing buckets can invalidate references.
            while (!buckets[k].empty()) {
                uint64_t key = buckets[k].back();
                buckets[k].pop_back();
                uint32_t u = uint32_t(key & node_mask);
                int64_t d = int64_t(key >> node_bits);
                if (d != slot[u].dist || d >= prune) continue;
                if (is_target[u] & (1 << b)) {
                    best = std::min(best, d + boundary_length(u));
                    if (!full) prune = best;
                }
                // Stage successful relaxations before queue insertion, retaining
                // the supplied adapter's branch-free inner neighbor loop.
                int64_t ru = slot[u].radius;
                const pm::weight_int* w = weights[u] + skip[u];
                size_t improved = 0;
                for (uint32_t e = offset[u], end = offset[u + 1]; e < end; ++e, ++w) {
                    uint32_t v = target[e];
                    Slot& sv = slot[v];
                    int64_t nd = d + std::max<int64_t>(0, int64_t(*w) - ru - sv.radius);
                    int64_t current = sv.stamp == id ? sv.dist : (Incremental ? start[v] : INF);
                    bool better = nd < std::min(current, prune);
                    sv.dist = better ? nd : sv.dist;
                    sv.stamp = better ? id : sv.stamp;
                    staged[improved] = (uint64_t(nd) << node_bits) | v;
                    improved += better;
                }
                for (size_t j = 0; j < improved; ++j)
                    push(int64_t(staged[j] >> node_bits), staged[j]);
            }
        }
        for (; k <= last_bucket; ++k) buckets[k].clear();
        last_bucket = 0;
        return best;
    }

    // Slow validation oracle: traverse the original adjacency with a binary
    // heap, read radii by a full scan, and use neither static bounds nor stamps.
    // Sharing only the live engine state makes this an independent check of
    // the optimized graph layout, radius gather, and both search methods.
    int64_t reference_score(int b) const {
        const auto& nodes = mwpm.flooder.graph.nodes;
        std::vector<int64_t> radius(n), distance(n, INF);
        for (size_t v = 0; v < n; ++v) {
            const auto& node = nodes[v];
            radius[v] = node.region_that_arrived_top == nullptr ? 0
                : int64_t(node.region_that_arrived_top->radius.y_intercept()) + node.wrapped_radius_cached;
            if (radius[v] != slot[v].radius) throw std::logic_error("Radius gather disagrees with full scan");
        }
        using Entry = std::pair<int64_t, uint32_t>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
        for (uint32_t u = 0; u < n; ++u) {
            if (sector[u] < 0 || sector_obs[sector[u]] != b) continue;
            for (size_t k = 0; k < nodes[u].neighbors.size(); ++k) {
                if (!nodes[u].neighbors[k] && !nodes[u].neighbor_observables[k]) {
                    distance[u] = std::max<int64_t>(0, int64_t(nodes[u].neighbor_weights[k]) - radius[u]);
                    queue.push({distance[u], u});
                }
            }
        }
        int64_t best = INF;
        while (!queue.empty()) {
            auto [d, u] = queue.top();
            queue.pop();
            if (d != distance[u]) continue;
            for (size_t k = 0; k < nodes[u].neighbors.size(); ++k) {
                auto* neighbor = nodes[u].neighbors[k];
                int64_t w = nodes[u].neighbor_weights[k];
                if (!neighbor) {
                    if (nodes[u].neighbor_observables[k] & (1 << b))
                        best = std::min(best, d + std::max<int64_t>(0, w - radius[u]));
                } else {
                    uint32_t v = uint32_t(neighbor - nodes.data());
                    int64_t nd = d + std::max<int64_t>(0, w - radius[u] - radius[v]);
                    if (nd < distance[v]) {
                        distance[v] = nd;
                        queue.push({nd, v});
                    }
                }
            }
        }
        return best;
    }

public:
    FastSoftDecoder(const std::string& dem_text, const std::string& mode, const std::string& method)
        : correlation(mode) {
        if (method != "incremental" && method != "dijkstra")
            throw std::invalid_argument("Search method must be incremental or dijkstra");
        incremental = method == "incremental";
        if (mode != "none" && mode != "cross" && mode != "all")
            throw std::invalid_argument("Correlation mode must be none, cross, or all");
        correlated = mode != "none";
        stim::DetectorErrorModel dem(dem_text.c_str());
        auto user = pm::detector_error_model_to_user_graph(dem, correlated, pm::NUM_DISTINCT_WEIGHTS);
        mwpm = user.to_mwpm(pm::NUM_DISTINCT_WEIGHTS, correlated);
        auto& g = mwpm.flooder.graph;
        n = g.num_nodes;
        scale = g.normalising_constant;
        if (g.num_observables != 2 || g.negative_weight_sum != 0)
            throw std::invalid_argument("Expected two observables and nonnegative weights");
        if (n >= std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("Too many detectors");

        // Connected components ignoring boundaries identify the two CSS sectors.
        sector.assign(n, -1);
        int count = 0;
        for (size_t root = 0; root < n; ++root) {
            if (sector[root] >= 0 || g.nodes[root].neighbors.empty()) continue;
            std::queue<size_t> q;
            q.push(root);
            sector[root] = count;
            while (!q.empty()) {
                auto u = q.front();
                q.pop();
                for (auto vptr : g.nodes[u].neighbors) if (vptr) {
                    size_t v = vptr - g.nodes.data();
                    if (sector[v] < 0) { sector[v] = count; q.push(v); }
                }
            }
            ++count;
        }
        if (count != 2) throw std::invalid_argument("Expected two detector sectors");
        sector_obs.assign(count, -1);
        for (size_t u = 0; u < n; ++u) {
            auto& node = g.nodes[u];
            for (size_t k = 0; k < node.neighbors.size(); ++k) {
                auto mask = node.neighbor_observables[k];
                if (!mask) continue;
                if (node.neighbors[k] || (mask != 1 && mask != 2))
                    throw std::invalid_argument("Expected one logical label on boundary edges only");
                int b = mask == 1 ? 0 : 1;
                auto& previous = sector_obs.at(sector[u]);
                if (previous != -1 && previous != b) throw std::invalid_argument("Mixed logical sectors");
                previous = b;
            }
        }
        if (sector_obs[0] < 0 || sector_obs[1] < 0 || sector_obs[0] == sector_obs[1])
            throw std::invalid_argument("Missing logical boundary");

        basis.assign(n, 0);
        for (size_t u = 0; u < n; ++u) if (sector[u] >= 0) basis[u] = uint8_t(sector_obs[sector[u]]);
        offset.assign(n + 1, 0);
        skip.assign(n, 0);
        weights.assign(n, nullptr);
        is_target.assign(n, 0);
        uint64_t weight_sum = 0;
        for (size_t u = 0; u < n; ++u) {
            auto& node = g.nodes[u];
            offset[u] = uint32_t(target.size());
            weights[u] = node.neighbor_weights.data();
            for (size_t k = 0; k < node.neighbors.size(); ++k) {
                weight_sum += node.neighbor_weights[k];
                if (node.neighbors[k]) {
                    target.push_back(uint32_t(node.neighbors[k] - g.nodes.data()));
                    continue;
                }
                if (k != 0) throw std::invalid_argument("Expected boundary neighbor at index zero");
                skip[u] = 1;
                int b = sector_obs.at(sector[u]);
                if (node.neighbor_observables[k]) is_target[u] |= 1 << b;
                else sources[b].push_back(u);
            }
        }
        if (target.size() >= std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Too many graph edges");
        offset[n] = uint32_t(target.size());
        for (int b = 0; b < 2; ++b) {
            bool has_target = false;
            for (size_t u = 0; u < n; ++u) has_target |= bool(is_target[u] & (1 << b));
            if (sources[b].empty() || !has_target) throw std::invalid_argument("Empty logical boundary");
        }
        node_bits = std::max(1, int(std::bit_width(n)));
        node_mask = (uint64_t(1) << node_bits) - 1;
        if (std::bit_width(weight_sum) + node_bits > 63)
            throw std::invalid_argument("Weights too large to pack");

        if (mode == "cross") {
            std::unordered_map<pm::weight_int*, int> ptr_sector;
            for (size_t u = 0; u < n; ++u)
                for (auto& w : g.nodes[u].neighbor_weights) ptr_sector[&w] = sector[u];
            for (size_t u = 0; u < n; ++u) for (auto& rules : g.nodes[u].neighbor_implied_weights) {
                std::erase_if(rules, [&](const pm::ImpliedWeight& r) {
                    return ptr_sector.at(r.edge0_ptr) == sector[u];
                });
            }
        }
        slot.assign(n, Slot{0, 0, 0});
        size_t max_degree = 0;
        for (size_t u = 0; u < n; ++u) max_degree = std::max<size_t>(max_degree, offset[u + 1] - offset[u]);
        staged.assign(max_degree, 0);
        unreached.assign(n, INF);
        uint64_t lightest = std::numeric_limits<uint64_t>::max();
        for (size_t u = 0; u < n; ++u)
            for (auto w : g.nodes[u].neighbor_weights) if (w > 0) lightest = std::min<uint64_t>(lightest, w);
        if (lightest == std::numeric_limits<uint64_t>::max()) lightest = 1;
        bucket_shift = int(std::bit_width(std::max<uint64_t>(lightest, 1 << 16))) - 1;
        for (int b = 0; b < 2; ++b) {
            static_score[b] = search<false>(b, true);
            static_dist[b].assign(n, INF);
            for (size_t v = 0; v < n; ++v)
                if (slot[v].stamp == search_id) static_dist[b][v] = slot[v].dist;
        }
        uint64_t bound = uint64_t(std::max(static_score[0], static_score[1]));
        uint64_t width = std::max<uint64_t>(lightest, (bound >> 12) + 1);
        bucket_shift = int(std::bit_width(width)) - 1;
        buckets.assign((bound >> bucket_shift) + 2, {});

        if (!incremental) return;
        std::unordered_map<const pm::weight_int*, std::pair<uint32_t, uint32_t>> owner;
        for (size_t u = 0; u < n; ++u)
            for (size_t k = 0; k < g.nodes[u].neighbor_weights.size(); ++k)
                owner[&g.nodes[u].neighbor_weights[k]] = {uint32_t(u), uint32_t(k)};
        first_rule.assign(n, 0);
        rule_offset.push_back(0);
        for (size_t u = 0; u < n; ++u) {
            first_rule[u] = uint32_t(rule_offset.size() - 1);
            if (correlated && g.nodes[u].neighbor_implied_weights.size() != g.nodes[u].neighbors.size())
                throw std::invalid_argument("Expected correlation rules for every edge");
            for (auto& rules : g.nodes[u].neighbor_implied_weights) {
                for (auto& rule : rules) rule_edge.push_back(owner.at(rule.edge0_ptr));
                rule_offset.push_back(uint32_t(rule_edge.size()));
            }
        }
    }

    py::dict info() const {
        py::dict d;
        d["detectors"] = n;
        d["observables"] = 2;
        d["correlation"] = correlation;
        d["integer_weight_scale"] = scale;
        d["sector_ids"] = sector;
        d["sector_observables"] = sector_obs;
        d["score"] = "MPP cluster-gap approximation, not exact complementary gap";
        d["engine"] = "PyMatching 2.4.0; shortest paths on the live matching graph";
        d["method"] = incremental ? "incremental" : "dijkstra";
        return d;
    }

    py::tuple decode_batch(const py::array_t<uint8_t>& shots, bool soft, bool verify) {
        validate_shots(shots);
        if (verify && !soft) throw std::invalid_argument("verify requires soft=True");
        auto s = shots.unchecked<2>();
        py::array_t<uint8_t> predictions({shots.shape(0), py::ssize_t(2)});
        py::array_t<double> scores({shots.shape(0), py::ssize_t(soft ? 2 : 0)});
        py::array_t<double> ws_array(shots.shape(0));
        auto pred = predictions.mutable_unchecked<2>();
        auto phi = scores.mutable_unchecked<2>();
        auto ws = ws_array.mutable_unchecked<1>();
        std::vector<uint64_t> events;
        auto& g = mwpm.flooder.graph;
        for (py::ssize_t i = 0; i < shots.shape(0); ++i) {
            events.clear();
            for (size_t j = 0; j < n; ++j) if (s(i, j)) events.push_back(j);
            try {
                if (correlated) {
                    matched_edges.clear();
                    pm::decode_detection_events_to_edges(mwpm, events, matched_edges);
                    g.reweight_for_edges(matched_edges);
                }
                process_timeline_until_completion(mwpm, events);
                // Scores must be read before shattering destroys region radii.
                if (soft) {
                    gather_radii(events);
                    if (incremental) collect_lowered_edges();
                    for (int b = 0; b < 2; ++b) {
                        int64_t score = incremental ? search<true>(b) : search<false>(b);
                        if (verify && score != reference_score(b))
                            throw std::logic_error("MPP score disagrees with heap-Dijkstra oracle");
                        phi(i, b) = double(score) / scale;
                    }
                    clear_radii();
                }
                auto res = shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(mwpm, events);
                pred(i, 0) = res.obs_mask & 1;
                pred(i, 1) = (res.obs_mask >> 1) & 1;
                ws(i) = double(res.weight) / scale;
                if (correlated) g.undo_reweights();
            } catch (...) {
                recover_after_exception();
                throw;
            }
        }
        return py::make_tuple(predictions, scores, ws_array);
    }

    // Original test hook: number of shots whose gathered radii disagree with
    // a full scan. This is intentionally excluded from normal decode timing.
    size_t radius_gather_mismatches(const py::array_t<uint8_t>& shots) {
        validate_shots(shots);
        auto s = shots.unchecked<2>();
        auto& g = mwpm.flooder.graph;
        std::vector<uint64_t> events;
        std::vector<int64_t> edges;
        size_t bad = 0;
        for (py::ssize_t i = 0; i < shots.shape(0); ++i) {
            events.clear();
            for (size_t j = 0; j < n; ++j) if (s(i, j)) events.push_back(j);
            try {
                if (correlated) {
                    edges.clear();
                    pm::decode_detection_events_to_edges(mwpm, events, edges);
                    g.reweight_for_edges(edges);
                }
                process_timeline_until_completion(mwpm, events);
                gather_radii(events);
                size_t scanned = 0;
                bool same = true;
                for (size_t v = 0; v < n; ++v) {
                    auto& node = g.nodes[v];
                    int64_t expected = node.region_that_arrived_top == nullptr ? 0
                        : int64_t(node.region_that_arrived_top->radius.y_intercept()) + node.wrapped_radius_cached;
                    scanned += node.region_that_arrived_top != nullptr;
                    same &= slot[v].radius == expected;
                }
                bad += !same || scanned != flooded.size();
                clear_radii();
                shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(mwpm, events);
                if (correlated) g.undo_reweights();
            } catch (...) {
                recover_after_exception();
                throw;
            }
        }
        return bad;
    }
};

PYBIND11_MODULE(_mpp_fast, m) {
    py::class_<FastSoftDecoder>(m, "SoftDecoder", py::module_local())
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("dem"), py::arg("correlation") = "none", py::arg("method") = "dijkstra")
        .def("info", &FastSoftDecoder::info)
        .def("decode_batch", &FastSoftDecoder::decode_batch, py::arg("shots").noconvert(),
             py::arg("soft") = true, py::arg("verify") = false)
        .def("radius_gather_mismatches", &FastSoftDecoder::radius_gather_mismatches,
             py::arg("shots").noconvert());
}
