// Experimental kernel only. Production UF and its defaults are unchanged.
// The heap simulates continuous parallel growth; it is not an FPGA design.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr int VARIANTS = 6, FIELDS = 15;
using Pair = std::array<double, 2>;
Pair combine(Pair a, Pair b) {
    return {std::min(a[0]+b[0], a[1]+b[1]), std::min(a[0]+b[1], a[1]+b[0])};
}
struct Edge { int u, v; double weight; uint32_t mask; };
struct Graph {
    int nd, n;
    std::vector<Edge> edges;
    std::vector<std::vector<int>> adjacency;
    std::vector<std::vector<std::pair<int, double>>> rules;
};
struct DSU {
    std::vector<int> parent, size;
    explicit DSU(int n): parent(n), size(n, 1) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int v) {
        while (v != parent[v]) { parent[v] = parent[parent[v]]; v = parent[v]; }
        return v;
    }
    std::pair<int,int> join(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return {a, -1};
        if (size[a] < size[b] || (size[a] == size[b] && a > b)) std::swap(a,b);
        parent[b] = a; size[a] += size[b];
        return {a,b};
    }
};
struct Growth: DSU {
    const Graph &g;
    const std::vector<double> &weights;
    bool normalized;
    std::vector<uint8_t> parity, terminal;
    std::vector<std::unordered_set<int>> frontier;
    std::vector<int> forest, generation;
    std::vector<double> grown, settled, rate;
    using Event = std::tuple<double,int,int>;
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> heap;
    int active_count = 0, epochs = 0, max_cluster = 1, max_frontier = 0;
    uint64_t evaluations = 0, frontier_visits = 0;
    double now = 0;
    Growth(const Graph &graph, const std::vector<double> &w, const std::vector<uint8_t> &s, bool norm):
        DSU(graph.n), g(graph), weights(w), normalized(norm), parity(s), terminal(g.n),
        frontier(g.n), generation(w.size()), grown(w.size()), settled(w.size()), rate(w.size()) {
        for (int v=0; v<g.n; v++) {
            terminal[v] = v >= g.nd;
            frontier[v].insert(g.adjacency[v].begin(), g.adjacency[v].end());
            max_frontier = std::max(max_frontier, int(frontier[v].size()));
            active_count += active(v);
        }
        for (int e=0; e<int(w.size()); e++) {
            rate[e] = speed(g.edges[e].u) + speed(g.edges[e].v);
            if (w[e] == 0) heap.emplace(0,e,0);
            else if (rate[e]) heap.emplace(w[e]/rate[e],e,0);
        }
        evaluations = w.size();
    }
    bool active(int r) const { return parity[r] && !terminal[r]; }
    double speed(int r) const {
        if (!active(r)) return 0;
        if (!normalized) return 1;
        // Power-of-two approximation to one unit of total frontier growth.
        size_t denominator=1;
        while (denominator < frontier[r].size()) denominator <<= 1;
        return 1.0 / double(denominator);
    }
    void settle(int e) {
        grown[e] = std::min(weights[e], grown[e] + rate[e]*(now-settled[e]));
        settled[e] = now;
    }
    double next() {
        while (!heap.empty() && std::get<2>(heap.top()) != generation[std::get<1>(heap.top())]) heap.pop();
        return heap.empty() ? INF : std::get<0>(heap.top());
    }
    void merge(std::vector<int> &edges) {
        std::sort(edges.begin(),edges.end());
        std::vector<int> parts;
        for (int e: edges) { parts.push_back(find(g.edges[e].u)); parts.push_back(find(g.edges[e].v)); }
        std::sort(parts.begin(),parts.end()); parts.erase(std::unique(parts.begin(),parts.end()),parts.end());
        std::vector<double> before_speed;
        std::vector<std::vector<int>> before_frontier;
        for (int r: parts) {
            before_speed.push_back(speed(r));
            before_frontier.emplace_back(frontier[r].begin(),frontier[r].end());
            frontier_visits += frontier[r].size();
        }
        std::unordered_set<int> internal;
        for (int e: edges) {
            int a=find(g.edges[e].u), b=find(g.edges[e].v);
            if (a==b) continue;
            active_count -= int(active(a))+int(active(b));
            auto roots=join(a,b); a=roots.first; b=roots.second;
            parity[a] ^= parity[b]; terminal[a] |= terminal[b];
            active_count += active(a);
            for (int f: frontier[b]) {
                if (frontier[a].erase(f)) internal.insert(f);
                else frontier[a].insert(f);
            }
            frontier[b].clear();
            max_cluster=std::max(max_cluster,size[a]);
            max_frontier=std::max(max_frontier,int(frontier[a].size()));
            forest.push_back(e);
        }
        for (int e: internal) {
            settle(e);
            if (std::binary_search(edges.begin(),edges.end(),e)) grown[e]=weights[e];
            rate[e]=0; generation[e]++;
        }
        std::unordered_set<int> affected;
        for (size_t k=0; k<parts.size(); k++)
            if (before_speed[k] != speed(find(parts[k])))
                affected.insert(before_frontier[k].begin(),before_frontier[k].end());
        for (int e: affected) {
            if (internal.count(e)) continue;
            evaluations++;
            int a=find(g.edges[e].u), b=find(g.edges[e].v);
            double new_rate = a==b ? 0 : speed(a)+speed(b);
            if (new_rate==rate[e]) continue;
            settle(e); rate[e]=new_rate; generation[e]++;
            if (new_rate) heap.emplace(now+std::max(0.,weights[e]-grown[e])/new_rate,e,generation[e]);
        }
    }
    void run() {
        while (active_count || next()==0) {
            now=next();
            if (!std::isfinite(now)) throw std::runtime_error("unresolved syndrome");
            double limit=now+1e-12+1e-12*std::abs(now);
            epochs++;
            while (next()<=limit) {
                std::vector<int> edges;
                while (next()<=limit) { edges.push_back(std::get<1>(heap.top())); heap.pop(); }
                merge(edges);
            }
        }
        for (int e=0; e<int(weights.size()); e++) settle(e);
    }
};

// Min-sum messages on a forest. The downward pass obtains the cost of
// toggling the syndrome at each endpoint, enabling cheap bridge scoring.
struct Tree {
    const Graph &g;
    const std::vector<double> &weights;
    const std::vector<uint8_t> &syndrome;
    std::vector<std::vector<std::pair<int,int>>> adjacency, children;
    std::vector<int> parent, parent_edge, order, roots, depth;
    std::vector<Pair> up, incoming;
    std::vector<double> flip_delta;
    int max_depth=0;
    double cost=0;
    Tree(const Graph &graph, const std::vector<double> &w, const std::vector<uint8_t> &s,
         const std::vector<int> &forest): g(graph), weights(w), syndrome(s), adjacency(g.n), children(g.n),
        parent(g.n,-1), parent_edge(g.n,-1), depth(g.n), up(g.n), incoming(g.n), flip_delta(g.n) {
        for (int e: forest) {
            auto edge=g.edges[e]; adjacency[edge.u].emplace_back(edge.v,e); adjacency[edge.v].emplace_back(edge.u,e);
        }
        // Same root and traversal convention as production peeling.
        for (int i=0; i<g.n; i++) {
            int r=(i+g.nd)%g.n;
            if (parent[r]!=-1) continue;
            roots.push_back(r); parent[r]=r;
            size_t begin=order.size(); order.push_back(r);
            for (size_t j=begin; j<order.size(); j++) {
                int u=order[j];
                for (auto ve: adjacency[u]) {
                    int v=ve.first;
                    if (v==parent[u]) continue;
                    if (parent[v]!=-1) throw std::runtime_error("cycle in correction forest");
                    parent[v]=u; parent_edge[v]=ve.second; depth[v]=depth[u]+1;
                    max_depth=std::max(max_depth,depth[v]); children[u].push_back(ve); order.push_back(v);
                }
            }
        }
        for (auto it=order.rbegin(); it!=order.rend(); it++) {
            int u=*it; Pair a={0,INF};
            for (auto ve: children[u]) a=combine(a,{up[ve.first][0],up[ve.first][1]+weights[ve.second]});
            up[u]=constrain(u,a);
        }
        for (int r: roots) cost+=up[r][0];
        if (!std::isfinite(cost)) throw std::runtime_error("invalid forest syndrome");
    }
    Pair constrain(int u, Pair a) const {
        if (u>=g.nd) return {std::min(a[0],a[1]),std::min(a[0],a[1])};
        return {a[syndrome[u]],a[1^syndrome[u]]};
    }
    void messages() {
        for (int r: roots) incoming[r]={0,INF};
        for (int u: order) {
            const auto &ch=children[u];
            std::vector<Pair> prefix(ch.size()+1),suffix(ch.size()+1);
            prefix[0]=incoming[u]; suffix[ch.size()]={0,INF};
            for (size_t j=0; j<ch.size(); j++) prefix[j+1]=combine(prefix[j],{up[ch[j].first][0],up[ch[j].first][1]+weights[ch[j].second]});
            for (int j=int(ch.size())-1; j>=0; j--) suffix[j]=combine(suffix[j+1],{up[ch[j].first][0],up[ch[j].first][1]+weights[ch[j].second]});
            Pair all=constrain(u,prefix.back());
            flip_delta[u]=all[1]-all[0];
            for (size_t j=0; j<ch.size(); j++) {
                Pair out=constrain(u,combine(prefix[j],suffix[j+1]));
                out[1]+=weights[ch[j].second]; incoming[ch[j].first]=out;
            }
        }
    }
    std::vector<int> peel() const {
        auto parity=syndrome; std::vector<int> selected;
        for (auto it=order.rbegin(); it!=order.rend(); it++) {
            int u=*it;
            if (u!=parent[u] && u<g.nd && parity[u]) {
                selected.push_back(parent_edge[u]); parity[parent[u]]^=1;
            }
        }
        for (int r: roots) if (r<g.nd && parity[r]) throw std::runtime_error("peeling residual");
        return selected;
    }
    std::vector<int> optimal() const {
        std::vector<int> selected, parent_bit(g.n);
        for (int u: order) {
            Pair a={0,INF};
            std::vector<std::array<int,2>> history;
            for (auto ve: children[u]) {
                Pair option={up[ve.first][0],up[ve.first][1]+weights[ve.second]};
                std::array<int,2> h;
                Pair next;
                for (int total=0; total<2; total++) {
                    double c0=a[0]+option[total],c1=a[1]+option[1^total];
                    h[total]=c1<c0 ? 1 : 0; next[total]=std::min(c0,c1);
                }
                history.push_back(h); a=next;
            }
            int parity=u<g.nd ? syndrome[u]^parent_bit[u] : int(a[1]<a[0]);
            for (int j=int(children[u].size())-1; j>=0; j--) {
                int previous=history[j][parity], bit=parity^previous;
                auto ve=children[u][j]; parent_bit[ve.first]=bit;
                if (bit) selected.push_back(ve.second);
                parity=previous;
            }
            if (parity) throw std::runtime_error("invalid traceback");
        }
        return selected;
    }
};
struct Extension {
    std::vector<int> forest, selected;
    int rounds=0, added=0, eligible=0, improving=0, max_pairs=0, max_depth=0;
    uint64_t message_edges=0;
};
Extension extend(Growth &growth, const std::vector<uint8_t> &s) {
    const auto &g=growth.g; const auto &w=growth.weights;
    Extension result; result.forest=growth.forest;
    DSU groups(g.n);
    for (int e: result.forest) groups.join(g.edges[e].u,g.edges[e].v);
    std::vector<int> eligible;
    for (int e=0; e<int(w.size()); e++) {
        auto edge=g.edges[e];
        if (groups.find(edge.u)!=groups.find(edge.v) && std::max(0.,w[e]-growth.grown[e])<=0.5) eligible.push_back(e);
    }
    result.eligible=eligible.size();
    for (int round=0; round<2; round++) {
        Tree tree(g,w,s,result.forest); tree.messages();
        result.max_depth=std::max(result.max_depth,tree.max_depth);
        result.message_edges+=2*result.forest.size(); // two directed min-sum messages per tree edge
        result.rounds++;
        std::vector<int> best(g.n,-1);
        std::vector<double> deltas(w.size(),INF);
        for (int e: eligible) {
            auto edge=g.edges[e]; int a=groups.find(edge.u),b=groups.find(edge.v);
            if (a==b) continue;
            double delta=w[e]+tree.flip_delta[edge.u]+tree.flip_delta[edge.v];
            if (!(delta < -1e-9)) continue;
            result.improving++; deltas[e]=delta;
            for (int r: {a,b})
                if (best[r]<0 || std::make_pair(delta,e)<std::make_pair(deltas[best[r]],best[r])) best[r]=e;
        }
        std::vector<int> accepted;
        for (int e: eligible) {
            auto edge=g.edges[e]; int a=groups.find(edge.u),b=groups.find(edge.v);
            if (a!=b && best[a]==e && best[b]==e) accepted.push_back(e);
        }
        result.max_pairs=std::max(result.max_pairs,int(accepted.size()));
        if (accepted.empty()) break;
        for (int e: accepted) {
            auto roots=groups.join(g.edges[e].u,g.edges[e].v);
            if (roots.second<0) throw std::runtime_error("refinement cycle");
            result.forest.push_back(e); result.added++;
        }
    }
    Tree final(g,w,s,result.forest);
    result.max_depth=std::max(result.max_depth,final.max_depth);
    result.message_edges+=2*result.forest.size(); // upward solve and downward traceback
    result.selected=final.optimal();
    return result;
}
void record(const Graph &g, const std::vector<double> &w, const std::vector<uint8_t> &s,
            Growth &growth, const std::vector<int> &selected, int depth, const Extension *extension,
            double *out, uint16_t *flags, int flag_bit) {
    uint32_t mask=0; double cost=0; auto residual=s;
    for (int e: selected) {
        auto edge=g.edges[e]; residual[edge.u]^=1; residual[edge.v]^=1;
        cost+=w[e]; mask^=edge.mask;
        if (flags) flags[e]|=uint16_t(1<<flag_bit);
    }
    for (int v=0; v<g.nd; v++) if (residual[v]) throw std::runtime_error("correction violates syndrome");
    out[0]=mask; out[1]=cost; out[2]=selected.size(); out[3]=growth.epochs;
    out[4]=growth.evaluations; out[5]=growth.frontier_visits; out[6]=growth.max_cluster;
    out[7]=growth.max_frontier; out[8]=depth; out[9]=growth.now;
    if (extension) {
        out[10]=extension->added; out[11]=extension->eligible; out[12]=extension->improving;
        out[13]=extension->message_edges; out[14]=extension->max_pairs;
    }
}
void decode(const Graph &g, const uint8_t *packed, double *out, uint16_t *flags) {
    std::fill(out,out+VARIANTS*FIELDS,0);
    if (flags) std::fill(flags,flags+g.edges.size(),0);
    std::vector<uint8_t> syndrome(g.n);
    for (int v=0; v<g.nd; v++) syndrome[v]=(packed[v/8]>>(v%8))&1;
    std::vector<double> original;
    for (auto edge:g.edges) original.push_back(edge.weight);
    Growth first(g,original,syndrome,false); first.run();
    Tree first_tree(g,original,syndrome,first.forest);
    auto first_selected=first_tree.peel(); auto weights=original;
    for (int e:first_selected) {
        if (flags) flags[e]|=1;
        for (auto rule:g.rules[e]) weights[rule.first]=std::min(weights[rule.first],rule.second);
    }
    for (int policy=0; policy<2; policy++) {
        Growth growth(g,weights,syndrome,policy); growth.run();
        Tree tree(g,weights,syndrome,growth.forest);
        auto peeled=tree.peel(), optimal=tree.optimal();
        auto extension=extend(growth,syndrome);
        if (flags) {
            for (int e:growth.forest) flags[e]|=uint16_t(1<<(1+policy));
            for (int e:extension.forest) flags[e]|=uint16_t(1<<(9+policy));
        }
        record(g,weights,syndrome,growth,peeled,tree.max_depth,nullptr,out+(policy*3)*FIELDS,flags,3+policy*3);
        record(g,weights,syndrome,growth,optimal,tree.max_depth,nullptr,out+(policy*3+1)*FIELDS,flags,4+policy*3);
        record(g,weights,syndrome,growth,extension.selected,extension.max_depth,&extension,out+(policy*3+2)*FIELDS,flags,5+policy*3);
        if (out[(policy*3+2)*FIELDS+1] > tree.cost+1e-7) throw std::runtime_error("refinement increases tree cost");
    }
}
} // namespace

extern "C" {
void *uf_create(int nd,int n,int ne,const int32_t *ends,const double *weights,const uint32_t *masks,
                int nr,const int32_t *rule_ends,const double *rule_weights) {
    auto *g=new Graph; g->nd=nd; g->n=n; g->adjacency.resize(n); g->rules.resize(ne);
    for (int e=0;e<ne;e++) {
        int u=ends[2*e],v=ends[2*e+1];
        g->edges.push_back({u,v,weights[e],masks[e]});
        g->adjacency[u].push_back(e); g->adjacency[v].push_back(e);
    }
    for(int r=0;r<nr;r++) g->rules[rule_ends[2*r]].emplace_back(rule_ends[2*r+1],rule_weights[r]);
    return g;
}
void uf_destroy(void *g) { delete static_cast<Graph*>(g); }
int uf_decode(void *graph,int shots,int stride,const uint8_t *packed,double *out,int threads,uint16_t *flags) {
    const auto &g=*static_cast<Graph*>(graph);
    int failures=0;
    #pragma omp parallel for num_threads(threads) reduction(+:failures) schedule(dynamic,1)
    for (int i=0;i<shots;i++) {
        try { decode(g,packed+i*stride,out+i*VARIANTS*FIELDS,flags ? flags+i*g.edges.size() : nullptr); }
        catch (...) { failures++; std::fill(out+i*VARIANTS*FIELDS,out+(i+1)*VARIANTS*FIELDS,-1); }
    }
    return failures;
}
}
