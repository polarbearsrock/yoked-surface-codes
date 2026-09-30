// Reuse the frozen, verified UF implementation without editing its behavior.
#include "../parallel_uf_clustering_d7_d9_p003_100k/kernel.cc"
#include <chrono>

namespace evidence {
constexpr int NV=5, NFIELDS=11;
constexpr double LIMIT=30., DAMPING=.5;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
double clip(double x) { return std::max(-LIMIT,std::min(LIMIT,x)); }
struct Model {
    const Graph &g;
    int nf;
    std::vector<double> probability, prior, prior_weights;
    std::vector<int> offsets, detectors, edge_offsets, edges;
    std::vector<std::vector<int>> checks;
    Model(const Graph &graph,int count,const double *p,const int32_t *o,const int32_t *d,
          const int32_t *eo,const int32_t *es):g(graph),nf(count),probability(p,p+count),
        offsets(o,o+count+1),detectors(d,d+o[count]),edge_offsets(eo,eo+count+1),
        edges(es,es+eo[count]),checks(g.nd) {
        for(double value:probability) {
            if (!(value>0 && value<1)) throw std::runtime_error("bad prior");
            prior.push_back(std::log1p(-value)-std::log(value));
        }
        for(int j=0;j<int(detectors.size());j++) checks.at(detectors[j]).push_back(j);
        prior_weights=project(probability);
    }
    std::vector<double> project(const std::vector<double> &posterior) const {
        std::vector<double> weights(g.edges.size());
        for(int f=0;f<nf;f++)
            for(int j=edge_offsets[f];j<edge_offsets[f+1];j++) weights[edges[j]]+=posterior[f];
        for(double &w:weights) w=-std::log(std::min(1.,std::max(1e-15,w)));
        return weights;
    }
};
struct BP {
    const Model &m;
    const std::vector<uint8_t> &syndrome;
    std::vector<double> q,r,t,prefix,total,posterior;
    BP(const Model &model,const std::vector<uint8_t> &s):m(model),syndrome(s),
        q(m.detectors.size()),r(q.size()),t(q.size()),prefix(q.size()),total(m.prior),
        posterior(m.probability) {
        for(int f=0;f<m.nf;f++)
            for(int j=m.offsets[f];j<m.offsets[f+1];j++) q[j]=clip(m.prior[f]);
    }
    void step(double damping) {
        const double cap=std::tanh(LIMIT/2);
        for(size_t j=0;j<q.size();j++) t[j]=std::tanh(q[j]/2);
        for(int detector=0;detector<m.g.nd;detector++) {
            double product=syndrome[detector] ? -1. : 1.;
            for(int j:m.checks[detector]) { prefix[j]=product; product*=t[j]; }
            double suffix=1.;
            for(auto it=m.checks[detector].rbegin();it!=m.checks[detector].rend();++it) {
                int j=*it;
                double x=std::max(-cap,std::min(cap,prefix[j]*suffix));
                double message=std::log1p(x)-std::log1p(-x);
                r[j]=(1-damping)*r[j]+damping*message;
                suffix*=t[j];
            }
        }
        for(int f=0;f<m.nf;f++) {
            double value=m.prior[f];
            for(int j=m.offsets[f];j<m.offsets[f+1];j++) value+=r[j];
            total[f]=value;
            for(int j=m.offsets[f];j<m.offsets[f+1];j++) q[j]=clip(value-r[j]);
            posterior[f]=1./(1.+std::exp(clip(value)));
        }
    }
};
std::vector<uint8_t> unpack(const Graph &g,const uint8_t *packed) {
    std::vector<uint8_t> s(g.n);
    for(int v=0;v<g.nd;v++) s[v]=(packed[v/8]>>(v%8))&1;
    return s;
}
std::vector<int> solve(const Graph &g,const std::vector<double> &weights,
                      const std::vector<uint8_t> &syndrome,double *out,uint16_t *flags,int variant) {
    auto start=Clock::now();
    Growth growth(g,weights,syndrome,false); growth.run();
    Tree tree(g,weights,syndrome,growth.forest);
    auto selected=tree.peel();
    out[10]=seconds(start);
    uint32_t mask=0; double cost=0; auto residual=syndrome;
    for(int e:selected) {
        const auto edge=g.edges[e];
        mask^=edge.mask; cost+=weights[e];
        residual[edge.u]^=1; residual[edge.v]^=1;
        if(flags) flags[e]|=uint16_t(1<<(2*variant+1));
    }
    for(int v=0;v<g.nd;v++) if(residual[v]) throw std::runtime_error("invalid correction");
    if(flags) for(int e:growth.forest) flags[e]|=uint16_t(1<<(2*variant));
    out[0]=mask; out[1]=cost; out[2]=selected.size(); out[3]=growth.forest.size();
    out[4]=growth.max_cluster; out[5]=growth.epochs;
    out[6]=growth.evaluations; out[7]=growth.frontier_visits;
    out[8]=std::count(weights.begin(),weights.end(),0.);
    return selected;
}
void decode(const Model &m,const uint8_t *packed,double *out,uint16_t *flags) {
    std::fill(out,out+NV*NFIELDS,0.);
    if(flags) std::fill(flags,flags+m.g.edges.size(),0);
    const auto &g=m.g;
    auto s=unpack(g,packed);
    std::vector<double> original;
    for(auto edge:g.edges) original.push_back(edge.weight);
    auto first=solve(g,original,s,out,flags,0);
    auto start=Clock::now();
    auto weights=original;
    for(int e:first) for(auto rule:g.rules[e]) weights[rule.first]=std::min(weights[rule.first],rule.second);
    double correlation_seconds=seconds(start)+out[10];
    solve(g,weights,s,out+NFIELDS,flags,1);
    out[NFIELDS+9]=correlation_seconds;
    solve(g,m.prior_weights,s,out+2*NFIELDS,flags,2);
    start=Clock::now();
    BP bp(m,s);
    double message_seconds=seconds(start);
    int last=0;
    for(int variant=3;variant<NV;variant++) {
        int iterations=variant==3 ? 5 : 20;
        start=Clock::now();
        for(int k=last;k<iterations;k++) bp.step(DAMPING);
        message_seconds+=seconds(start);
        start=Clock::now();
        weights=m.project(bp.posterior);
        double evidence_seconds=message_seconds+seconds(start);
        solve(g,weights,s,out+variant*NFIELDS,flags,variant);
        out[variant*NFIELDS+9]=evidence_seconds;
        last=iterations;
    }
}
} // namespace evidence

extern "C" {
void *evidence_create(void *graph,int nf,const double *p,const int32_t *offsets,const int32_t *detectors,
                       const int32_t *edge_offsets,const int32_t *edges) {
    try { return new evidence::Model(*static_cast<Graph*>(graph),nf,p,offsets,detectors,edge_offsets,edges); }
    catch(...) { return nullptr; }
}
void evidence_destroy(void *handle) { delete static_cast<evidence::Model*>(handle); }
int evidence_weights(void *handle,const uint8_t *packed,int iterations,double damping,
                     double *posterior,double *weights) {
    try {
        auto &m=*static_cast<evidence::Model*>(handle);
        auto syndrome=evidence::unpack(m.g,packed);
        evidence::BP bp(m,syndrome);
        for(int k=0;k<iterations;k++) bp.step(damping);
        auto w=m.project(bp.posterior);
        std::copy(bp.posterior.begin(),bp.posterior.end(),posterior);
        std::copy(w.begin(),w.end(),weights);
        return 0;
    } catch(...) { return 1; }
}
int evidence_decode(void *handle,int shots,int stride,const uint8_t *packed,double *out,
                     int threads,uint16_t *flags) {
    auto &m=*static_cast<evidence::Model*>(handle);
    int failures=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(+:failures)
    for(int i=0;i<shots;i++) {
        try { evidence::decode(m,packed+i*stride,out+i*evidence::NV*evidence::NFIELDS,
                               flags ? flags+i*m.g.edges.size() : nullptr); }
        catch(...) { failures++; }
    }
    return failures;
}
}
