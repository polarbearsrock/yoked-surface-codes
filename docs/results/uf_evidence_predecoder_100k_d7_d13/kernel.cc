// Reuse the frozen BP and UF implementations without changing their arithmetic.
#include "../uf_evidence_predecoder_d7_d13/kernel.cc"

namespace full_evidence {
constexpr int VARIANTS=2;
void decode(const evidence::Model &m,const uint8_t *packed,double *out,uint16_t *flags) {
    using namespace evidence;
    std::fill(out,out+VARIANTS*NFIELDS,0.);
    if(flags) std::fill(flags,flags+m.g.edges.size(),0);
    auto syndrome=unpack(m.g,packed);
    auto start=Clock::now();
    BP bp(m,syndrome);
    for(int k=0;k<5;k++) bp.step(DAMPING);
    auto weights=m.project(bp.posterior);
    double bp_seconds=seconds(start);
    auto first=solve(m.g,weights,syndrome,out,flags,0);
    out[9]=bp_seconds;
    start=Clock::now();
    for(int e:first) for(auto rule:m.g.rules[e])
        weights[rule.first]=std::min(weights[rule.first],rule.second);
    double evidence_seconds=bp_seconds+out[10]+seconds(start);
    solve(m.g,weights,syndrome,out+NFIELDS,flags,1);
    out[NFIELDS+9]=evidence_seconds;
}
}

extern "C" int full_evidence_decode(void *handle,int shots,int stride,const uint8_t *packed,
                                     double *out,int threads,uint16_t *flags) {
    auto &m=*static_cast<evidence::Model*>(handle);
    int failures=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(+:failures)
    for(int i=0;i<shots;i++) {
        try { full_evidence::decode(m,packed+i*stride,
            out+i*full_evidence::VARIANTS*evidence::NFIELDS,
            flags ? flags+i*m.g.edges.size() : nullptr); }
        catch(...) { failures++; }
    }
    return failures;
}
