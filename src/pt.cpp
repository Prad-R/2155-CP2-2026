// Parallel tempering (replica exchange) for the 100-grid set under exact advisor validity.
// Same moves, rules and objective as src/anneal.cpp; R replicas run at fixed temperatures in threads and
// neighbouring temperatures exchange configurations.  The best state ever seen is kept in the state file
// (same header layout as anneal.cpp, so src/anneal.py can read it).
// Build:  g++ -O3 -mavx2 -std=c++17 -pthread -o bin/pt src/pt.cpp
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static const int N = 100, L = 49, K = 10, NB = 51;
static int MAN[L][L];
static int THR[4] = {70, 17, -187, -1155};
static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }
static int P_nswap = 200, P_dmin = 38, P_max_tries = 128;
static double P_focus = 15.0, P_lam = 0.0;

struct Rng {
    uint64_t s[4];
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    void seed(uint64_t z) { for (int i = 0; i < 4; i++) { z += 0x9e3779b97f4a7c15ULL; uint64_t x = z; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; s[i] = x ^ (x >> 31); } }
    uint64_t next() { uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45); return r; }
    double uni() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    int below(int n) { return (int)(uni() * n); }
    double gumbel() { return -std::log(-std::log(uni())); }
    double expo() { return -std::log(uni()); }
};

static bool raw4(const uint8_t *g, int *raw) {
    int cnt[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < L; i++) cnt[g[i]]++;
    if (!cnt[0] || !cnt[1] || !cnt[2] || !cnt[3]) return false;
    int nR = cnt[0], nI = cnt[1], nC = cnt[2], nP = cnt[3], nO = cnt[4];
    int well = 0, tax = 10 * nR + 3 * nC - 20 * nP, trans = nR, bus = -24 * nI - 37 * nC - 32 * nP - 22 * nO;
    int un = 10 * nR - 3 * nI - 3 * nC - 4 * nO;
    if (un > 0) bus -= 8 * un;
    for (int r = 0; r < 7; r++)
        for (int c = 0; c < 7; c++) {
            int a = g[r * 7 + c];
            if (a == 0 && c == 0) trans -= 1;
            if (a == 2) { if (c > 0) bus -= 2; if (r < 6) bus -= 5; }
            if (c < 6) {
                int b = g[r * 7 + c + 1];
                if (a == 3 && b == 0) { well += 20; tax += 8; }
                else if (a == 3 && b == 3) well += 20;
                else if (a == 0 && b == 3) tax += 3;
                else if ((a == 0 && b == 1) || (a == 1 && b == 0)) tax -= 2;
                else if (a == 1 && b == 1) { trans += 3; bus += 10; }
                else if (a == 2 && b == 2) trans += 3;
            }
            if (r < 6) {
                int b = g[(r + 1) * 7 + c];
                if (a == 0 && b == 3) { well += 15; tax += 10; }
                else if (a == 3 && b == 3) well += 20;
                else if (a == 0 && b == 1) { well += 4; tax -= 2; }
                else if (a == 1 && b == 0) tax -= 2;
                else if (a == 3 && b == 0) tax += 3;
                else if (a == 1 && b == 1) { trans += 3; bus += 10; }
                else if (a == 2 && b == 2) trans += 3;
            }
        }
    if (tax < THR[1] || bus < THR[3]) { raw[0] = -100000; raw[1] = tax; raw[2] = -100000; raw[3] = bus; return true; }   // cheap early exit (invalid anyway)
    int lst[4][L], ln[4] = {0, 0, 0, 0};
    for (int i = 0; i < L; i++) if (g[i] < 4) lst[g[i]][ln[g[i]]++] = i;
    auto nearest = [&](int i, int t) { int m = 99; const int *row = MAN[i]; for (int k = 0; k < ln[t]; k++) { int d = row[lst[t][k]]; if (d < m) m = d; } return m; };
    for (int i = 0; i < L; i++) {
        switch (g[i]) {
            case 0: { int dC = nearest(i, 2), dP = nearest(i, 3); well += -3 * dC * dC - dP * dP; trans -= 2 * dC; break; }
            case 1: { int dR = nearest(i, 0), dC = nearest(i, 2); trans -= dR + 5 * dC; break; }
            case 2: { int dI = nearest(i, 1); trans -= 5 * dI; break; }
            case 3: { int dI = nearest(i, 1); well += 2 * dI; break; }
            default: { int dR = nearest(i, 0); trans -= dR; }
        }
    }
    raw[0] = well; raw[1] = tax; raw[2] = trans; raw[3] = bus;
    return true;
}
static inline bool valid_grid(const uint8_t *g) { int r[4]; return raw4(g, r) && r[0] >= THR[0] && r[1] >= THR[1] && r[2] >= THR[2] && r[3] >= THR[3]; }
static inline int top_sum(const int *h, int need) { int s = 0; for (int d = 0; d < NB && need > 0; d++) { int c = h[d]; if (c >= need) { s += need * d; need = 0; } else { s += c * d; need -= c; } } return s; }

struct Cand { int p, q; uint8_t vp, vq; float gain, geff, key; };

struct Chain {
    uint8_t G[N][L], GT[L][N], bestG[N][L], tmpg[L];
    int D[N][N], hist[N][NB], S9[N], t10[N], nd[N];
    long long cur = 0, best = -1, acc = 0;
    Rng rng;
    std::vector<Cand> cands; std::vector<int> ord;

    long long full_score() const { long long tot = 0; for (int i = 0; i < N; i++) { int h[NB] = {0}; for (int j = 0; j < N; j++) if (j != i) h[D[i][j]]++; tot += top_sum(h, K); } return tot; }
    void load(const uint8_t *g) {
        memcpy(G, g, N * L);
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { int d = 0; for (int p = 0; p < L; p++) d += G[i][p] != G[j][p]; D[i][j] = d; }
        for (int i = 0; i < N; i++) for (int p = 0; p < L; p++) GT[p][i] = G[i][p];
        cur = full_score(); best = cur; memcpy(bestG, G, sizeof(G));
    }
    void gen(int i, bool allsw) {
        for (int j = 0; j < N; j++) { int *h = hist[j]; memset(h, 0, sizeof(int) * NB); for (int k = 0; k < N; k++) if (k != j && k != i) h[D[j][k]]++; }
        for (int j = 0; j < N; j++) { if (j == i) { S9[j] = 0; t10[j] = 0; continue; } const int *h = hist[j]; int need = K - 1, s = 0, d = 0; for (; d < NB; d++) { int c = h[d]; if (c > need) break; s += c * d; need -= c; } s += need * d; S9[j] = s; t10[j] = d; }
        long long base = 0; for (int j = 0; j < N; j++) base += S9[j];
        double pen_cur = 0; if (P_lam > 0) for (int j = 0; j < N; j++) if (j != i && D[i][j] < P_dmin) pen_cur += P_dmin - D[i][j];
        cands.clear();
        const int *Di = D[i];
        auto eval = [&](Cand &c) {
            int tot = 0; for (int j = 0; j < N; j++) { int x = nd[j], t = t10[j]; tot += x < t ? x : t; }          // nd[i] is arbitrary, t10[i] = 0 -> adds min(x,0) = 0 when x >= 0
            int h[NB] = {0}; for (int j = 0; j < N; j++) h[nd[j]]++; h[nd[i]]--;
            double pen = 0; if (P_lam > 0) for (int j = 0; j < N; j++) if (j != i && nd[j] < P_dmin) pen += P_dmin - nd[j];
            c.gain = (float)(base + tot + top_sum(h, K) - cur); c.geff = (float)(c.gain - P_lam * (pen - pen_cur));
        };
        const uint8_t *gi = G[i];
        for (int p = 0; p < L; p++) {
            uint8_t a = gi[p]; const uint8_t *col = GT[p];
            for (int b = 0; b < 5; b++) { if (b == a) continue; for (int j = 0; j < N; j++) nd[j] = Di[j] + (col[j] == a) - (col[j] == b); nd[i] = 0; Cand c; c.p = p; c.q = -1; c.vp = (uint8_t)b; c.vq = 0; eval(c); cands.push_back(c); }
        }
        auto add_swap = [&](int p, int q) {
            uint8_t a = gi[p], b = gi[q]; const uint8_t *cp = GT[p], *cq = GT[q];
            for (int j = 0; j < N; j++) nd[j] = Di[j] + (cp[j] == a) - (cp[j] == b) + (cq[j] == b) - (cq[j] == a);
            nd[i] = 0; Cand c; c.p = p; c.q = q; c.vp = b; c.vq = a; eval(c); cands.push_back(c);
        };
        if (allsw) { for (int p = 0; p < L; p++) for (int q = p + 1; q < L; q++) if (gi[p] != gi[q]) add_swap(p, q); }
        else { int made = 0; for (int t = 0; t < 3 * P_nswap && made < P_nswap; t++) { int p = rng.below(L), q = rng.below(L); if (gi[p] != gi[q]) { add_swap(p, q); made++; } } }
    }
    void make(int i, const Cand &c) { memcpy(tmpg, G[i], L); tmpg[c.p] = c.vp; if (c.q >= 0) tmpg[c.q] = c.vq; }
    void apply(int i, const Cand &c) {
        make(i, c);
        for (int j = 0; j < N; j++) { if (j == i) continue; int d = 0; for (int p = 0; p < L; p++) d += tmpg[p] != G[j][p]; D[i][j] = D[j][i] = d; }
        memcpy(G[i], tmpg, L); GT[c.p][i] = tmpg[c.p]; if (c.q >= 0) GT[c.q][i] = tmpg[c.q];
        cur += (long long)c.gain;
    }
    void run(long long nsteps, double T) {
        bool allsw = P_nswap < 0;
        for (long long s = 0; s < nsteps; s++) {
            int contrib[N], cmin = 1 << 30; double w[N], ws = 0;
            for (int j = 0; j < N; j++) { int h[NB] = {0}; for (int k = 0; k < N; k++) h[D[j][k]]++; h[D[j][j]]--; contrib[j] = top_sum(h, K); if (contrib[j] < cmin) cmin = contrib[j]; }
            for (int j = 0; j < N; j++) { w[j] = std::exp(-(contrib[j] - cmin) / P_focus); ws += w[j]; }
            double u = rng.uni() * ws; int i = 0; for (; i < N - 1; i++) { u -= w[i]; if (u <= 0) break; }
            gen(i, allsw);
            int M = (int)cands.size();
            for (int k = 0; k < M; k++) cands[k].key = cands[k].geff + (float)(T * rng.gumbel());
            ord.resize(M); for (int k = 0; k < M; k++) ord[k] = k;
            int top = std::min(P_max_tries, M);
            std::partial_sort(ord.begin(), ord.begin() + top, ord.end(), [&](int a, int b) { return cands[a].key > cands[b].key; });
            double thresh = -T * rng.expo();
            for (int t = 0; t < top; t++) {
                const Cand &c = cands[ord[t]];
                if (c.geff < thresh) continue;
                make(i, c);
                if (!valid_grid(tmpg)) continue;
                apply(i, c); acc++;
                if (cur > best) { best = cur; memcpy(bestG, G, sizeof(G)); }
                break;
            }
        }
    }
};

int main(int argc, char **argv) {
    std::string init, state;
    int R = 12, block = 2000; long long rounds = 1000000000LL, log_rounds = 50, ckpt_rounds = 200;
    double tmin = 0.3, tmax = 1.3, hours = 1e9; uint64_t seed = 0; std::string temps;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--init") init = nx(); else if (a == "--state") state = nx(); else if (a == "--replicas") R = atoi(nx().c_str());
        else if (a == "--block") block = atoi(nx().c_str()); else if (a == "--rounds") rounds = atoll(nx().c_str()); else if (a == "--hours") hours = atof(nx().c_str());
        else if (a == "--tmin") tmin = atof(nx().c_str()); else if (a == "--tmax") tmax = atof(nx().c_str()); else if (a == "--temps") temps = nx();
        else if (a == "--focus") P_focus = atof(nx().c_str()); else if (a == "--lam") P_lam = atof(nx().c_str()); else if (a == "--dmin") P_dmin = atoi(nx().c_str());
        else if (a == "--nswap") P_nswap = atoi(nx().c_str()); else if (a == "--max-tries") P_max_tries = atoi(nx().c_str()); else if (a == "--seed") seed = strtoull(nx().c_str(), 0, 10);
        else if (a == "--log-rounds") log_rounds = atoll(nx().c_str()); else if (a == "--ckpt-rounds") ckpt_rounds = atoll(nx().c_str());
        else if (a == "--thr") { for (int k = 0; k < 4; k++) THR[k] = atoi(nx().c_str()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    for (int i = 0; i < L; i++) for (int j = 0; j < L; j++) MAN[i][j] = abs(i / 7 - j / 7) + abs(i % 7 - j % 7);
    std::vector<double> Tl(R);
    if (!temps.empty()) { Tl.clear(); size_t pos = 0; while (pos < temps.size()) { size_t e = temps.find(',', pos); if (e == std::string::npos) e = temps.size(); Tl.push_back(atof(temps.substr(pos, e - pos).c_str())); pos = e + 1; } R = (int)Tl.size(); }
    else for (int k = 0; k < R; k++) Tl[k] = R > 1 ? tmin * std::pow(tmax / tmin, (double)k / (R - 1)) : tmin;
    std::vector<std::unique_ptr<Chain>> ch(R);
    for (int k = 0; k < R; k++) { ch[k].reset(new Chain()); ch[k]->rng.seed(seed * 1000003ULL + k * 7919ULL + 17); }
    static uint8_t gbest[N][L], buf[N][L]; long long best = -1, round0 = 0; bool resumed = false;
    const long long MAGIC = 0x414e4e45414c31LL;
    { FILE *f = fopen(state.c_str(), "rb");
      if (f) { long long hd[4]; uint64_t dummy[4]; long long rr = 0;
          if (fread(hd, 8, 4, f) == 4 && hd[0] == MAGIC && fread(buf, 1, N * L, f) == (size_t)N * L && fread(gbest, 1, N * L, f) == (size_t)N * L && fread(dummy, 8, 4, f) == 4 && fread(&rr, 8, 1, f) == 1 && rr == R) {
              bool ok = true;
              for (int k = 0; k < R && ok; k++) { ok = fread(buf, 1, N * L, f) == (size_t)N * L && fread(ch[k]->rng.s, 8, 4, f) == 4; if (ok) ch[k]->load(&buf[0][0]); }
              if (ok) { round0 = hd[1]; best = hd[3]; resumed = true; }
          }
          fclose(f); } }
    if (!resumed) {
        FILE *f = fopen(init.c_str(), "rb"); if (!f || fread(buf, 1, N * L, f) != (size_t)N * L) { fprintf(stderr, "cannot read init\n"); return 3; } fclose(f);
        for (int i = 0; i < N; i++) if (!valid_grid(buf[i])) { fprintf(stderr, "init grid %d invalid (repair it with bin/anneal first)\n", i); return 4; }
        for (int k = 0; k < R; k++) ch[k]->load(&buf[0][0]);
        best = ch[0]->cur; memcpy(gbest, buf, sizeof(buf));
    }
    signal(SIGTERM, on_sig); signal(SIGUSR1, on_sig); signal(SIGINT, on_sig);
    time_t tstart = time(0);
    printf("[%s] %s at round %lld: %d replicas, T %.3f..%.3f, block %d, best %lld\n", state.c_str(), resumed ? "resumed" : "start", round0, R, Tl[0], Tl[R - 1], block, best); fflush(stdout);
    auto save = [&](long long round, bool done) {
        std::string tmp = state + ".tmp"; FILE *f = fopen(tmp.c_str(), "wb"); if (!f) return;
        long long hd[4] = {MAGIC, round, done ? 1 : 0, best}; uint64_t dummy[4] = {0, 0, 0, 0}; long long rr = R;
        fwrite(hd, 8, 4, f); fwrite(gbest, 1, N * L, f); fwrite(gbest, 1, N * L, f); fwrite(dummy, 8, 4, f); fwrite(&rr, 8, 1, f);
        for (int k = 0; k < R; k++) { fwrite(ch[k]->G, 1, N * L, f); fwrite(ch[k]->rng.s, 8, 4, f); }
        fclose(f); rename(tmp.c_str(), state.c_str());
    };
    Rng srng; srng.seed(seed * 31 + 5);
    std::vector<long long> sw_try(R, 0), sw_acc(R, 0); std::vector<double> ssum(R, 0); long long nacc_rounds = 0;
    long long round = round0;
    for (; round < rounds && !g_stop; round++) {
        if ((time(0) - tstart) / 3600.0 > hours) break;
        std::vector<std::thread> th;
        for (int k = 0; k < R; k++) th.emplace_back([&, k]() { ch[k]->run(block, Tl[k]); });
        for (auto &t : th) t.join();
        for (int k = 0; k < R; k++) { if (ch[k]->best > best) { best = ch[k]->best; memcpy(gbest, ch[k]->bestG, sizeof(gbest)); } ch[k]->best = ch[k]->cur; ssum[k] += ch[k]->cur; }
        nacc_rounds++;
        for (int k = (int)(round % 2); k + 1 < R; k += 2) {
            double dlt = (1.0 / Tl[k] - 1.0 / Tl[k + 1]) * (double)(ch[k + 1]->cur - ch[k]->cur);
            sw_try[k]++;
            if (dlt >= 0 || srng.uni() < std::exp(dlt)) { std::swap(ch[k], ch[k + 1]); sw_acc[k]++; }
        }
        if ((round + 1) % log_rounds == 0) {
            printf("round %lld (%.2f h, %lld steps/replica) best %lld (%.5f)\n   T:   ", round + 1, (time(0) - tstart) / 3600.0, (round + 1 - round0) * block, best, best / 49000.0);
            for (int k = 0; k < R; k++) printf(" %6.3f", Tl[k]);
            printf("\n   <S>: "); for (int k = 0; k < R; k++) printf(" %6.0f", ssum[k] / nacc_rounds);
            printf("\n   swap:"); for (int k = 0; k + 1 < R; k++) printf(" %6.2f", sw_try[k] ? (double)sw_acc[k] / sw_try[k] : 0.0);
            printf("\n"); fflush(stdout);
            std::fill(sw_try.begin(), sw_try.end(), 0); std::fill(sw_acc.begin(), sw_acc.end(), 0); std::fill(ssum.begin(), ssum.end(), 0.0); nacc_rounds = 0;
        }
        if ((round + 1) % ckpt_rounds == 0) save(round + 1, false);
    }
    for (int k = 0; k < R; k++) if (ch[k]->full_score() != ch[k]->cur) { fprintf(stderr, "score bookkeeping mismatch in replica %d\n", k); return 6; }
    bool done = !g_stop;
    save(round, done);
    printf("%s at round %lld: best %lld (%.10f) wall %lds\n", done ? "finished" : "stopped", round, best, best / 49000.0, (long)(time(0) - tstart)); fflush(stdout);
    return done ? 0 : 99;
}
