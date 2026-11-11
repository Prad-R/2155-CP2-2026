// Set annealer under exact advisor validity (C++ port of src/optimize_exact.py; rules as in src/rules.py).
// Objective: integer 10-NN Hamming score of 100 grids; every grid must stay valid for all four advisors.
// Build:  g++ -O3 -mavx2 -std=c++17 -o bin/anneal2 src/anneal.cpp   (bin/anneal is the first, slower build)
// Run through src/anneal.py (prepares init.bin, forwards signals, verifies the result with the Python rules).
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

static const int N = 100, L = 49, K = 10, NB = 51;
static int MAN[L][L];
static int THR[4] = {70, 17, -187, -1155};
static const double SC[4] = {78.0, 56.0, 32.0, 160.0};
static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

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

// raw scores [Wellness, Tax, Transportation, Business]; returns false when a required district type is missing
static bool raw4(const uint8_t *g, int *raw, bool early = false) {
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
    if (early && (tax < THR[1] || bus < THR[3])) { raw[0] = -100000; raw[1] = tax; raw[2] = -100000; raw[3] = bus; return true; }
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
static inline bool valid_grid(const uint8_t *g) { int r[4]; return raw4(g, r, true) && r[0] >= THR[0] && r[1] >= THR[1] && r[2] >= THR[2] && r[3] >= THR[3]; }
static double deficit(const uint8_t *g) { int r[4]; if (!raw4(g, r)) return 50.0; double d = 0; for (int a = 0; a < 4; a++) if (r[a] < THR[a]) d += (THR[a] - r[a]) / SC[a]; return d; }

static uint8_t G[N][L], GT[L][N], bestG[N][L];
static int D[N][N];
static long long cur = 0, best = -1, step0 = 0;

static inline int top_sum(const int *h, int need) { int s = 0; for (int d = 0; d < NB && need > 0; d++) { int c = h[d]; if (c >= need) { s += need * d; need = 0; } else { s += c * d; need -= c; } } return s; }
static long long full_score() { long long tot = 0; for (int i = 0; i < N; i++) { int h[NB] = {0}; for (int j = 0; j < N; j++) if (j != i) h[D[i][j]]++; tot += top_sum(h, K); } return tot; }
static void build_D() { for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { int d = 0; for (int p = 0; p < L; p++) d += G[i][p] != G[j][p]; D[i][j] = d; } for (int i = 0; i < N; i++) for (int p = 0; p < L; p++) GT[p][i] = G[i][p]; }

struct Cand { int p, q; uint8_t vp, vq; double gain, geff, key; };

int main(int argc, char **argv) {
    std::string init, state, mode = "run";
    long long steps = 1000000, log_every = 100000, ckpt_every = 500000;
    double t0 = 1.0, t1 = 0.05, focus = 15.0, lam = 0.0;
    int nswap = 200, dmin = 38, max_tries = 128, cycles = 1;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--init") init = nx(); else if (a == "--state") state = nx(); else if (a == "--steps") steps = atoll(nx().c_str());
        else if (a == "--t0") t0 = atof(nx().c_str()); else if (a == "--t1") t1 = atof(nx().c_str()); else if (a == "--focus") focus = atof(nx().c_str());
        else if (a == "--lam") lam = atof(nx().c_str()); else if (a == "--dmin") dmin = atoi(nx().c_str()); else if (a == "--nswap") nswap = atoi(nx().c_str());
        else if (a == "--max-tries") max_tries = atoi(nx().c_str()); else if (a == "--seed") seed = strtoull(nx().c_str(), 0, 10); else if (a == "--cycles") cycles = atoi(nx().c_str());
        else if (a == "--log-every") log_every = atoll(nx().c_str()); else if (a == "--ckpt-every") ckpt_every = atoll(nx().c_str());
        else if (a == "--thr") { for (int k = 0; k < 4; k++) THR[k] = atoi(nx().c_str()); }
        else if (a == "--score") { mode = "score"; init = nx(); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    for (int i = 0; i < L; i++) for (int j = 0; j < L; j++) MAN[i][j] = abs(i / 7 - j / 7) + abs(i % 7 - j % 7);
    if (mode == "score") {                     // print raw scores of every 49-byte grid in the file (cross-check against Python)
        FILE *f = fopen(init.c_str(), "rb"); if (!f) return 3; uint8_t g[L];
        while (fread(g, 1, L, f) == L) { int r[4] = {0, 0, 0, 0}; bool ok = raw4(g, r); printf("%d %d %d %d %d\n", ok ? 1 : 0, r[0], r[1], r[2], r[3]); }
        fclose(f); return 0;
    }
    Rng rng; rng.seed(seed * 7919 + 12345);
    bool resumed = false;
    { FILE *f = fopen(state.c_str(), "rb");
      if (f) { long long hd[4]; if (fread(hd, 8, 4, f) == 4 && hd[0] == 0x414e4e45414c31LL && fread(G, 1, N * L, f) == (size_t)N * L && fread(bestG, 1, N * L, f) == (size_t)N * L && fread(rng.s, 8, 4, f) == 4) { step0 = hd[1]; best = hd[3]; resumed = true; } fclose(f); } }
    if (!resumed) { FILE *f = fopen(init.c_str(), "rb"); if (!f || fread(G, 1, N * L, f) != (size_t)N * L) { fprintf(stderr, "cannot read init\n"); return 3; } fclose(f); }
    build_D(); cur = full_score();
    signal(SIGTERM, on_sig); signal(SIGUSR1, on_sig); signal(SIGINT, on_sig);
    time_t tstart = time(0);
    printf("[%s] %s at step %lld: score %lld (%.5f) best %lld\n", state.c_str(), resumed ? "resumed" : "start", step0, cur, cur / 49000.0, best); fflush(stdout);

    auto save = [&](long long step, bool done) {
        std::string tmp = state + ".tmp"; FILE *f = fopen(tmp.c_str(), "wb"); if (!f) return;
        long long hd[4] = {0x414e4e45414c31LL, step, done ? 1 : 0, best};
        fwrite(hd, 8, 4, f); fwrite(G, 1, N * L, f); fwrite(bestG, 1, N * L, f); fwrite(rng.s, 8, 4, f); fclose(f); rename(tmp.c_str(), state.c_str());
    };
    std::vector<Cand> cands; cands.reserve(196 + 1176);
    std::vector<int> ord;
    static int hist[N][NB];
    int S9[N], t10[N], nd[N];
    uint8_t tmpg[L];

    // generate all candidates of grid i with their exact gains (cur must be consistent)
    auto gen = [&](int i, bool allsw) {
        for (int j = 0; j < N; j++) { int *h = hist[j]; memset(h, 0, sizeof(int) * NB); for (int k = 0; k < N; k++) if (k != j && k != i) h[D[j][k]]++; }
        for (int j = 0; j < N; j++) { if (j == i) { S9[j] = 0; t10[j] = 0; continue; } const int *h = hist[j]; int need = K - 1, s = 0, d = 0; for (; d < NB; d++) { int c = h[d]; if (c > need) break; s += c * d; need -= c; } s += need * d; S9[j] = s; t10[j] = d; }
        long long base = 0; for (int j = 0; j < N; j++) base += S9[j];
        double pen_cur = 0; if (lam > 0) for (int j = 0; j < N; j++) if (j != i && D[i][j] < dmin) pen_cur += dmin - D[i][j];
        cands.clear();
        auto eval = [&](Cand &c) {
            int tot = 0; for (int j = 0; j < N; j++) { int x = nd[j], t = t10[j]; tot += x < t ? x : t; }
            int h[NB] = {0}; for (int j = 0; j < N; j++) h[nd[j]]++; h[nd[i]]--;
            double pen = 0; if (lam > 0) for (int j = 0; j < N; j++) if (j != i && nd[j] < dmin) pen += dmin - nd[j];
            c.gain = (double)(base + tot + top_sum(h, K) - cur); c.geff = c.gain - lam * (pen - pen_cur);
        };
        const uint8_t *gi = G[i];
        for (int p = 0; p < L; p++) {
            uint8_t a = gi[p]; const uint8_t *col = GT[p];
            for (int b = 0; b < 5; b++) { if (b == a) continue; for (int j = 0; j < N; j++) nd[j] = D[i][j] + (col[j] == a) - (col[j] == b); nd[i] = 0; Cand c; c.p = p; c.q = -1; c.vp = (uint8_t)b; c.vq = 0; eval(c); cands.push_back(c); }
        }
        auto add_swap = [&](int p, int q) {
            uint8_t a = gi[p], b = gi[q]; const uint8_t *cp = GT[p], *cq = GT[q];
            for (int j = 0; j < N; j++) nd[j] = D[i][j] + (cp[j] == a) - (cp[j] == b) + (cq[j] == b) - (cq[j] == a);
            nd[i] = 0; Cand c; c.p = p; c.q = q; c.vp = b; c.vq = a; eval(c); cands.push_back(c);
        };
        if (allsw) { for (int p = 0; p < L; p++) for (int q = p + 1; q < L; q++) if (gi[p] != gi[q]) add_swap(p, q); }
        else { int made = 0; for (int t = 0; t < 3 * nswap && made < nswap; t++) { int p = rng.below(L), q = rng.below(L); if (gi[p] != gi[q]) { add_swap(p, q); made++; } } }
    };
    auto make = [&](int i, const Cand &c) { memcpy(tmpg, G[i], L); tmpg[c.p] = c.vp; if (c.q >= 0) tmpg[c.q] = c.vq; };
    auto apply = [&](int i, const Cand &c) {
        make(i, c);
        for (int j = 0; j < N; j++) { if (j == i) continue; int d = 0; for (int p = 0; p < L; p++) d += tmpg[p] != G[j][p]; D[i][j] = D[j][i] = d; }
        memcpy(G[i], tmpg, L); GT[c.p][i] = tmpg[c.p]; if (c.q >= 0) GT[c.q][i] = tmpg[c.q];
        cur += (long long)c.gain;
    };

    // repair phase: make every grid valid
    { double def[N]; int ninv = 0; for (int i = 0; i < N; i++) { def[i] = deficit(G[i]); ninv += def[i] > 0; }
      long long rep = 0;
      while (ninv > 0 && rep < 200000) {
          int pick = rng.below(ninv), i = -1; for (int j = 0; j < N; j++) if (def[j] > 0 && pick-- == 0) { i = j; break; }
          gen(i, false);
          double bu = -1e300, bd = 0; int bk = -1;
          for (size_t k = 0; k < cands.size(); k++) { make(i, cands[k]); double dc = deficit(tmpg); double u = cands[k].gain - 2000.0 * (dc - def[i]) + 0.5 * rng.gumbel(); if (u > bu) { bu = u; bk = (int)k; bd = dc; } }
          apply(i, cands[bk]); def[i] = bd; rep++;
          ninv = 0; for (int j = 0; j < N; j++) ninv += def[j] > 0;
      }
      if (rep) { printf("repair: %lld moves, invalid left %d, score %lld (%.5f)\n", rep, ninv, cur, cur / 49000.0); fflush(stdout); }
      if (ninv > 0) { fprintf(stderr, "could not repair the start set\n"); return 4; }
    }
    for (int i = 0; i < N; i++) if (!valid_grid(G[i])) { fprintf(stderr, "invalid grid %d in state\n", i); return 5; }
    if (best < 0) { best = cur; memcpy(bestG, G, sizeof(G)); }

    long long acc = 0, step = step0; clock_t c0 = clock();
    bool allsw = nswap < 0;
    long long cyc_len = steps / (cycles > 0 ? cycles : 1);
    for (; step < steps && !g_stop; step++) {
        double frac = cycles > 1 ? (double)(step % cyc_len) / cyc_len : (double)step / steps;
        double T = t0 * std::pow(t1 / t0, frac);
        if (step % log_every == 0 && step > step0) {
            int p38 = 0, p37 = 0, mn = 99; for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { int d = D[i][j]; p38 += d < 38; p37 += d < 37; if (d < mn) mn = d; }
            if (full_score() != cur) { fprintf(stderr, "score bookkeeping mismatch\n"); return 6; }
            printf("step %lld T %.3f score %lld (%.5f) best %lld (%.5f) acc %lld pairs<38 %d <37 %d min %d  %.0f steps/s\n", step, T, cur, cur / 49000.0, best, best / 49000.0, acc, p38, p37, mn,
                   (step - step0) / ((double)(clock() - c0) / CLOCKS_PER_SEC + 1e-9)); fflush(stdout);
        }
        if (step % ckpt_every == 0 && step > step0) save(step, false);
        // choose a grid, favouring low contributors
        int contrib[N], cmin = 1 << 30; double w[N], ws = 0;
        for (int j = 0; j < N; j++) { int h[NB] = {0}; for (int k = 0; k < N; k++) if (k != j) h[D[j][k]]++; contrib[j] = top_sum(h, K); if (contrib[j] < cmin) cmin = contrib[j]; }
        for (int j = 0; j < N; j++) { w[j] = std::exp(-(contrib[j] - cmin) / focus); ws += w[j]; }
        double u = rng.uni() * ws; int i = 0; for (; i < N - 1; i++) { u -= w[i]; if (u <= 0) break; }
        gen(i, allsw);
        int M = (int)cands.size();
        for (int k = 0; k < M; k++) cands[k].key = cands[k].geff + T * rng.gumbel();
        ord.resize(M); for (int k = 0; k < M; k++) ord[k] = k;
        int top = std::min(max_tries, M);
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
    bool done = step >= steps;
    if (full_score() != cur) { fprintf(stderr, "final score bookkeeping mismatch\n"); return 6; }
    save(step, done);
    printf("%s at step %lld: score %lld best %lld (%.10f) wall %lds\n", done ? "finished" : "stopped", step, cur, best, best / 49000.0, (long)(time(0) - tstart)); fflush(stdout);
    return done ? 0 : 99;
}
