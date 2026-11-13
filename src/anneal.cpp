// Set annealer under exact advisor validity (C++ port of src/optimize_exact.py; rules as in src/rules.py).
// Objective: integer 10-NN Hamming score of 100 grids; every grid must stay valid for all four advisors.
// Build:  g++ -O3 -mavx2 -std=c++17 -o bin/anneal3 src/anneal.cpp   (bin/anneal, bin/anneal2 are earlier builds; src/anneal.py uses the newest)
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
static double RM[N];   // extra pair-penalty weight of each grid (--special / --boost), 0 for ordinary grids
static long long cur = 0, best = -1, step0 = 0;

static inline int top_sum(const int *h, int need) { int s = 0; for (int d = 0; d < NB && need > 0; d++) { int c = h[d]; if (c >= need) { s += need * d; need = 0; } else { s += c * d; need -= c; } } return s; }
static long long full_score() { long long tot = 0; for (int i = 0; i < N; i++) { int h[NB] = {0}; for (int j = 0; j < N; j++) if (j != i) h[D[i][j]]++; tot += top_sum(h, K); } return tot; }
static void build_D() { for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { int d = 0; for (int p = 0; p < L; p++) d += G[i][p] != G[j][p]; D[i][j] = d; } for (int i = 0; i < N; i++) for (int p = 0; p < L; p++) GT[p][i] = G[i][p]; }

struct Cand { int p, q; uint8_t vp, vq; double gain, geff, key; };

// ---- Probe of the algebraic construction (UPDATES.md 2026-11-12): the 125 affine functions over GF(5)^2 drawn on 49 cells ----
// State: site[c] in 0..24 (24 sites hold two cells, one holds one) and perm[c][v] = district shown by cell c for code value v.
// Codeword (al, be, ga) -> g[c] = perm[c][(al * a + be * b + ga) % 5] with (a, b) = (site[c] / 5, site[c] % 5).
// Any two of the 125 grids differ in >= 39 cells whatever site/perm are.  Simulated annealing over site/perm looks for a
// design that makes as many of them valid as possible: energy = sum of the `keep` smallest validity deficits.
// Output file: 125 x 49 bytes (grids of the best design), then 49 x (site, 5 districts).   Run: --crystal --state <out> ...
static int crystal_main(const std::string &out, long long steps, double t0, double t1, uint64_t seed, long long log_every, int keep, int pairing, const std::string &cinit, double cstep) {
    Rng rng; rng.seed(seed * 7919 + 777);
    int site[L]; uint8_t perm[L][5];
    if (pairing == 0) {            // 21 vertical dominoes in rows 0-5, three horizontal dominoes and one single cell in row 6
        int ns = 0;
        for (int c = 0; c < 7; c++) for (int r = 0; r < 6; r += 2) { site[r * 7 + c] = ns; site[(r + 1) * 7 + c] = ns; ns++; }
        for (int c = 0; c < 6; c += 2) { site[42 + c] = ns; site[43 + c] = ns; ns++; }
        site[48] = ns;
    } else {                       // random pairing of the cells
        int cells[L]; for (int i = 0; i < L; i++) cells[i] = i;
        for (int i = L - 1; i > 0; i--) std::swap(cells[i], cells[rng.below(i + 1)]);
        for (int i = 0; i < L; i++) site[cells[i]] = i / 2;
    }
    { int lab[25]; for (int i = 0; i < 25; i++) lab[i] = i; for (int i = 24; i > 0; i--) std::swap(lab[i], lab[rng.below(i + 1)]); for (int c = 0; c < L; c++) site[c] = lab[site[c]]; }
    for (int c = 0; c < L; c++) { for (int v = 0; v < 5; v++) perm[c][v] = (uint8_t)v; for (int v = 4; v > 0; v--) std::swap(perm[c][v], perm[c][rng.below(v + 1)]); }
    if (!cinit.empty()) {          // --cinit <earlier output file>: continue from that design instead of a random one
        FILE *fi = fopen(cinit.c_str(), "rb"); bool ok = fi && fseek(fi, 125L * L, SEEK_SET) == 0;
        for (int c = 0; ok && c < L; c++) { uint8_t s = 0; ok = fread(&s, 1, 1, fi) == 1 && fread(perm[c], 1, 5, fi) == 5; site[c] = s; }
        if (fi) fclose(fi);
        if (!ok) { fprintf(stderr, "cannot read design %s\n", cinit.c_str()); return 3; }
    }
    static uint8_t CW[125][L];
    double def[125], nd[125], srt[125];
    if (keep > 125) keep = 125;
    auto energy = [&](double *d, int *nv) {
        int v = 0;
        for (int f = 0; f < 125; f++) {
            int al = f / 25, be = (f / 5) % 5, ga = f % 5; uint8_t *g = CW[f];
            for (int c = 0; c < L; c++) { int s = site[c]; g[c] = perm[c][(al * (s / 5) + be * (s % 5) + ga) % 5]; }
            d[f] = deficit(g); if (d[f] > 0) d[f] += cstep; else v++;   // --cstep: fixed extra cost of every invalid grid
        }
        memcpy(srt, d, sizeof(double) * 125); std::sort(srt, srt + 125);
        double e = 0; for (int k = 0; k < keep; k++) e += srt[k];
        *nv = v; return e;
    };
    int V; double E = energy(def, &V), bestE = E; int bestV = V;
    int bsite[L]; uint8_t bperm[L][5]; memcpy(bsite, site, sizeof(site)); memcpy(bperm, perm, sizeof(perm));
    auto save = [&]() {            // writes the grids and the design of the best state so far
        int ssite[L]; uint8_t sperm[L][5]; memcpy(ssite, site, sizeof(site)); memcpy(sperm, perm, sizeof(perm));
        memcpy(site, bsite, sizeof(site)); memcpy(perm, bperm, sizeof(perm));
        int nv; energy(nd, &nv);
        std::string tmp = out + ".tmp"; FILE *fo = fopen(tmp.c_str(), "wb");
        if (fo) { fwrite(CW, 1, sizeof(CW), fo); for (int c = 0; c < L; c++) { uint8_t s = (uint8_t)site[c]; fwrite(&s, 1, 1, fo); fwrite(perm[c], 1, 5, fo); } fclose(fo); rename(tmp.c_str(), out.c_str()); }
        memcpy(site, ssite, sizeof(site)); memcpy(perm, sperm, sizeof(perm));
    };
    signal(SIGTERM, on_sig); signal(SIGUSR1, on_sig); signal(SIGINT, on_sig);
    long long acc = 0; clock_t c0 = clock();
    printf("[crystal %s] start: E %.2f valid %d / 125 (keep %d, pairing %d)\n", out.c_str(), E, V, keep, pairing); fflush(stdout);
    for (long long step = 1; step <= steps && !g_stop; step++) {
        double T = t0 * std::pow(t1 / t0, (double)step / steps);
        int kind = rng.below(10), c1 = rng.below(L), c2 = rng.below(L), u = rng.below(5), v = rng.below(5);
        int s1 = site[c1], s2 = site[c2];
        if ((kind <= 4 && u == v) || ((kind == 5 || kind == 6 || kind == 9) && s1 == s2) || ((kind == 7 || kind == 8) && c1 == c2)) continue;
        // 0-2: swap two districts in one cell; 3-4: the same value swap in every cell of one site; 5-6: exchange two sites;
        // 7-8: exchange the content of two cells; 9: exchange the sites of two cells.  Every move is its own inverse.
        auto doit = [&]() {
            if (kind <= 2) std::swap(perm[c1][u], perm[c1][v]);
            else if (kind <= 4) { for (int c = 0; c < L; c++) if (site[c] == s1) std::swap(perm[c][u], perm[c][v]); }
            else if (kind <= 6) { for (int c = 0; c < L; c++) { if (site[c] == s1) site[c] = s2; else if (site[c] == s2) site[c] = s1; } }
            else if (kind <= 8) { std::swap(site[c1], site[c2]); for (int x = 0; x < 5; x++) std::swap(perm[c1][x], perm[c2][x]); }
            else std::swap(site[c1], site[c2]);
        };
        doit();
        int nv; double e = energy(nd, &nv);
        if (e <= E || rng.uni() < std::exp((E - e) / T)) {
            E = e; V = nv; acc++; memcpy(def, nd, sizeof(def));
            if (V > bestV || (V == bestV && E < bestE)) { bestV = V; bestE = E; memcpy(bsite, site, sizeof(site)); memcpy(bperm, perm, sizeof(perm)); }
        } else doit();
        if (step % log_every == 0) {
            printf("step %lld T %.3f E %.2f valid %d best %d (E %.2f) acc %lld  %.0f moves/s\n", step, T, E, V, bestV, bestE, acc, step / ((double)(clock() - c0) / CLOCKS_PER_SEC + 1e-9)); fflush(stdout);
            save();
        }
    }
    save();
    memcpy(site, bsite, sizeof(site)); memcpy(perm, bperm, sizeof(perm));
    int nv; energy(nd, &nv);
    int fail[4] = {0, 0, 0, 0}, parks[L + 1]; memset(parks, 0, sizeof(parks));
    for (int f = 0; f < 125; f++) { int r[4]; if (raw4(CW[f], r)) for (int a = 0; a < 4; a++) fail[a] += r[a] < THR[a]; int np = 0; for (int c = 0; c < L; c++) np += CW[f][c] == 3; parks[np]++; }
    printf("finished: best valid %d / 125 (E %.2f); failing Wellness %d Tax %d Transportation %d Business %d; parks per grid:", nv, bestE, fail[0], fail[1], fail[2], fail[3]);
    for (int k = 0; k <= L; k++) if (parks[k]) printf(" %d:%d", k, parks[k]);
    printf("\n"); fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    std::string init, state, mode = "run";
    long long steps = 1000000, log_every = 100000, ckpt_every = 500000;
    double t0 = 1.0, t1 = 0.05, focus = 15.0, lam = 0.0, lam2 = 0.0, lam_end = -1.0, lam2_end = -1.0;
    int nswap = 200, dmin = 38, dmin2 = 40, max_tries = 128, cycles = 1, special = 0;
    double boost = 0.0;   // --special M --boost B: pair penalties involving one of the first M grids are multiplied by 1+B (1+2B if both)
    int keep = 100, pairing = 0; std::string cinit; double cstep = 0.0;   // --crystal mode only (see crystal_main)
    int frozen = 0;       // --frozen M: the first M grids are never changed (they must be valid); build as bin/anneal6
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--init") init = nx(); else if (a == "--state") state = nx(); else if (a == "--steps") steps = atoll(nx().c_str());
        else if (a == "--t0") t0 = atof(nx().c_str()); else if (a == "--t1") t1 = atof(nx().c_str()); else if (a == "--focus") focus = atof(nx().c_str());
        else if (a == "--lam") lam = atof(nx().c_str()); else if (a == "--dmin") dmin = atoi(nx().c_str());
        else if (a == "--lam2") lam2 = atof(nx().c_str()); else if (a == "--dmin2") dmin2 = atoi(nx().c_str());
        else if (a == "--lam-end") lam_end = atof(nx().c_str()); else if (a == "--lam2-end") lam2_end = atof(nx().c_str()); else if (a == "--nswap") nswap = atoi(nx().c_str());
        else if (a == "--max-tries") max_tries = atoi(nx().c_str()); else if (a == "--seed") seed = strtoull(nx().c_str(), 0, 10); else if (a == "--cycles") cycles = atoi(nx().c_str());
        else if (a == "--log-every") log_every = atoll(nx().c_str()); else if (a == "--ckpt-every") ckpt_every = atoll(nx().c_str());
        else if (a == "--thr") { for (int k = 0; k < 4; k++) THR[k] = atoi(nx().c_str()); }
        else if (a == "--score") { mode = "score"; init = nx(); }
        else if (a == "--special") special = atoi(nx().c_str()); else if (a == "--boost") boost = atof(nx().c_str());
        else if (a == "--crystal") mode = "crystal"; else if (a == "--keep") keep = atoi(nx().c_str()); else if (a == "--pairing") pairing = atoi(nx().c_str());
        else if (a == "--frozen") frozen = atoi(nx().c_str()); else if (a == "--cinit") cinit = nx(); else if (a == "--cstep") cstep = atof(nx().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    for (int i = 0; i < L; i++) for (int j = 0; j < L; j++) MAN[i][j] = abs(i / 7 - j / 7) + abs(i % 7 - j % 7);
    if (mode == "score") {                     // print raw scores of every 49-byte grid in the file (cross-check against Python)
        FILE *f = fopen(init.c_str(), "rb"); if (!f) return 3; uint8_t g[L];
        while (fread(g, 1, L, f) == L) { int r[4] = {0, 0, 0, 0}; bool ok = raw4(g, r); printf("%d %d %d %d %d\n", ok ? 1 : 0, r[0], r[1], r[2], r[3]); }
        fclose(f); return 0;
    }
    if (mode == "crystal") return crystal_main(state, steps, t0, t1, seed, log_every, keep, pairing, cinit, cstep);
    double W[NB]; bool usepen = lam > 0 || lam2 > 0;
    if (lam_end < 0) lam_end = lam; if (lam2_end < 0) lam2_end = lam2;
    auto setW = [&](double frac) { double l1 = lam + (lam_end - lam) * frac, l2 = lam2 + (lam2_end - lam2) * frac; for (int d = 0; d < NB; d++) W[d] = l1 * std::max(0, dmin - d) + l2 * std::max(0, dmin2 - d); };
    setW(0.0);
    if (special > N) special = N;
    if (frozen < 0) frozen = 0; if (frozen > N - 1) frozen = N - 1;
    for (int j = 0; j < N; j++) RM[j] = j < special ? boost : 0.0;
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
        double pen_cur = 0; if (usepen) for (int j = 0; j < N; j++) if (j != i) pen_cur += (1.0 + RM[i] + RM[j]) * W[D[i][j]];
        cands.clear();
        auto eval = [&](Cand &c) {
            int tot = 0; for (int j = 0; j < N; j++) { int x = nd[j], t = t10[j]; tot += x < t ? x : t; }
            int h[NB] = {0}; for (int j = 0; j < N; j++) h[nd[j]]++; h[nd[i]]--;
            double pen = 0;
            if (usepen) {
                if (special > 0) { double ri = 1.0 + RM[i]; for (int j = 0; j < N; j++) if (j != i) pen += (ri + RM[j]) * W[nd[j]]; }
                else for (int d = 0; d < NB; d++) pen += h[d] * W[d];
            }
            c.gain = (double)(base + tot + top_sum(h, K) - cur); c.geff = c.gain - (pen - pen_cur);
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

    for (int j = 0; j < frozen; j++) if (!valid_grid(G[j])) { fprintf(stderr, "frozen grid %d is invalid\n", j); return 7; }
    // repair phase: make every grid valid (frozen grids are valid already, so they are never touched)
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
        if (usepen && (step & 1023) == 0) setW((double)step / steps);
        if (step % log_every == 0 && step > step0) {
            int p38 = 0, p37 = 0, mn = 99; for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { int d = D[i][j]; p38 += d < 38; p37 += d < 37; if (d < mn) mn = d; }
            if (full_score() != cur) { fprintf(stderr, "score bookkeeping mismatch\n"); return 6; }
            if (special > 0) {   // how many neighbours at distance <= 38 the special grids still have (fewer than 10 raises the score)
                int ks = 0, kmin = 999, nlt = 0; long long cs = 0;
                for (int a = 0; a < special; a++) { int k = 0, h[NB] = {0}; for (int j = 0; j < N; j++) if (j != a) { k += D[a][j] <= 38; h[D[a][j]]++; } cs += top_sum(h, K); ks += k; if (k < kmin) kmin = k; nlt += k < 10; }
                printf("special rows: mean %.1f min %d neighbours at <=38, %d of %d below 10, mean contribution %.2f (other rows %.2f)\n", (double)ks / special, kmin, nlt, special, (double)cs / special, special < N ? (double)(cur - cs) / (N - special) : 0.0);
            }
            printf("step %lld T %.3f score %lld (%.5f) best %lld (%.5f) acc %lld pairs<38 %d <37 %d min %d  %.0f steps/s\n", step, T, cur, cur / 49000.0, best, best / 49000.0, acc, p38, p37, mn,
                   (step - step0) / ((double)(clock() - c0) / CLOCKS_PER_SEC + 1e-9)); fflush(stdout);
        }
        if (step % ckpt_every == 0 && step > step0) save(step, false);
        // choose a grid, favouring low contributors
        int contrib[N], cmin = 1 << 30; double w[N], ws = 0;
        for (int j = frozen; j < N; j++) { int h[NB] = {0}; for (int k = 0; k < N; k++) if (k != j) h[D[j][k]]++; contrib[j] = top_sum(h, K); if (contrib[j] < cmin) cmin = contrib[j]; }
        for (int j = frozen; j < N; j++) { w[j] = std::exp(-(contrib[j] - cmin) / focus); ws += w[j]; }
        double u = rng.uni() * ws; int i = frozen; for (; i < N - 1; i++) { u -= w[i]; if (u <= 0) break; }
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
