// ============================================================================
// pt.cpp — a small path tracer with next-event estimation (NEE) for direct light
//
// Scene
//   * perfectly specular (mirror) sphere, radius 1, centered at the origin
//   * diffuse box, |x|,|y|,|z| <= 2; the front wall (z = +2) is left open
//   * a 1.6 m x 1.6 m quad area light on the ceiling (y = +2), pointing down
//   * pinhole camera at (0, 0, +5), 48 deg FOV, looking through the opening
//
// Renderer
//   * unbiased path tracing, Russian-roulette termination, max depth 16
//   * diffuse (Lambertian) surfaces: direct lighting via NEE — a point is
//     sampled uniformly on the area light and visibility-tested — plus a
//     cosine-weighted indirect bounce (the albedo/pi cancels the cos/pdf)
//   * mirror surfaces: a delta BSDF needs no NEE; the deterministic reflected
//     ray simply lands on the light quad (sharp reflection) or not
//
// Output: output.ppm — 512x512, 8-bit sRGB, Reinhard tone mapping.
//
// Build:  g++ -O2 -std=c++17 -pthread pt.cpp -o pt
// Usage:  ./pt [spp] [threads]       (defaults: 128 spp, all hardware threads)
// ============================================================================

#define _USE_MATH_DEFINES   // MSVC: expose M_PI in <cmath>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal vector math
// ---------------------------------------------------------------------------
struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double len() const { return std::sqrt(dot(*this)); }
    Vec3 norm() const { return *this / len(); }
};

static inline Vec3 reflect(const Vec3& d, const Vec3& n) { return d - n * (2.0 * d.dot(n)); }
static inline double clamp01(double v) { return std::min(1.0, std::max(0.0, v)); }

// Fast per-thread RNG: 53-bit uniforms straight from mt19937_64.
struct RNG {
    std::mt19937_64 g;
    explicit RNG(unsigned long long seed) : g(seed) {}
    double u() { return (g() >> 11) * (1.0 / 9007199254740992.0); }  // 2^53
};

// ---------------------------------------------------------------------------
// Geometry: one sphere + axis-aligned quad patches, all in one Scene
// ---------------------------------------------------------------------------
struct Ray { Vec3 o, d; };
struct Hit { double t = -1; Vec3 p, n; int id = -1; };   // t < 0 means miss

enum { MAT_DIFFUSE, MAT_MIRROR, MAT_LIGHT };
struct Material { int kind; Vec3 albedo, emit; };

struct Sphere { Vec3 c; double r; int id; };
// Quad = { c + u*a + v*b : |a| <= hu, |b| <= hv }, with unit normal n.
struct Quad { Vec3 c, n, u, v; double hu, hv; int id; };

struct Scene {
    std::vector<Material> mats;
    std::vector<Sphere> spheres;
    std::vector<Quad> quads;
    int light = -1;                    // index of the (only) emitting quad

    // Nearest hit in [tmin, tmax]; returns t < 0 on a miss.
    Hit intersect(const Ray& r, double tmin, double tmax) const {
        Hit best{tmax, Vec3{}, Vec3{}, -1};
        for (const Sphere& s : spheres) {
            Vec3 oc = r.o - s.c;
            double b = oc.dot(r.d);                    // dir is unit length
            double c = oc.dot(oc) - s.r * s.r;
            double disc = b * b - c;
            if (disc < 0) continue;
            double t = -b - std::sqrt(disc);
            if (t < tmin || t > best.t) continue;
            Vec3 p = r.o + r.d * t;
            best = {t, p, (p - s.c).norm(), s.id};
        }
        for (const Quad& q : quads) {
            double dn = r.d.dot(q.n);
            if (std::fabs(dn) < 1e-12) continue;       // parallel to the plane
            double t = (q.c - r.o).dot(q.n) / dn;
            if (t < tmin || t > best.t) continue;
            Vec3 p = r.o + r.d * t, w = p - q.c;
            if (std::fabs(w.dot(q.u)) > q.hu || std::fabs(w.dot(q.v)) > q.hv) continue;
            best = {t, p, q.n, q.id};
        }
        return best;
    }
};

static Scene build_scene() {
    Scene sc;
    auto mat = [&](int kind, Vec3 albedo, Vec3 emit) {
        sc.mats.push_back({kind, albedo, emit});
        return (int)sc.mats.size() - 1;
    };
    const Vec3 X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);

    // Box walls, normals pointing inward. The front wall (z = +2) is
    // deliberately omitted so the camera can look inside.
    sc.quads.push_back({{ 0, -2, 0},  Y, X, Z, 2, 2, mat(MAT_DIFFUSE, {0.85, 0.78, 0.68}, {0, 0, 0})});  // floor
    sc.quads.push_back({{ 0,  2, 0}, -Y, X, Z, 2, 2, mat(MAT_DIFFUSE, {0.80, 0.80, 0.80}, {0, 0, 0})});  // ceiling
    sc.quads.push_back({{ 0,  0, -2},  Z, X, Y, 2, 2, mat(MAT_DIFFUSE, {0.72, 0.72, 0.80}, {0, 0, 0})});  // back
    sc.quads.push_back({{-2,  0,  0},  X, Z, Y, 2, 2, mat(MAT_DIFFUSE, {0.80, 0.74, 0.74}, {0, 0, 0})});  // left
    sc.quads.push_back({{ 2,  0,  0}, -X, Z, Y, 2, 2, mat(MAT_DIFFUSE, {0.74, 0.80, 0.74}, {0, 0, 0})});  // right

    // Mirror sphere, radius 1.
    sc.spheres.push_back({{0, 0, 0}, 1.0, mat(MAT_MIRROR, {0.98, 0.98, 0.99}, {0, 0, 0})});

    // 1.6 m square area light on the ceiling, facing down.
    sc.light = (int)sc.quads.size();
    sc.quads.push_back({{0, 2 - 1e-3, 0}, {0, -1, 0}, X, Z, 0.8, 0.8,
                        mat(MAT_LIGHT, {0, 0, 0}, {4, 3.75, 3.5})});
    return sc;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

// Next event estimation: one uniform sample on the area light.
// Ld = ∫_light (albedo/pi) * Le * cosT * cosL / d^2 dA,  pdf = 1/A,
// so the sample is (albedo/pi) * Le * cosT * cosL * A / d^2.
static Vec3 direct_light(const Scene& sc, const Vec3& p, const Vec3& n,
                         const Vec3& albedo, RNG& rng) {
    const Quad& L = sc.quads[sc.light];
    Vec3 lp = L.c + L.u * (L.hu * (2 * rng.u() - 1)) + L.v * (L.hv * (2 * rng.u() - 1));
    Vec3 w = lp - p;
    double d2 = w.dot(w);
    double d = std::sqrt(d2);
    w = w / d;
    double cosT = n.dot(w);            // cosine at the shading point
    double cosL = (-w).dot(L.n);       // cosine at the light (must face back)
    if (cosT <= 0 || cosL <= 0) return {};
    // Visibility: any occluder strictly between p and the light sample?
    Hit h = sc.intersect({p + n * 1e-4, w}, 1e-4, d - 1e-3);
    if (h.t >= 0) return {};           // shadowed
    double area = 4 * L.hu * L.hv;     // = 1/pdf for a uniform sample
    Vec3 Le = sc.mats[L.id].emit;      // radiance of the light (per its material)
    return Le * albedo * (cosT * cosL * area) / (M_PI * d2);
}

// Cosine-weighted direction in the hemisphere around n; pdf = cos(theta)/pi.
static Vec3 cosine_hemisphere(const Vec3& n, RNG& rng) {
    double r = std::sqrt(rng.u()), phi = 2 * M_PI * rng.u();
    Vec3 local = {r * std::cos(phi), r * std::sin(phi), std::sqrt(1 - r * r)};
    Vec3 ref = std::fabs(n.y) < 0.99 ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 t = ref.cross(n).norm(), b = n.cross(t);      // orthonormal basis, z = n
    return t * local.x + b * local.y + n * local.z;
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------
static constexpr double EPS = 1e-4;    // surface-offset to avoid self-hits
static constexpr int MAX_DEPTH = 16;

static Vec3 trace(const Scene& sc, Ray r, int depth, RNG& rng) {
    Hit h = sc.intersect(r, EPS, 1e30);
    if (h.t < 0) return {};                        // escaped the open box → black
    const Material& m = sc.mats[h.id];
    if (m.kind == MAT_LIGHT) return m.emit;        // ray hit the light head-on
    if (depth >= MAX_DEPTH) return {};

    // Double-sided shading: make the normal face the incoming ray.
    if (r.d.dot(h.n) > 0) h.n = -h.n;

    if (m.kind == MAT_MIRROR) {
        // Perfect specular is a delta BSDF: the reflected direction is the
        // only sample, so no NEE term is needed; the light appears as a
        // sharp reflection whenever the reflected ray lands on the quad.
        return m.albedo * trace(sc, {h.p + h.n * EPS, reflect(r.d, h.n)}, depth + 1, rng);
    }

    // Diffuse (Lambertian): direct via NEE + one cosine-weighted bounce.
    Vec3 col = direct_light(sc, h.p, h.n, m.albedo, rng);
    Vec3 w = cosine_hemisphere(h.n, rng);
    col += m.albedo * trace(sc, {h.p + h.n * EPS, w}, depth + 1, rng);  // /pi cancels pdf
    // Russian roulette: terminate the path with probability 1-p, scale by 1/p.
    double p = std::clamp(std::max({col.x, col.y, col.z}) * 0.9 + 0.1, 0.1, 0.99);
    if (rng.u() < p) col = col / p;
    return col;
}

// ---------------------------------------------------------------------------
// Camera, tone mapping, main
// ---------------------------------------------------------------------------
static const int W = 512, H = 512;
static constexpr double EXPOSURE = 1.0;

static double to_srgb(double c) {
    c = clamp01(c);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

int main(int argc, char** argv) {
    int spp = argc > 1 ? std::atoi(argv[1]) : 128;
    if (spp < 1) spp = 1;
    int nthreads = argc > 2 ? std::atoi(argv[2]) : (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 256) nthreads = 256;

    Scene sc = build_scene();

    // Pinhole camera in front of the open face, aimed down the -z axis at the
    // sphere. Focal length is in pixel units: half the sensor (W/2 px) subtends
    // half the FOV, so f = (W/2) / tan(FOV/2). From z = 5 with a 48 deg FOV the
    // 4x4 box opening (half-angle atan(2/5) ~ 22 deg) fills ~90% of the frame,
    // with the sphere centered and the ceiling light visible at the top.
    Vec3 ro(0, 0, 5), camDir(0, 0, -1), right(1, 0, 0), up(0, 1, 0);
    double f = (W / 2.0) / std::tan(24.0 * M_PI / 180.0);

    std::vector<double> img((size_t)W * H * 3, 0.0);
    std::atomic<int> done_rows{0};

    auto render_rows = [&](int r0, int r1, unsigned long long seed) {
        RNG rng(seed);
        for (int y = r0; y < r1; ++y) {
            for (int x = 0; x < W; ++x) {
                Vec3 acc{0, 0, 0};
                for (int s = 0; s < spp; ++s) {
                    // Jittered sample inside the pixel, centered on the optical
                    // axis. Image y runs downward, so py is negated to put the
                    // world "up" (ceiling) at the top of the frame.
                    double px = (x + rng.u()) - W / 2.0;
                    double py = H / 2.0 - (y + rng.u());
                    Ray r{ro, (right * (px / f) + up * (py / f) + camDir).norm()};
                    acc += trace(sc, r, 0, rng);
                }
                acc = acc * (1.0 / spp);
                double* out = &img[(size_t)(y * W + x) * 3];
                for (int c = 0; c < 3; ++c) {
                    double v = (&acc.x)[c] * EXPOSURE;   // exposure
                    out[c] = to_srgb(v / (1.0 + v));     // Reinhard + sRGB
                }
            }
            int d = ++done_rows;
            if (d % 32 == 0) std::fprintf(stderr, "  %d/%d rows\n", d, H);
        }
    };

    std::vector<std::thread> pool;
    const int rows_per = (H + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        int r0 = t * rows_per, r1 = std::min(H, r0 + rows_per);
        if (r0 >= r1) break;
        pool.emplace_back(render_rows, r0, r1,
                          0x9E3779B97F4A7C15ULL + t * 0xBF58476D1CE4E5B9ULL);
    }
    for (auto& th : pool) th.join();

    FILE* fp = std::fopen("output.ppm", "wb");
    if (!fp) { std::fprintf(stderr, "cannot open output.ppm\n"); return 1; }
    std::fprintf(fp, "P6\n%d %d\n255\n", W, H);
    for (double v : img) fputc((int)std::lrint(clamp01(v) * 255.0), fp);
    std::fclose(fp);
    std::fprintf(stderr, "wrote output.ppm (%d x %d, %d spp, %d threads)\n", W, H, spp, nthreads);
    return 0;
}
