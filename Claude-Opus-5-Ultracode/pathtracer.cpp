// pathtracer.cpp -- unidirectional path tracer with next-event estimation.
//
// Scene: a perfect-mirror sphere (r = 1) at the origin inside a diffuse Cornell-style box whose
// interior spans [-2,2]^3. Five walls (left red, right green, floor/ceiling/back neutral); the
// +z face is OPEN, so rays that leave through it return black. The only emitter is a 1.2 x 1.2
// quad in the ceiling at y = 1.998 facing straight down. Camera at (0,0,6.5) looking down -z.
//
// Build: c++ -std=c++17 -O3 -Wall -Wextra -pthread pathtracer.cpp -o pathtracer
// Run:   ./pathtracer          (no arguments; writes ./output.ppm)
// Out:   512x512 binary PPM (P6, maxval 255), 8-bit sRGB-encoded, no tone map.
//
// Correct-but-surprising output, do not "fix":
//   - The sphere has a large black core: it mirrors the open front face (~44 px of a ~90 px
//     silhouette). Its rim is CONTINUOUS with the back wall behind it; a hard dark ring there
//     would be the real bug.
//   - No point of the floor is ever fully lit: the penumbra reaches r ~3.45, past the floor's 2.83
//     corner radius, so even the corners see only ~81% of the panel. The central ~29% is true
//     umbra (out to r = 1.32 on an axis, 1.16 on a diagonal) and gets no direct light at all.
//     The ceiling is lit only by interreflection -- the emitter faces away from it.
//   - Two regions clip to pure white and only two: the emitter trapezoid and its mirror image on
//     the sphere. Nothing else clips, but the margin is thin: the red wall beside the light runs
//     to code 254 and the back wall to 240.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <thread>
#include <vector>

static constexpr int    kWidth = 512, kHeight = 512;
static constexpr int    kSqrtSpp   = 96;      // 9216 spp, stratified 96x96 over the pixel; ~19 s
                                              // on 18 cores. NEE removes the dominant variance, so
                                              // what is left is the interreflection-only ceiling:
                                              // still visibly grainy at 1024 spp, smooth here.
static constexpr int    kMaxDepth  = 32;      // hard backstop; RR ends almost every path first
static constexpr int    kRRDepth   = 3;       // roulette from the 4th vertex on
static constexpr double kRayEps    = 1e-6;    // tmin for every ray: 1e9x the ~1e-15 rounding in a
                                              // hit point here, 2000x under the 0.002 light gap
static constexpr double kShadowEps = 1e-4;    // shadow rays stop short of the light plane
static constexpr double kPi        = 3.14159265358979323846;
static constexpr double kInf       = std::numeric_limits<double>::infinity();
static constexpr double kTanHalf   = 2.0 / 4.5;  // vfov = 2*atan(2/4.5); derived, never degrees
static constexpr int    kLight     = 5;          // index of the emitter in Scene::quads

// ---------------------------------------------------------------------------- vector and ray

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
    // Ternary indexing: (&x)[i] on a non-array member is undefined behaviour.
    double  operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    double& operator[](int i)       { return i == 0 ? x : (i == 1 ? y : z); }
};
static Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
static Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
// Vec3 * Vec3 is the componentwise SPECTRAL product (albedo x radiance), not a dot product.
static Vec3 operator*(const Vec3& a, const Vec3& b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
static Vec3 operator*(const Vec3& a, double s)      { return Vec3(a.x * s, a.y * s, a.z * s); }
static Vec3& operator+=(Vec3& a, const Vec3& b)     { a = a + b; return a; }
static double dot(const Vec3& a, const Vec3& b)     { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 normalize(const Vec3& a)                { return a * (1.0 / std::sqrt(dot(a, a))); }
static double max_comp(const Vec3& a)               { return std::max(a.x, std::max(a.y, a.z)); }

struct Ray { Vec3 o, d; };   // invariant: d is unit, so the sphere quadratic has a == 1

static const Vec3 kEye(0.0, 0.0, 6.5);
// Emitted radiance. The image is exactly linear in kLe, so exposure is one constant. The brightest
// diffuse band -- the red wall beside the light -- sits at 0.94 linear (sRGB code 248; its noisiest
// pixel reaches 254), so nothing clips but the emitter and its mirror image and there is almost no
// headroom: raising kLe or any albedo clips that band. Retune with kLe_new = 20 * target / 0.94 --
// and MEASURE: direct light tops out at 0.58 here, interreflection supplies the other 40%.
static const Vec3 kLe (20.0, 20.0, 20.0);

// ---------------------------------------------------------------------------- rng

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

struct Pcg32 {
    uint64_t state, inc;
    Pcg32(uint64_t seq, uint64_t seed) : state(0), inc((seq << 1) | 1ULL) {
        next_u32(); state += seed; next_u32();      // canonical PCG seeding: step, add, step
    }
    uint32_t next_u32() {
        const uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        const uint32_t xorshifted = uint32_t(((old >> 18) ^ old) >> 27);
        const uint32_t rot = uint32_t(old >> 59);
        return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
    }
    // Exact in double; range [0, 1-2^-32], so never 1.0.
    double uniform() { return next_u32() * 0x1p-32; }
};

// ---------------------------------------------------------------------------- primitives

enum class Mat { Diffuse, Mirror, Light };

struct Hit {
    double t = kInf;          // traversal tmax on entry AND the resulting distance
    Vec3   p, n, albedo;      // n is the fixed geometric normal, NEVER flipped toward the ray
    Mat    mat = Mat::Diffuse;
};

// Axis-aligned rectangle on the plane component[axis] == k, extents given in u = (axis+1)%3 and
// v = (axis+2)%3; the normal is nsign along axis and points INTO the box.
struct Quad {
    int    axis;
    double k, lo0, hi0, lo1, hi1, nsign;
    Vec3   albedo;
    Mat    mat;

    Vec3   normal() const { Vec3 n; n[axis] = nsign; return n; }
    double area()   const { return (hi0 - lo0) * (hi1 - lo1); }

    // Uniform point on the quad, so pdf_A = 1/area.
    Vec3 sample(double u, double v) const {
        Vec3 p;
        p[axis]            = k;
        p[(axis + 1) % 3]  = lo0 + (hi0 - lo0) * u;
        p[(axis + 2) % 3]  = lo1 + (hi1 - lo1) * v;
        return p;
    }

    // Traversal contract, shared by every primitive: h.t is the current tmax on entry and is
    // overwritten only for a hit in (tmin, h.t). A default-constructed Hit means "unbounded";
    // occluded() reuses the same code by presetting h.t to a shortened distance.
    bool intersect(const Ray& r, double tmin, Hit& h) const {
        const int u = (axis + 1) % 3, v = (axis + 2) % 3;
        // No parallel-ray guard: r.d[axis] == 0 yields +-inf (or NaN if the origin lies on the
        // plane) and the ordered test below is false for all three, which is the correct reject.
        const double t = (k - r.o[axis]) / r.d[axis];
        if (!(t > tmin && t < h.t)) return false;
        const Vec3 p = r.o + r.d * t;
        if (p[u] < lo0 || p[u] > hi0 || p[v] < lo1 || p[v] > hi1) return false;
        h.t = t; h.p = p; h.n = normal(); h.albedo = albedo; h.mat = mat;
        return true;
    }
};

struct Sphere {
    Vec3   c;
    double r;
    bool intersect(const Ray& ray, double tmin, Hit& h) const {
        const Vec3 oc = ray.o - c;
        const double b = dot(oc, ray.d), cc = dot(oc, oc) - r * r;
        const double disc = b * b - cc;      // |d| == 1 so a == 1: the 4s drop out of the quadratic
        if (disc < 0.0) return false;
        const double sq = std::sqrt(disc);
        double t = -b - sq;                  // near root first
        if (t <= tmin) t = -b + sq;          // near root behind us -> try the far root
        if (!(t > tmin && t < h.t)) return false;
        h.t = t; h.p = ray.o + ray.d * t;
        h.n = (h.p - c) * (1.0 / r);         // exact, and cheaper than normalize()
        h.albedo = Vec3(); h.mat = Mat::Mirror;
        return true;
    }
};

struct Scene { std::array<Quad, 6> quads; Sphere sphere; };

static Scene make_scene() {
    Scene s{};
    const Vec3 white(0.73, 0.73, 0.73);
    // Red peak 0.76 vs green 0.58: equal peaks give the green wall ~2.1x the luminance of the red
    // one and the frame reads lopsided. Every albedo stays <= 0.8 so the box does not over-brighten.
    //                axis      k   lo0  hi0   lo1  hi1  nsign  albedo                  material
    s.quads[0] = Quad{0, -2.00, -2.0, 2.0, -2.0, 2.0,  1.0, Vec3(0.76, 0.12, 0.10), Mat::Diffuse};
    s.quads[1] = Quad{0,  2.00, -2.0, 2.0, -2.0, 2.0, -1.0, Vec3(0.12, 0.58, 0.14), Mat::Diffuse};
    s.quads[2] = Quad{1, -2.00, -2.0, 2.0, -2.0, 2.0,  1.0, white,                  Mat::Diffuse};
    s.quads[3] = Quad{1,  2.00, -2.0, 2.0, -2.0, 2.0, -1.0, white,                  Mat::Diffuse};
    s.quads[4] = Quad{2, -2.00, -2.0, 2.0, -2.0, 2.0,  1.0, white,                  Mat::Diffuse};
    // Emitter, 1.2 x 1.2 -> area 1.44. A unit-area light would make 1/pdf_A == 1 and silently hide
    // a dropped area factor. y = 1.998 rather than 2.0: coplanar with the ceiling gives ordering
    // ambiguity and speckle, and the 0.002 gap is 20x kShadowEps, so nothing above the panel can
    // occlude a shadow ray. Do not recess it further: a camera ray that just clears the near edge
    // lands on the strip of ceiling above the panel, which gets no direct light, and at y = 1.98
    // that strip is a whole pixel wide -- a black seam across the light. The panel's albedo is 0,
    // so a path that does enter the gap dies on a black absorber.
    s.quads[kLight] = Quad{1, 1.998, -0.6, 0.6, -0.6, 0.6, -1.0, Vec3(), Mat::Light};
    s.sphere = Sphere{Vec3(), 1.0};
    return s;
}

// Seven primitives: linear traversal. A BVH would be pure overhead.
static bool intersect(const Scene& s, const Ray& r, Hit& h) {
    bool hit = false;
    for (const Quad& q : s.quads) hit |= q.intersect(r, kRayEps, h);
    hit |= s.sphere.intersect(r, kRayEps, h);
    return hit;
}

static bool occluded(const Scene& s, const Ray& r, double dist) {
    Hit h;
    // The sampled light point sits at exactly t == dist. Shortening the ray is the whole
    // self-occlusion fix: the emitter stays in the loop and is rejected because its only crossing
    // of its own plane is at t == dist > h.t.
    h.t = dist - kShadowEps;
    for (const Quad& q : s.quads) if (q.intersect(r, kRayEps, h)) return true;
    return s.sphere.intersect(r, kRayEps, h);
}

// ---------------------------------------------------------------------------- sampling

// Duff et al., branchless: exact for every unit n, including the floor/ceiling normals (0,+-1,0)
// where cross(n,(0,1,0)) is the zero vector and normalize() would produce NaN.
static void onb(const Vec3& n, Vec3& t, Vec3& b) {
    const double sg = std::copysign(1.0, n.z);
    const double a  = -1.0 / (sg + n.z);          // sg + n.z is never 0 for a unit n
    const double bb = n.x * n.y * a;
    t = Vec3(1.0 + sg * n.x * n.x * a, sg * bb, -sg * n.x);
    b = Vec3(bb, sg + n.y * n.y * a, -n.y);
}

static Vec3 cosine_hemisphere(const Vec3& n, Pcg32& rng) {
    const double u1 = rng.uniform(), u2 = rng.uniform();
    // Malley's method: a uniform disk sample lifted to the hemisphere is cosine-distributed,
    // pdf(wi) = cos/PI. uniform() returns k * 2^-32 exactly, so 1 - u1 >= 2^-32 and the max() below
    // never binds -- it is there only so a future uniform() that can return 1.0 cannot NaN a pixel.
    const double rr = std::sqrt(u1), phi = 2.0 * kPi * u2;
    Vec3 t, b; onb(n, t, b);
    return t * (rr * std::cos(phi)) + b * (rr * std::sin(phi))
             + n * std::sqrt(std::max(0.0, 1.0 - u1));
}

static Vec3 reflect(const Vec3& d, const Vec3& n) { return d - n * (2.0 * dot(d, n)); }

// ---------------------------------------------------------------------------- integrator

// Next-event estimation: one uniform sample on the emitter, one shadow ray.
static Vec3 direct_light(const Scene& s, const Vec3& p, const Vec3& n,
                         const Vec3& albedo, Pcg32& rng) {
    const Quad& L = s.quads[kLight];
    // The draws are hoisted out of the call: as arguments they would be indeterminately sequenced,
    // so which one became u would be a compiler choice and the image would vary between toolchains.
    const double su = rng.uniform(), sv = rng.uniform();
    const Vec3 y  = L.sample(su, sv);                         // pdf_A = 1/area
    const Vec3 to = y - p;
    const double d2 = dot(to, to), d = std::sqrt(d2);         // d2 first: never recompute d*d
    const Vec3 wi = to * (1.0 / d);
    const double cos_s = dot(n, wi);                          // cosine at the shading point
    const double cos_l = -dot(L.normal(), wi);                // cosine at the emitter; the quad
    // emits only along -y, so cos_l <= 0 is a REJECT, not a case for fabs() -- every ceiling point
    // has cos_l < 0, and fabs() there would light the ceiling from a downward-facing panel.
    if (cos_s <= 0.0 || cos_l <= 0.0) return Vec3();          // cheap tests before the shadow ray
    if (occluded(s, Ray{p, wi}, d)) return Vec3();
    // Area -> solid angle: dw = cos_l * dA / d2, hence pdf_w = pdf_A * d2 / cos_l = d2/(area*cos_l).
    // contribution = f * Le * cos_s / pdf_w = (albedo/PI) * Le * cos_s * cos_l * area / d2.
    return albedo * kLe * (cos_s * cos_l * L.area() / (kPi * d2));
}

static Vec3 radiance(const Scene& s, Ray r, Pcg32& rng) {
    Vec3 L, T(1.0, 1.0, 1.0);
    bool specular = true;   // a pinhole camera is itself a delta "BSDF", so depth 0 may see Le
    for (int depth = 0; depth < kMaxDepth; ++depth) {
        Hit h;
        if (!intersect(s, r, h)) break;        // escaped through the open +z face -> black
        if (h.mat == Mat::Light) {
            // Le is added only when the emitter is reached by a camera ray or straight after a
            // specular bounce. Any other arrival was already estimated by NEE at the previous
            // diffuse vertex; adding both would double-count that connection. This is pure light
            // sampling, not MIS: exactly one strategy owns each path, with weight 1.
            // dot(r.d,h.n) < 0 keeps the emitter one-sided -- its back face is a black absorber.
            if (specular && dot(r.d, h.n) < 0.0) L += T * kLe;
            break;                             // the emitter neither reflects nor transmits
        }
        if (h.mat == Mat::Mirror) {
            // Delta BSDF: a light sample has zero probability of connecting through it, so NEE is
            // skipped and the flag is set instead -- the light this mirror reflects arrives via the
            // emitter-hit branch above. T is unchanged: the reflectance is exactly 1.
            r = Ray{h.p, reflect(r.d, h.n)};
            specular = true;
        } else {
            L += T * direct_light(s, h.p, h.n, h.albedo, rng);   // uses T BEFORE the update below
            specular = false;
            const Vec3 wi = cosine_hemisphere(h.n, rng);
            // Cosine-weighted sampling collapses the estimator exactly:
            //   f*cos/pdf = (albedo/PI)*cos / (cos/PI) = albedo.
            // The cosine and the 1/PI cancel identically -- do not "restore" either of them.
            T = T * h.albedo;
            r = Ray{h.p, wi};
        }
        if (depth >= kRRDepth) {
            // Russian roulette keyed on the max throughput component, capped so termination stays
            // possible. E[(S/q)*X] = E[X] for any q in (0,1], so dividing the survivor by q keeps
            // the estimator unbiased; omitting T/q is a systematic darkening that grows with depth.
            // The draw must be fresh -- reusing a direction sample correlates survival with the
            // direction and breaks the independence the proof needs. RR runs AFTER NEE, so a killed
            // path only forgoes future indirect light, never the direct term already collected.
            const double q = std::min(max_comp(T), 0.95);
            if (rng.uniform() >= q) break;
            T = T * (1.0 / q);
        }
    }
    return L;
}

// ---------------------------------------------------------------------------- output

static unsigned char to_byte(double v) {
    if (!(v > 0.0)) v = 0.0;   // negated form also maps NaN to black; std::max would pass it on
    if (v > 1.0) v = 1.0;      // the spec's clamp to [0,1]; it also keeps c*255+0.5 inside the
                               // narrowing cast's range below (v = 2 would give 345.6)
    // Exact IEC 61966-2-1 sRGB encode. 0.0031308 is the breakpoint on the LINEAR value (0.04045 is
    // the decode-direction breakpoint and mangles the shadows); a bare 1/2.2 gamma is off by up to
    // ~8.5/255 in the darks (worst near L = 0.002: code 7 against 16).
    const double c = (v <= 0.0031308) ? 12.92 * v : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
    return (unsigned char)(c * 255.0 + 0.5);   // round, do not truncate
}

static bool write_ppm(const char* path, const std::vector<Vec3>& fb) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << kWidth << " " << kHeight << "\n255\n";
    std::vector<unsigned char> bytes(size_t(3) * kWidth * kHeight);
    for (size_t i = 0; i < fb.size(); ++i) {
        bytes[3 * i + 0] = to_byte(fb[i].x);
        bytes[3 * i + 1] = to_byte(fb[i].y);
        bytes[3 * i + 2] = to_byte(fb[i].z);
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
    out.flush();   // surface a full disk or a lost mount here, not silently in the destructor
    return bool(out);
}

// ---------------------------------------------------------------------------- camera

static void render_row(const Scene& s, int y, Vec3* row) {
    for (int x = 0; x < kWidth; ++x) {
        // Stream and seed are both hashed and depend on (x,y) only -- never on a thread id, the row
        // order, or a shared counter. One thread owns a whole pixel and sums its samples into a
        // local accumulator in a fixed order, so the image is bit-identical for any thread count.
        // Feeding the raw pixel index into the LCG state instead produces structured banding,
        // because the first draw is then near-linear in the index.
        const uint64_t idx = uint64_t(y) * kWidth + uint64_t(x);
        Pcg32 rng(splitmix64(idx), splitmix64(idx ^ 0x9E3779B97F4A7C15ULL));
        Vec3 sum;
        for (int j = 0; j < kSqrtSpp; ++j)
        for (int i = 0; i < kSqrtSpp; ++i) {
            // Only the pixel domain is stratified; the light and hemisphere samples stay plain
            // random, because padding a stratified grid into higher dimensions without
            // decorrelation reintroduces correlation artifacts.
            const double jx = (i + rng.uniform()) / kSqrtSpp;
            const double jy = (j + rng.uniform()) / kSqrtSpp;
            // The opening is the square [-2,2]^2 at z = 2: half-height 2 at distance 4.5, so
            // tan(vfov/2) = 2/4.5 and the FRAME EDGES (not the extreme pixel centres) land on +-2.
            // Jitter is in [0,1), so (x+jx)/W spans [0,1) and every ray enters the opening.
            // cx > 0 maps to world +x and to the image right, so the -x wall (red) is on the left
            // and +x (green) on the right -- the Cornell convention.
            const double cx = (2.0 * (x + jx) / kWidth  - 1.0) * kTanHalf;
            const double cy = (1.0 - 2.0 * (y + jy) / kHeight) * kTanHalf;   // row 0 is the TOP
            sum += radiance(s, Ray{kEye, normalize(Vec3(cx, cy, -1.0))}, rng);
        }
        row[x] = sum * (1.0 / (double(kSqrtSpp) * kSqrtSpp));
    }
}

int main() {
    const Scene scene = make_scene();
    std::vector<Vec3> fb(size_t(kWidth) * kHeight);

    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 4;             // hardware_concurrency is allowed to return 0
    std::atomic<int> next_row{0}, done{0};
    const auto t0 = std::chrono::steady_clock::now();

    // Dynamic per-row claiming rather than a static split: per-row cost varies several-fold on an
    // asymmetric core layout. Determinism comes from the per-pixel seeding, not the row order.
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
        pool.emplace_back([&] {
            for (int y = next_row.fetch_add(1); y < kHeight; y = next_row.fetch_add(1)) {
                render_row(scene, y, &fb[size_t(y) * kWidth]);
                const int d = done.fetch_add(1) + 1;
                if (d % 16 == 0 || d == kHeight)
                    std::fprintf(stderr, "\rrendering %4d / %d rows", d, kHeight);
            }
        });
    }
    for (std::thread& t : pool) t.join();

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "\ndone in %.2f s (%d spp, %u threads)\n",
                 secs, kSqrtSpp * kSqrtSpp, nthreads);
    if (!write_ppm("output.ppm", fb)) {
        std::fprintf(stderr, "error: could not write output.ppm\n");
        return 1;                                // 20 s of render is worth more than a silent 0
    }
    return 0;
}
