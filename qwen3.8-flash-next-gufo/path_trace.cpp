// =============================================================================
//  path_trace.cpp -- a small, dependency-free path tracer with direct lighting
//                    (next-event estimation / explicit light sampling).
//
//  build :  g++ -O3 -std=c++17 -pthread path_trace.cpp -o path_trace
//  run   :  ./path_trace [spp] [outfile]        defaults: 64  output.ppm
//
//  scene :  a perfect mirror sphere (r = 1) inside a diffuse box of "radius" 2
//           (inner half-width 2) whose front face is open, plus a rectangular
//           area light recessed in the ceiling pointing straight down.
//           The camera looks through the open front face.
//
//  integrator :  classic path tracing, one path per pixel sample.
//      * diffuse vertices use cosine-weighted sampling; the mirror uses the
//        exact reflection direction (delta BSDF).
//      * direct lighting at every DIFFUSE vertex is estimated by picking a
//        point straight on the light quad and shadow-testing it (NEE).
//      * a light hit is added as emission only when the previous vertex was a
//        delta surface (or on the primary ray).  At diffuse vertices NEE has
//        already accounted for the light, so BSDF-sampled light hits are
//        dropped there -- that is what keeps the two estimators from counting
//        the same light path twice.
//
//  output :  512x512, 8-bit sRGB binary PPM (P6), tone mapped.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

using Float = float;
static constexpr Float kPi  = Float(3.14159265358979);
static constexpr Float kInf = Float(1e30);
static constexpr int   kMaxDepth    = 16;   // path depth cap
static constexpr int   kRROStart    = 3;    // start Russian roulette here

// ------------------------------------------------------------------------- //
//  Math
// ------------------------------------------------------------------------- //
struct Vec3 {
    Float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(Float a) : x(a), y(a), z(a) {}
    Vec3(Float x_, Float y_, Float z_) : x(x_), y(y_), z(z_) {}
    Float  operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    Float& operator[](int i)       { return i == 0 ? x : (i == 1 ? y : z); }
};

inline Vec3 operator-(Vec3 a)          { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3 operator/(Vec3 a, Vec3 b) { return {a.x / b.x, a.y / b.y, a.z / b.z}; }
inline Vec3 operator*(Vec3 a, Float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(Float s, Vec3 a) { return a * s; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
inline Vec3& operator*=(Vec3& a, Vec3 b) { a = a * b; return a; }

inline Float dot(Vec3 a, Vec3 b)     { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3  cross(Vec3 a, Vec3 b)   { return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
inline Float length2(Vec3 a)         { return dot(a, a); }
inline Float length(Vec3 a)          { return std::sqrt(dot(a, a)); }
inline Vec3  normalize(Vec3 a)       { Float l = length(a); return l > 1e-12f ? a / l : Vec3(0,0,0); }
inline Float clampF(Float v, Float lo, Float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline Vec3  clamp01(Vec3 v) { return {clampF(v.x,0.f,1.f), clampF(v.y,0.f,1.f), clampF(v.z,0.f,1.f)}; }
inline Float maxComponent(Vec3 v)    { return std::max({v.x, v.y, v.z}); }

// Ray: p(t) = o + t*d.
struct Ray { Vec3 o, d; Vec3 at(Float t) const { return o + d * t; } };

// ------------------------------------------------------------------------- //
//  RNG -- PCG32, cheap and well distributed; seeded per (pixel, sample) so
//  the image is reproducible and independent of the thread count.
// ------------------------------------------------------------------------- //
struct Pcg32 {
    uint64_t state;
    explicit Pcg32(uint64_t seed, uint64_t seq = 0) : state(0) {
        state = 0; operator()(); state += seed * 6364136223846793005ull + seq + 1442695040888963407ull;
        operator()();
    }
    uint32_t operator()() {
        uint64_t old = state; state = old * 6364136223846793005ull + 1442695040888963407ull;
        uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = uint32_t(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }
    Float nextFloat() { return Float((*this)() >> 8) * Float(1.0 / (1u << 24)); }   // [0,1)
};

// Cosine-weighted hemisphere direction about n (Lambert importance sampling);
// pdf(w) = cos(theta)/pi, which cancels the Lambert denominator below.
static Vec3 cosineHemisphere(Vec3 n, Pcg32& rng) {
    Vec3 w = normalize(n);
    Vec3 tmp = std::fabs(w.x) > 0.9f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 u = normalize(cross(w, tmp)), v = cross(w, u);
    Float r1 = rng.nextFloat(), r2 = rng.nextFloat();
    Float r = std::sqrt(r1), phi = Float(2 * kPi) * r2;
    return normalize(u * (r * std::cos(phi)) + v * (r * std::sin(phi)) + w * std::sqrt(std::max(0.f, 1 - r1)));
}

// ------------------------------------------------------------------------- //
//  Materials
// ------------------------------------------------------------------------- //
enum class MatType { Diffuse, Mirror, Emissive };

struct Material {
    MatType type     = MatType::Diffuse;
    Vec3    albedo   {0.6f};        // Diffuse:  Lambert reflectance
    Vec3    refl     {1.0f};        // Mirror:   Fresnel-free tint of the reflection
    Vec3    emission {0.0f};        // Emissive: radiance in W/sr/m^2 (arbitrary units)
};

// ------------------------------------------------------------------------- //
//  Geometry
// ------------------------------------------------------------------------- //
struct Hit { Float t; Vec3 p, n; const Material* mat; };

struct Shape {
    virtual ~Shape() = default;
    // Closest intersection within [tMin, tMax].  Shapes are single sided:
    // only the side the normal points to is visible / intersectable.
    virtual bool hit(const Ray& r, Float tMin, Float tMax, Hit& h) const = 0;
};

// Axis-aligned rectangle.  `axis` is the normal axis (0=x,1=y,2=z); the rect
// covers [u0,u1] x [v0,v1] in the two remaining axes (au, av).
struct Quad : Shape {
    int   axis; Float k, sgn;               // plane: coord(axis) == k, normal = sgn * axis
    Float u0, u1, v0, v1; int au, av;
    const Material* mat;

    Quad(int axis_, Float k_, Float sgn_, Float u0_, Float u1_, Float v0_, Float v1_, const Material* m)
        : axis(axis_), k(k_), sgn(sgn_), u0(u0_), u1(u1_), v0(v0_), v1(v1_), mat(m) {
        au = (axis + 1) % 3; av = (axis + 2) % 3;                    // the two tangent axes
    }
    Vec3 normal() const { Vec3 n(0); n[axis] = sgn; return n; }
    Float area()  const { return (u1 - u0) * (v1 - v0); }
    Vec3  pointAt(Float u, Float v) const { Vec3 p(0); p[axis] = k; p[au] = u; p[av] = v; return p; }

    bool hit(const Ray& r, Float tMin, Float tMax, Hit& h) const override {
        Float denom = r.d[axis];
        if (std::fabs(denom) < 1e-9f) return false;                  // parallel to the plane
        Float t = (k - r.o[axis]) / denom;
        if (t < tMin || t > tMax) return false;
        Vec3 p = r.at(t), n = normal();
        if (dot(r.d, n) >= 0) return false;                          // back side -> invisible
        if (p[au] < u0 || p[au] > u1 || p[av] < v0 || p[av] > v1) return false;
        h = Hit{t, p, n, mat};
        return true;
    }
};

struct Sphere : Shape {
    Vec3  center; Float radius; const Material* mat;
    Sphere(Vec3 c, Float r, const Material* m) : center(c), radius(r), mat(m) {}

    bool hit(const Ray& r, Float tMin, Float tMax, Hit& h) const override {
        Vec3  oc = r.o - center;
        Float b = dot(oc, r.d), c = length2(oc) - radius * radius;
        Float disc = b * b - c;
        if (disc < 0) return false;
        Float t = -b - std::sqrt(disc);                              // near root (sphere is solid)
        if (t < tMin || t > tMax) return false;
        h = Hit{t, r.at(t), normalize(r.at(t) - center), mat};
        return true;
    }
};

// Emissive rectangle (the ceiling light).  Also acts as a sampler for NEE.
struct RectLight : Quad {
    RectLight(int axis_, Float k_, Float sgn_, Float u0_, Float u1_, Float v0_, Float v1_, const Material* m)
        : Quad(axis_, k_, sgn_, u0_, u1_, v0_, v1_, m) {}

    struct Sample { Vec3 target, n; Vec3 wi; Float dist2, cosLight, pdfA; };

    // Uniformly sample a point on the quad (area measure) as seen from p.
    Sample sample(Vec3 p, Pcg32& rng) const {
        Vec3 q = pointAt(u0 + (u1 - u0) * rng.nextFloat(), v0 + (v1 - v0) * rng.nextFloat());
        Sample s; s.target = q; s.n = normal(); s.pdfA = 1 / area();
        s.dist2 = std::max(length2(q - p), 1e-8f);
        s.wi = (q - p) * (1 / std::sqrt(s.dist2));
        s.cosLight = dot(s.n, -s.wi);                                // light faces p?
        return s;
    }
};

// ------------------------------------------------------------------------- //
//  Scene
// ------------------------------------------------------------------------- //
struct Scene {
    std::vector<const Shape*> shapes;
    std::vector<const RectLight*> lights;

    bool closestHit(const Ray& r, Float tMin, Float tMax, Hit& h) const {
        bool any = false; Float best = tMax;
        for (const Shape* s : shapes) { Hit hh; if (s->hit(r, tMin, best, hh)) { best = hh.t; h = hh; any = true; } }
        return any;
    }
    // Shadow test: does anything block the segment p -> p+wi*maxT?  (the light
    // itself is skipped, it is the target of the ray.)
    bool occluded(const Vec3& p, const Vec3& wi, Float maxT, const Shape* ignore) const {
        Ray r{p + wi * 1e-4f, wi}; Hit hh;
        for (const Shape* s : shapes) if (s != ignore && s->hit(r, 1e-4f, maxT, hh)) return true;
        return false;
    }
};

// ------------------------------------------------------------------------- //
//  Film / camera
// ------------------------------------------------------------------------- //
struct Camera {
    Vec3 origin, w, u, v, horizontal, vertical;

    static Camera lookAt(Vec3 eye, Vec3 target, Vec3 vup, Float vFovDeg, Float aspect) {
        Camera c; c.origin = eye;
        c.w = normalize(eye - target);                               // looks along -w
        c.u = normalize(cross(vup, c.w));
        c.v = cross(c.w, c.u);
        Float halfH = std::tan(vFovDeg * Float(kPi / 180) * 0.5f);   // frame half height at t = 1
        c.horizontal = c.u * (2 * halfH * aspect);
        c.vertical   = c.v * (2 * halfH);
        return c;
    }
    // (s,t) in [0,1]^2 over the frame (already jittered).
    Ray shoot(Float s, Float t) const {
        return {origin, normalize(-w + horizontal * (s - 0.5f) + vertical * (t - 0.5f))};
    }
};

// ------------------------------------------------------------------------- //
//  Integrator
// ------------------------------------------------------------------------- //
// What a ray sees when it leaves the box through the open front face: the dim
// surrounding room.  Without this the mirror shows a pure-black hole where it
// reflects the opening.
static Vec3 background(const Ray& r) {
    Float a = clampF(0.5f * (normalize(r.d).y + 1), 0, 1);           // 0 down, 1 up
    return Vec3(0.03f, 0.035f, 0.045f) * (1 - a) + Vec3(0.16f, 0.18f, 0.22f) * a;
}

static Vec3 trace(const Scene& sc, Ray r, Pcg32& rng) {
    Vec3 sum(0), throughput(1);
    // Light hits are counted as emission only when they are not already
    // covered by NEE: i.e. for the primary ray and after specular bounces.
    bool countLightHits = true;

    for (int depth = 0; depth < kMaxDepth; ++depth) {
        Hit h;
        // Ray left the scene (through the open face): pick up the environment.
        if (!sc.closestHit(r, 1e-4f, kInf, h)) { sum += throughput * background(r); break; }
        const Material& m = *h.mat;

        if (m.type == MatType::Emissive) {
            if (countLightHits) sum += throughput * m.emission;
            break;                                                   // emitters are opaque
        }

        // ---- direct lighting: next event estimation ------------------------
        // One sample per light, uniform in area on the quad:
        //   L <- f * Le * cos(shading) * cos(light) / (dist^2 * pdfArea)
        if (m.type == MatType::Diffuse) {
            for (const RectLight* L : sc.lights) {
                RectLight::Sample s = L->sample(h.p, rng);
                Float cosShade = dot(h.n, s.wi);
                if (cosShade <= 0 || s.cosLight <= 0) continue;      // facing away
                if (sc.occluded(h.p, s.wi, std::sqrt(s.dist2) - 1e-3f, L)) continue;  // in shadow
                sum += throughput * (m.albedo / kPi) * L->mat->emission
                     * (cosShade * s.cosLight / (s.dist2 * s.pdfA));
            }
        }

        // ---- indirect lighting: sample the BSDF and keep going -------------
        if (m.type == MatType::Diffuse) {
            r = Ray{h.p + h.n * 1e-4f, cosineHemisphere(h.n, rng)};
            throughput *= m.albedo;                                  // (albedo/pi)*cos / (cos/pi)
            countLightHits = false;                                  // NEE already did this vertex
        } else {                                                     // perfect mirror (delta BSDF)
            r = Ray{h.p + h.n * 1e-4f, r.d - h.n * (2 * dot(r.d, h.n))};
            throughput *= m.refl;                                    // delta terms cancel
            countLightHits = true;                                   // light visible only by bouncing
        }

        // Russian roulette: stop dim paths early, unbiasedly compensating.
        if (depth >= kRROStart) {
            Float p = std::min(maxComponent(throughput), 0.95f);
            if (rng.nextFloat() > p) break;
            throughput = throughput * (1 / p);
        }
    }
    return sum;
}

// ------------------------------------------------------------------------- //
//  Tone mapping + sRGB
// ------------------------------------------------------------------------- //
static Float tonemap(Float c, Float exposure) {
    c *= exposure;
    c = 1 - std::exp(-c);                                    // exponential exposure: smooth rolloff, never clips hard
    return c;
}
static Float toSrgb(Float c) {                               // linear -> sRGB (IEC 61966-2-1)
    c = clampF(c, 0, 1);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1 / 2.4f) - 0.055f;
}

// ------------------------------------------------------------------------- //
//  Main
// ------------------------------------------------------------------------- //
int main(int argc, char** argv) {
    const int    spp  = argc > 1 ? std::atoi(argv[1]) : 64;
    const char*  file = argc > 2 ? argv[2] : "output.ppm";
    const int    W = 512, H = 512;
    const Float  exposure = Float(1.0);

    // ---- materials ---------------------------------------------------------
    Material floor  {MatType::Diffuse,  {0.62f, 0.58f, 0.54f}, {}, {}};
    Material wallB  {MatType::Diffuse,  {0.55f, 0.57f, 0.60f}, {}, {}};
    Material wallS  {MatType::Diffuse,  {0.60f, 0.60f, 0.62f}, {}, {}};
    Material ceil   {MatType::Diffuse,  {0.68f, 0.68f, 0.68f}, {}, {}};
    Material mirror {MatType::Mirror,   {}, {0.96f, 0.97f, 0.99f}, {}};   // slightly cool mirror
    Material lamp   {MatType::Emissive, {}, {}, {7.5f, 6.9f, 6.0f}};      // warm panel

    // ---- scene: box of half-width 2, front face (-z) left open -------------
    const Float B = 2;                                                  // "radius" of the box
    Scene sc;
    std::vector<Shape*> owned{
        new Quad(1, -B, +1, -B, B, -B, B, &floor),                     // floor  y = -2
        new Quad(1,  B, -1, -B, B, -B, B, &ceil),                      // ceiling y = +2
        new Quad(2,  B, -1, -B, B, -B, B, &wallB),                     // back   z = +2
        new Quad(0, -B, +1, -B, B, -B, B, &wallS),                     // left   x = -2
        new Quad(0,  B, -1, -B, B, -B, B, &wallS),                     // right  x = +2
        new Sphere{{0, -1, 0}, 1.f, &mirror},                         // mirror sphere r = 1, resting on the floor
    };
    // 2 x 2 ceiling light, recessed just under the ceiling, nudged towards the
    // camera so the front of the sphere and the floor in front of it get lit.
    RectLight* light = new RectLight(1, B - 0.02f, -1, -1.0f, 1.0f, -1.6f, 0.4f, &lamp);
    owned.push_back(light);
    for (Shape* s : owned) sc.shapes.push_back(s);
    sc.lights.push_back(light);

    // ---- camera: outside the box, looking in through the open front --------
    Camera cam = Camera::lookAt({0, 0.35f, -5.3f}, {0, -0.05f, 0.4f}, {0, 1, 0}, 55, 1.0f);

    // ---- render (row bands across hardware threads) ------------------------
    std::vector<Vec3> accum(size_t(W) * H, Vec3(0));
    unsigned nThreads = std::min<unsigned>(std::thread::hardware_concurrency(), 32u);
    if (nThreads == 0) nThreads = 1;
    std::vector<std::thread> pool;
    for (unsigned tid = 0; tid < nThreads; ++tid) {
        pool.emplace_back([&, tid] {
            for (int y = int(tid); y < H; y += int(nThreads)) {
                for (int x = 0; x < W; ++x) {
                    Vec3 col(0);
                    for (int s = 0; s < spp; ++s) {
                        Pcg32 rng(uint64_t(y) * W + x, uint64_t(s) * 7919 + tid * 104729);
                        Float jx = (x + rng.nextFloat()) / W;           // jitter for AA
                        Float jy = (y + rng.nextFloat()) / H;
                        col += trace(sc, cam.shoot(jx, jy), rng);
                    }
                    accum[size_t(y) * W + x] = col * (1.f / spp);
                }
            }
        });
    }
    for (auto& t : pool) t.join();

    // ---- write 8-bit sRGB PPM (P6).  Row 0 of a PPM is the top of the image,
    //      which is our last row. -------------------------------------------
    if (std::FILE* f = std::fopen(file, "wb")) {
        std::fprintf(f, "P6\n%d %d\n255\n", W, H);
        std::vector<unsigned char> row(size_t(W) * 3);
        for (int y = H - 1; y >= 0; --y) {
            for (int x = 0; x < W; ++x) {
                Vec3 c = accum[size_t(y) * W + x];
                row[size_t(x) * 3 + 0] = unsigned(toSrgb(tonemap(c.x, exposure)) * 255.f + 0.5f);
                row[size_t(x) * 3 + 1] = unsigned(toSrgb(tonemap(c.y, exposure)) * 255.f + 0.5f);
                row[size_t(x) * 3 + 2] = unsigned(toSrgb(tonemap(c.z, exposure)) * 255.f + 0.5f);
            }
            std::fwrite(row.data(), 1, row.size(), f);
        }
        std::fclose(f);
        std::printf("wrote %s  (%dx%d, %d spp)\n", file, W, H, spp);
    } else {
        std::fprintf(stderr, "cannot open %s\n", file);
        return 1;
    }
    for (Shape* s : owned) delete s;
    return 0;
}
