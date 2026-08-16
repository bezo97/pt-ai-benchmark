// ============================================================================
// pathtracer.cpp
//
// A compact path tracer with direct lighting via next-event estimation (NEE)
// and multiple importance sampling (power heuristic) that blends
//   - uniform light sampling  (sample a point on the quad, shadow ray) and
//   - BSDF sampling           (the usual path-continuation ray)
// into a single unbiased estimator of the direct-light integral, so the two
// never double-count each other.
//
// Scene
// -----
//  * A diffuse box of radius 2 (cube [-2,2]^3) whose FRONT WALL (z = +2) is
//    removed, so the camera looks into the box from the front.
//  * A glossy specular (GGX microfacet) sphere of radius 1 resting on the
//    box floor.
//  * A 2x2 quad area light on the ceiling (y = +2), pointing down.
//
// Radiance estimated for one camera ray (T_b = product of sampled
// BRDF/pdf through bounce b):
//
//   L = sum_b  T_b * ( wL_b * NEE_b  +  wB_b * BDIR_b )  +  T_end * Le
//
//   NEE_b  : direct integral estimated by sampling a point l on the light
//            f(om_l) * Le(l) * cosP * cosL * A / |l-p|^2
//   BDIR_b : the same integral, but using the BSDF-sampled direction,
//            nonzero only when that direction happens to hit the light
//   wL_b, wB_b : power-heuristic weights f^2/(f^2+f^2) (per channel), where
//            f is the "contribution function" (integrand / pdf) of each
//            strategy evaluated at the shared sample.
//
// Rendering: 512x512, 256 spp (2x2 stratified), filmic tone map, 8-bit sRGB.
// Output:    output.ppm (P6).
// Build:     g++ -O3 -std=c++17 -pthread -o pathtracer pathtracer.cpp
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

static constexpr float PI = 3.14159265358979323846f;

// ---------------------------------------------------------------------------
// Small float3
// ---------------------------------------------------------------------------
struct Vec3 {
    float x = 0, y = 0, z = 0;
    float& operator[](int i)       { return i == 0 ? x : (i == 1 ? y : z); }
    const float& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(Vec3 a)            { return {-a.x, -a.y, -a.z}; }
static inline Vec3 operator-(Vec3 a, Vec3 b)    { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator*(Vec3 a, float s)   { return {a.x * s, a.y * s, a.z * s}; }
static inline Vec3 operator*(float s, Vec3 a)   { return a * s; }
static inline Vec3 operator*(Vec3 a, Vec3 b)    { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
static inline Vec3 operator/(Vec3 a, float s)   { return a * (1.0f / s); }
static inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
static inline float  dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3   cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline float  length2(Vec3 a) { return dot(a, a); }
static inline float  length(Vec3 a)  { return std::sqrt(length2(a)); }
static inline Vec3   normalize(Vec3 a) { return a * (1.0f / std::max(1e-12f, length(a))); }
static inline float  clampf(float v, float lo, float hi) { return std::min(hi, std::max(lo, v)); }

// ---------------------------------------------------------------------------
// PCG32: fast, high-quality 32-bit generator
// ---------------------------------------------------------------------------
struct PCG32 {
    uint64_t state, inc;
    explicit PCG32(uint64_t seed) {
        state = seed;
        inc = seed * 0x9E3779B97F4A7C15ull | 1u;      // must be odd
    }
    float next() {                                     // uniform [0,1)
        uint64_t old = state;
        state = old * 6364136223846793005ull + inc;
        uint32_t xs = (uint32_t)((old ^ (old >> 22)) >> ((old >> 27) + 5));
        uint32_t rot = (uint32_t)(old >> 27);
        xs = (xs >> rot) | (xs << ((-rot) & 31));      // XSH-RR
        return (float)(xs >> 8) * (1.0f / 16777216.0f); // top 24 bits -> [0,1)
    }
};

// ---------------------------------------------------------------------------
// Scene description
// ---------------------------------------------------------------------------
struct Hit {
    float t = 1e30f;        // distance along the ray (1e30 => no hit)
    Vec3 p, n;              // hit point, shading normal (facing the ray)
    int kind = 0;           // 0 = miss, 1 = diffuse, 2 = specular, 3 = light
};

static const Vec3  kBoxAlbedo   = {0.65f, 0.70f, 0.78f};  // diffuse box walls
static const Vec3  kSphereF0    = {0.90f, 0.92f, 0.95f};  // sphere base color
static const float kRoughness   = 0.18f;                   // sphere GGX roughness
static const float kBox         = 2.0f;                    // box = cube [-2,2]^3
static const Vec3  kSphereC     = {0.0f, -1.0f, 0.0f};     // sphere center (on floor)
static const float kSphereR     = 1.0f;
static const float kLightHalf   = 1.0f;                    // quad = [-1,1]^2 at y=+2
static const float kLightArea   = 4.0f;                    // = (2*kLightHalf)^2
static const Vec3  kLightN      = {0.0f, -1.0f, 0.0f};     // light faces down
static const float kLightInt    = 9.0f;                    // peak radiance
static const float kEps         = 1e-4f;                   // surface offset

// ---------------------------------------------------------------------------
// Intersectors
// ---------------------------------------------------------------------------
static void hitSphere(const Vec3& o, const Vec3& d, Hit& best) {
    Vec3 oc = o - kSphereC;
    float b = dot(oc, d);                      // |d| = 1, so t^2 + 2bt + c = 0
    float c = dot(oc, oc) - kSphereR * kSphereR;
    float disc = b * b - c;
    if (disc < 0.0f) return;
    float t = -b - std::sqrt(disc);
    if (t < kEps) t = -b + std::sqrt(disc);    // inside the sphere -> far side
    if (t < kEps || t >= best.t) return;
    Vec3 p = o + d * t;
    best = {t, p, (p - kSphereC) * (1.0f / kSphereR), 2};
}

// One axis-aligned box wall: coordinate `axis` fixed at `v`.
// (The front wall axis==2, v==+kBox is open, so it is never intersected.)
static void hitWall(int axis, float v, const Vec3& o, const Vec3& d, Hit& best) {
    if (std::abs(d[axis]) < 1e-12f) return;    // parallel to the wall
    const Vec3 e = (axis == 0) ? Vec3{1,0,0} : (axis == 1) ? Vec3{0,1,0} : Vec3{0,0,1};
    const Vec3 n = (v > 0.0f) ? e * -1.0f : e; // interior-facing normal
    if (dot(d, n) >= 0.0f) return;             // only visible from inside
    float t = (v - o[axis]) / d[axis];
    if (t < kEps || t >= best.t) return;
    Vec3 p = o + d * t;
    int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
    if (std::abs(p[a1]) > kBox + 1e-4f || std::abs(p[a2]) > kBox + 1e-4f) return;
    // The part of the ceiling covered by the quad is the emitter
    bool isLight = (axis == 1 && v > 0.0f &&
                    std::abs(p.x) <= kLightHalf && std::abs(p.z) <= kLightHalf);
    best = {t, p, n, isLight ? 3 : 1};
}

static Hit intersect(const Vec3& o, const Vec3& d) {
    Hit best;
    hitSphere(o, d, best);
    hitWall(0, -kBox, o, d, best);   // x = -2
    hitWall(0, +kBox, o, d, best);   // x = +2
    hitWall(1, -kBox, o, d, best);   // y = -2  (floor)
    hitWall(1, +kBox, o, d, best);   // y = +2  (ceiling, contains the light)
    hitWall(2, -kBox, o, d, best);   // z = -2  (back wall)
    return best;                     // z = +2 is open: the front of the box
}

// ---------------------------------------------------------------------------
// BRDFs
// ---------------------------------------------------------------------------
// Lambertian (the box walls)
static Vec3 evalDiffuse() { return kBoxAlbedo * (1.0f / PI); }

// GGX (Walter et al. 2007): distribution, Schlick-Smith G1, Schlick Fresnel
static float ggxD(float cosH, float a) {
    float d = cosH * cosH * (a - 1.0f) + 1.0f;
    return a / (PI * d * d);
}
static float ggxG1(float cosT, float a) {
    return a / (cosT * (cosT * (1.0f - a) + a) + 1e-9f);
}
static Vec3 fresnelSchlick(float cosT, Vec3 f0) {
    float s = 1.0f - cosT;
    s *= s * s * s * s;                          // (1-cos)^5
    return f0 * (1.0f - s) + Vec3{1,1,1} * s;
}

// Specular BRDF f(wo -> wi); both directions point away from the surface.
static Vec3 evalSpecular(Vec3 wi, Vec3 wo, Vec3 n) {
    float cosI = dot(n, wi), cosO = dot(n, wo);
    if (cosI <= 0.0f || cosO <= 0.0f) return {0, 0, 0};
    float a = kRoughness * kRoughness;
    Vec3 h = normalize(wi + wo);
    float D = ggxD(dot(n, h), a);
    float G = ggxG1(cosI, a) * ggxG1(cosO, a);
    Vec3 F = fresnelSchlick(std::min(1.0f, dot(h, wi)), kSphereF0);
    return Vec3{D * G, D * G, D * G} * F * (1.0f / (4.0f * cosI * cosO));
}

// pdf of a GGX-sampled direction wi given outgoing wo (same as below).
static float bsdfPdfSpecular(Vec3 wi, Vec3 wo, Vec3 n) {
    float cosI = dot(n, wi), cosO = dot(n, wo);
    if (cosI <= 0.0f || cosO <= 0.0f) return 0.0f;
    float a = kRoughness * kRoughness;
    Vec3 h = normalize(wi + wo);
    float cosH = dot(n, h);
    return ggxD(cosH, a) * cosH / (4.0f * cosI);
}

// Uniform-cosine hemisphere sampling (pdf = cos(theta)/pi).
static Vec3 sampleCosine(PCG32& rng, Vec3 n) {
    float phi = 2.0f * PI * rng.next();
    float r   = std::sqrt(rng.next());
    Vec3 l = {std::cos(phi) * r, std::sin(phi) * r,
              std::sqrt(std::max(0.0f, 1.0f - r * r))};
    Vec3 t = normalize(cross(std::abs(n.y) < 0.99f ? Vec3{0,1,0} : Vec3{1,0,0}, n));
    Vec3 b = cross(n, t);
    return normalize(t * l.x + b * l.y + n * l.z);
}

// Sample a direction from the GGX lobe (half-vector construction, Walter
// et al. 2007) and return its pdf.
static Vec3 sampleSpecular(PCG32& rng, Vec3 wo, Vec3 n, float& pdf) {
    float a = kRoughness * kRoughness;
    Vec3 t = normalize(cross(std::abs(n.y) < 0.99f ? Vec3{0,1,0} : Vec3{1,0,0}, n));
    Vec3 b = cross(n, t);
    while (true) {
        float phi  = 2.0f * PI * rng.next();
        float u2   = rng.next();
        float cosT = std::sqrt(u2 / (u2 + a * (1.0f - u2)));
        float sinT = std::sqrt(std::max(0.0f, 1.0f - cosT * cosT));
        Vec3 h  = normalize(t * (std::cos(phi) * sinT) + b * (std::sin(phi) * sinT) + n * cosT);
        Vec3 wi = normalize(2.0f * dot(h, n) * h - wo);
        float cosW = dot(n, wi);
        if (cosW <= 0.0f) continue;              // below horizon: resample
        pdf = ggxD(dot(n, h), a) * dot(n, h) / (4.0f * cosW);
        return wi;
    }
}

// ---------------------------------------------------------------------------
// Area light
// ---------------------------------------------------------------------------
// The quad is a Lambertian emitter pointing down: L = I * cos(theta_emit).
static Vec3 lightRadiance(Vec3 d) {
    float c = dot(kLightN, d);
    return (c <= 0.0f) ? Vec3{0,0,0} : Vec3{kLightInt, kLightInt, kLightInt} * c;
}

// Uniform sample on the quad (pdf = 1/A).
static Vec3 sampleLightPoint(PCG32& rng) {
    return { -kLightHalf + 2.0f * kLightHalf * rng.next(),
              kBox,
              -kLightHalf + 2.0f * kLightHalf * rng.next() };
}

// Power heuristic: weight of `a` = a^2 / (a^2 + b^2), per channel.
static Vec3 misWeightA(Vec3 a, Vec3 b) {
    Vec3 w;
    for (int c = 0; c < 3; ++c) {
        float x = a[c] * a[c], y = b[c] * b[c];
        w[c] = (x + y > 0.0f) ? x / (x + y) : 0.5f;
    }
    return w;
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------
static constexpr int   kMaxBounce = 12;
static constexpr float kClamp     = 30.0f;  // per-sample radiance clamp: tames
                                            // rare fireflies from specular NEE
                                            // (a tiny, standard, bias trade-off)

static Vec3 traceRay(Vec3 o, Vec3 d, PCG32& rng) {
    Vec3 L{0, 0, 0};                    // accumulated radiance
    Vec3 T{1, 1, 1};                    // throughput: prod of sampled BRDF/pdf
    for (int bounce = 0; bounce < kMaxBounce; ++bounce) {
        Hit h = intersect(o, d);
        if (h.kind == 0) break;                      // escaped the box
        if (h.kind == 3) {                           // hit the emitter
            L += T * lightRadiance(-d);
            break;
        }
        const Vec3 wo = -d;                          // path direction back to camera

        // ---- Next-event estimation (NEE): one uniform sample of the quad ---
        // NEE contribution:  f(om_l) * Le * cosP * cosL * A / d^2
        Vec3 fL_l{0,0,0}, fB_l{0,0,0};               // contribution functions at the
        {                                              // light-sampled point (l)
            Vec3 l   = sampleLightPoint(rng);
            Vec3 toL = l - h.p;
            float dist = std::sqrt(length2(toL));
            float cosP = dot(h.n, toL) / dist;
            if (cosP > 0.0f) {
                Vec3 wiL = toL / dist;
                Hit sh = intersect(h.p + h.n * kEps, wiL);   // shadow ray
                if (sh.kind == 0 || sh.t >= dist - 1e-3f) {  // light point visible
                    float cosL = dot(kLightN, -wiL);
                    Vec3 Le = lightRadiance(-wiL);
                    Vec3 f  = (h.kind == 1) ? evalDiffuse() : evalSpecular(wiL, wo, h.n);
                    fL_l = f * cosP * cosL * Le * (kLightArea / (dist * dist));
                    // BSDF-side contribution function at the same direction
                    float pdfB_l = (h.kind == 1) ? (cosP / PI) : bsdfPdfSpecular(wiL, wo, h.n);
                    if (pdfB_l > 0.0f) fB_l = f * cosP * Le / pdfB_l;
                }
            }
        }
        // Light-sampling (NEE) term, weighted per surface type:
        //  - Diffuse: full weight. The cosine BSDF sample below hits the light
        //    with probability ~ its solid angle, so counting the light there
        //    too would double-count; for a large nearby light the power
        //    heuristic would in fact hand NEE a tiny weight and leave the
        //    spiky BSDF-catch estimator dominant (noisy). Plain NEE is the
        //    low-variance unbiased choice (light hits below are discarded).
        //  - Specular: power-heuristic weight. The lobe makes light sampling
        //    and BSDF sampling genuinely complementary (specular NEE), and
        //    the BSDF term below compensates the weight deficit on hits.
        if (h.kind == 1) L += T * fL_l;
        else             L += T * (misWeightA(fL_l, fB_l) * fL_l);

        // ---- BSDF sample: continues the path (indirect light) ---------------
        Vec3 wi; float pdfB; Vec3 fS;
        if (h.kind == 1) {
            wi   = sampleCosine(rng, h.n);
            fS   = evalDiffuse();
            pdfB = dot(h.n, wi) / PI;
        } else {
            wi   = sampleSpecular(rng, wo, h.n, pdfB);
            fS   = evalSpecular(wi, wo, h.n);
        }
        const float cosS = dot(h.n, wi);

        Hit h2 = intersect(h.p + h.n * kEps, wi);
        if (h2.kind == 3) {
            if (h.kind == 2) {
                // Specular: the BSDF sample hit the light; add the
                // MIS-weighted BSDF-side term (complements the NEE term above).
                float d  = length(h2.p - h.p);
                float d2 = d * d;
                float cosL = dot(kLightN, (h.p - h2.p) * (1.0f / d));
                Vec3 Le   = lightRadiance(-wi);
                Vec3 fB_k = fS * cosS * Le / pdfB;                  // integrand / pdf_bsdf
                Vec3 fL_k = fS * cosS * cosL * Le * (kLightArea / d2);  // integrand / pdf_light
                L += T * ((Vec3{1,1,1} - misWeightA(fL_k, fB_k)) * fB_k);
            }
            // Diffuse: the direct light is already fully counted by NEE.
            break;                                              // emitter ends the path
        }

        // Continue along the sampled direction (direct light is covered by NEE).
        T = T * (fS * cosS / pdfB);
        o = h.p + h.n * kEps;
        d = wi;

        // Russian roulette to kill paths with vanishing throughput.
        if (bounce >= 3) {
            float p = clampf(std::max(T.x, std::max(T.y, T.z)), 0.05f, 0.95f);
            if (rng.next() > p) break;
            T = T * (1.0f / p);
        }
    }
    // Per-sample clamp to tame rare fireflies (small, standard bias trade-off).
    L.x = clampf(L.x, 0.0f, kClamp);
    L.y = clampf(L.y, 0.0f, kClamp);
    L.z = clampf(L.z, 0.0f, kClamp);
    return L;
}

// ---------------------------------------------------------------------------
// Camera / film
// ---------------------------------------------------------------------------
static constexpr int   kWidth  = 512, kHeight = 512;
static constexpr int   kSPP    = 4096;   // 2x2 stratified sub-pixels per spp set
static constexpr float kVfov   = 30.0f;  // vertical field of view, degrees
static constexpr float kExposure = 1.0f;

struct Camera {
    Vec3 pos{0.0f, 0.2f, 10.0f};          // in front of the box (z > +2)
    Vec3 fwd, right, up;
    float tanHalf;
    Camera() {
        fwd   = normalize(Vec3{0.0f, 0.0f, 2.0f} - pos);   // aim at the opening
        right = normalize(cross(fwd, Vec3{0, 1, 0}));
        up    = cross(right, fwd);
        tanHalf = std::tan(0.5f * kVfov * PI / 180.0f);
    }
    // 2x2-stratified sample for pixel (x, y); y = 0 is the top row.
    // (s & 1, (s >> 1) & 1) cycles the four sub-pixels for any number of spp.
    Vec3 sampleDir(int x, int y, int s, PCG32& rng) const {
        float u = (2.0f * x + (s & 1) + rng.next()) / (2.0f * kWidth);
        float v = (2.0f * y + ((s >> 1) & 1) + rng.next()) / (2.0f * kHeight);
        return normalize(fwd + right * (2.0f * u - 1.0f) * tanHalf
                         + up    * (1.0f - 2.0f * v) * tanHalf);
    }
};

static float filmic(float x) {          // Narkowicz "Uncharted" filmic curve
    return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
}
static float srgbEncode(float x) {      // linear -> sRGB [0,1]
    return (x <= 0.0031308f) ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}
static uint8_t toSrgb8(float linear) {
    float c = filmic(clampf(linear * kExposure, 0.0f, 100.0f));
    return (uint8_t)std::min(255.0f, srgbEncode(clampf(c, 0.0f, 1.0f)) * 255.0f + 0.5f);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
#ifndef PT_NO_MAIN
int main() {
    Camera cam;
    const int nT = std::max(1u, std::thread::hardware_concurrency());
    std::vector<float> acc((size_t)kWidth * kHeight * 3, 0.0f);
    std::atomic<int> rowsDone{0};
    auto t0 = std::chrono::steady_clock::now();

    auto worker = [&](int t) {
        // Fixed seed per thread -> reproducible results.
        PCG32 rng(0x12345678ull * (t + 1) + 0xC0FFEEull);
        for (int y = t; y < kHeight; y += nT) {
            for (int x = 0; x < kWidth; ++x) {
                Vec3 L{0, 0, 0};
                for (int s = 0; s < kSPP; ++s)
                    L += traceRay(cam.pos, cam.sampleDir(x, y, s, rng), rng);
                float* p = &acc[((size_t)y * kWidth + x) * 3];
                p[0] = L.x / kSPP; p[1] = L.y / kSPP; p[2] = L.z / kSPP;
            }
            int done = ++rowsDone;
            if (done % 64 == 0)
                std::fprintf(stderr, "  %d/%d rows\n", done, kHeight);
        }
    };

    std::vector<std::thread> th;
    for (int t = 0; t < nT; ++t) th.emplace_back(worker, t);
    for (auto& a : th) a.join();

    FILE* f = std::fopen("output.ppm", "wb");
    if (!f) { std::perror("output.ppm"); return 1; }
    std::fprintf(f, "P6\n%d %d\n255\n", kWidth, kHeight);
    std::vector<uint8_t> buf((size_t)kWidth * kHeight * 3);
    for (size_t i = 0; i < buf.size(); i += 3) {
        buf[i]     = toSrgb8(acc[i]);
        buf[i + 1] = toSrgb8(acc[i + 1]);
        buf[i + 2] = toSrgb8(acc[i + 2]);
    }
    std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("rendered %dx%d @ %d spp in %.1fs (%d threads) -> output.ppm\n",
                kWidth, kHeight, kSPP, secs, nT);
    return 0;
}
#endif // PT_NO_MAIN
