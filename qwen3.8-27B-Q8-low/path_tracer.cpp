// ============================================================================
// path_tracer.cpp — small path tracer with NEE (next event estimation)
// ============================================================================
//
// Scene
//   • A diffuse box: the interior of a cube of half-extent 2 (i.e. spanning
//     [-2, 2] on each axis). The front wall (z = +2) is removed so the
//     camera can look in from the front.
//   • A perfectly specular (mirror) sphere of radius 1 at the origin.
//   • A quad area light attached to the ceiling (y = +2), facing down.
//
// Rendering
//   • 512×512 pixels, SPP primary samples per pixel (1 path each).
//   • NEE: every diffuse bounce directly samples the area light, adds its
//     shadow-tested contribution, and then continues with a
//     cosine-weighted hemisphere bounce for the indirect component.
//   • Output: 8-bit sRGB, gamma-encoded, written to output.ppm (P6).
//
// Build (single file, no dependencies):
//   g++  -O3 -o path_tracer path_tracer.cpp
//   cl   /O2 /EHsc path_tracer.cpp      (MSVC)
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// ------------------------------- Vec3 --------------------------------------

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    // Per-component (Schur) product — used to scale color throughputs.
    Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
};
static inline Vec3  operator*(float s, const Vec3& v) { return v * s; }
static inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3  cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static inline float vlen(const Vec3& v)  { return std::sqrt(dot(v, v)); }
static inline Vec3  vnorm(const Vec3& v) { return v * (1.f / vlen(v)); }
static inline float vmax(const Vec3& v)  { return std::max({v.x, v.y, v.z}); }

static constexpr float PI = 3.14159265358979f;

// Deterministic RNG so renders are reproducible.
static std::mt19937 g_rng(0x9e3779b9u);
static inline float randf() {
    static std::uniform_real_distribution<float> d(0.f, 1.f);
    return d(g_rng);
}

// ------------------------------- Scene -------------------------------------

enum Mat { M_NONE, M_DIFFUSE, M_MIRROR, M_LIGHT };

struct Tri {
    Vec3 a, b, c;
    Vec3 n;        // precomputed normal (faces used two-sided)
    Mat  mat;
    Vec3 alb;      // diffuse albedo, or light radiance when mat == M_LIGHT
};

struct Ray  { Vec3 o, d; };
struct Hit  {
    float t = 1e30f;
    Vec3  n, alb;
    Mat   mat = M_NONE;
};

static std::vector<Tri> g_tris;

// Scene parameters.
static constexpr float BOX_HALF   = 2.0f;   // box spans [-2,2]^3 (front wall removed)
static constexpr float SPHERE_R   = 1.0f;   // mirror sphere radius
static constexpr float LIGHT_HALF = 0.75f;  // light quad is 1.5 × 1.5
static const Vec3     LIGHT_CENTER (0.f, BOX_HALF - 1e-3f, 0.f); // on ceiling
static const Vec3     LIGHT_NORMAL (0.f, -1.f, 0.f);             // faces down
static const Vec3     LIGHT_RADIANCE = 8.f * Vec3(1, 1, 1);      // tunable exposure knob
static const float    MIRROR_ALBEDO  = 0.95f;

// Split a quad into two triangles with a shared precomputed normal.
static void addQuad(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3,
                    Mat mat, Vec3 alb) {
    Vec3 n = vnorm(cross(p1 - p0, p2 - p0));
    g_tris.push_back({p0, p1, p2, n, mat, alb});
    g_tris.push_back({p0, p2, p3, n, mat, alb});
}

static void buildScene() {
    const float H = BOX_HALF;
    const Vec3 wall(0.90f, 0.90f, 0.90f), floorC(0.85f, 0.85f, 0.85f);

    // Box walls. Note: the front wall at z = +H is deliberately omitted —
    // that is the open side the camera looks through.
    addQuad({-H,-H,-H}, { H,-H,-H}, { H,-H, H}, {-H,-H, H}, M_DIFFUSE, floorC); // floor
    addQuad({-H, H,-H}, { H, H,-H}, { H, H, H}, {-H, H, H}, M_DIFFUSE, wall);   // ceiling
    addQuad({-H,-H,-H}, {-H, H,-H}, { H, H,-H}, { H,-H,-H}, M_DIFFUSE, wall);   // back
    addQuad({-H,-H,-H}, {-H, H,-H}, {-H, H, H}, {-H,-H, H}, M_DIFFUSE, wall);   // left
    addQuad({ H,-H,-H}, { H, H,-H}, { H, H, H}, { H,-H, H}, M_DIFFUSE, wall);   // right

    // Area light: flat quad on the ceiling, emitting from its front face.
    const float y = LIGHT_CENTER.y;
    addQuad({-LIGHT_HALF, y, -LIGHT_HALF},
            { LIGHT_HALF, y, -LIGHT_HALF},
            { LIGHT_HALF, y,  LIGHT_HALF},
            {-LIGHT_HALF, y,  LIGHT_HALF},
            M_LIGHT, LIGHT_RADIANCE);
}

// Möller–Trumbore triangle intersection. Returns t or -1 on miss.
// The normal is flipped so it always faces the ray (two-sided faces).
static float hitTri(const Ray& r, const Tri& tri, Vec3& n) {
    constexpr float EPS = 1e-4f;
    Vec3 e1 = tri.b - tri.a, e2 = tri.c - tri.a;
    Vec3 p  = cross(r.d, e2);
    float det = dot(e1, p);
    if (std::fabs(det) < 1e-6f) return -1.f;
    float invDet = 1.f / det;
    Vec3 tv = r.o - tri.a;
    float u = dot(tv, p) * invDet;
    if (u < -EPS || u > 1.f + EPS) return -1.f;
    Vec3 q = cross(tv, e1);
    float v = dot(r.d, q) * invDet;
    if (v < -EPS || u + v > 1.f + EPS) return -1.f;
    float t = dot(e2, q) * invDet;
    if (t < 1e-3f) return -1.f;
    n = (dot(tri.n, r.d) < 0.f) ? tri.n : -tri.n;
    return t;
}

// Nearest hit against the sphere and all triangles.
static Hit intersect(const Ray& r) {
    Hit best;
    // Mirror sphere at the origin: solve |o + t d|² = R².
    {
        float b = dot(r.o, r.d);
        float c = dot(r.o, r.o) - SPHERE_R * SPHERE_R;
        float disc = b * b - c;
        if (disc > 0.f) {
            float t = -b - std::sqrt(disc);
            if (t > 1e-3f && t < best.t) {
                best.t = t;
                best.mat = M_MIRROR;
                best.n = (r.o + r.d * t) * (1.f / SPHERE_R);
                best.alb = MIRROR_ALBEDO * Vec3(1, 1, 1);
            }
        }
    }
    for (const Tri& tri : g_tris) {
        Vec3 n;
        float t = hitTri(r, tri, n);
        if (t > 0.f && t < best.t) {
            best.t = t;
            best.n = n;
            best.mat = tri.mat;
            best.alb = tri.alb;
        }
    }
    return best;
}

// ---------------------------- Sampling / NEE --------------------------------

// Uniform random point on the light quad (a flat quad in the x–z plane,
// centered at LIGHT_CENTER with half-extent LIGHT_HALF).
static Vec3 sampleLightPoint() {
    return { LIGHT_CENTER.x + (2.f * randf() - 1.f) * LIGHT_HALF,
             LIGHT_CENTER.y,
             LIGHT_CENTER.z + (2.f * randf() - 1.f) * LIGHT_HALF };
}

// Cosine-weighted random direction in the hemisphere around n.
static Vec3 sampleCosineHemi(const Vec3& n) {
    float r = std::sqrt(randf());
    float phi = 2.f * PI * randf();
    Vec3 local = {r * std::cos(phi), r * std::sin(phi), std::sqrt(1.f - r * r)};
    // Orthonormal basis (t1, t2) around n (avoid a near-parallel axis).
    Vec3 t1 = (std::fabs(n.x) < 0.9f) ? vnorm(cross(n, Vec3(1, 0, 0)))
                                      : vnorm(cross(n, Vec3(0, 1, 0)));
    Vec3 t2 = cross(n, t1);
    return vnorm(t1 * local.x + t2 * local.y + n * local.z);
}

// Trace one path. NEE is applied at every diffuse surface.
static Vec3 trace(const Ray& r0) {
    Ray  r = r0;
    Vec3 through(1, 1, 1);   // running product of BSDF terms so far
    Vec3 total;              // accumulated radiance along the path
    constexpr int MAX_BOUNCE = 6;

    for (int b = 0; b < MAX_BOUNCE; ++b) {
        Hit h = intersect(r);
        if (h.mat == M_NONE) break;      // escaped out the open front: black
        if (h.mat == M_LIGHT) {          // path landed on the light itself
            total += through * h.alb;
            break;
        }

        Vec3 p = r.o + r.d * h.t;        // hit point

        if (h.mat == M_MIRROR) {
            // Perfectly specular: mirror-reflect and continue. (No NEE is
            // needed here; the light is captured directly when the reflected
            // ray lands on it via the M_LIGHT case above.)
            through = through * h.alb;
            r = { p, r.d - 2.f * dot(r.d, h.n) * h.n };
        } else { // M_DIFFUSE
            // ---- NEE: directly sample the area light ----------------------
            Vec3 lp = sampleLightPoint();
            Vec3 lv = lp - (p + h.n * 1e-3f);   // offset to avoid self-hit
            float dist2 = dot(lv, lv);
            float ndl = dot(h.n, lv);
            if (ndl > 0.f && dist2 > 0.f) {
                // Shadow test: the light counts only if the nearest thing on
                // the segment from the surface to the sample point is the
                // light itself.
                float dist = std::sqrt(dist2);
                Hit s = intersect({p + h.n * 1e-3f, lv * (1.f / dist)});
                if (s.mat == M_LIGHT && s.t < dist * 1.001f + 1e-3f) {
                    // Lambertian BRDF (albedo/pi) × Le × geometric factor
                    // (ndl·ndv/d²) × light area A (uniform area sampling pdf
                    // = 1/A) × |nL·v| (light emits from its front face only).
                    float ndv = dot(h.n, lv * (1.f / dist));
                    float nlv = dot(LIGHT_NORMAL, -lv * (1.f / dist));
                    if (nlv > 0.f) {
                        float area = 4.f * LIGHT_HALF * LIGHT_HALF;
                        total += (h.alb * (1.f / PI)) * LIGHT_RADIANCE * area
                               * (ndl * ndv * nlv / dist2) * through;
                    }
                }
            }
            // ---- Indirect: continue with a cosine-weighted bounce ---------
            r = { p + h.n * 1e-3f, sampleCosineHemi(h.n) };
            through = through * (h.alb * (1.f / PI));
        }

        // Russian roulette: after a few bounces, terminate with probability
        // 1 - max(throughput) and renormalize the remaining contribution.
        if (b >= 2) {
            float pKeep = std::max(vmax(through), 1e-4f);
            if (randf() > pKeep) break;
            through = through * (1.f / pKeep);
        }
    }
    return total;
}

// ------------------------------- main --------------------------------------

int main() {
    buildScene();

    constexpr int W = 512, H = 512;    // output resolution (square)
    constexpr int SPP = 128;           // primary samples per pixel
    constexpr float EXPOSURE = 1.0f;

    // Camera: outside the box on the open +z side, aimed at the origin.
    Vec3 ro(0.f, 0.3f, 7.0f), ta(0.f, 0.f, 0.f);
    Vec3 w = vnorm(ta - ro);                       // forward
    Vec3 u = vnorm(cross(w, Vec3(0, 1, 0)));       // right
    Vec3 v = cross(u, w);                          // up
    float ppm = 2.f * std::tan(22.5f * PI / 180.f);  // 45° vertical FOV:
    // full-frame lateral extent in world units at unit focal distance.

    std::vector<Vec3> img(W * H);

    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            Vec3 sum;
            for (int s = 0; s < SPP; ++s) {
                // Jittered sample inside the pixel; v flipped so py=0 is top.
                float uu = (px + randf()) / W;
                float vv = 1.f - (py + randf()) / H;
                // Pinhole model: rd = forward + lateral offset on the focal plane.
                Vec3 rd = vnorm(u * (uu - 0.5f) * ppm
                              + v * (vv - 0.5f) * ppm
                              + w);
                sum = sum + trace({ro, rd});
            }
            img[py * W + px] = sum * (1.f / SPP);  // mean over samples
        }
        std::printf("row %d/%d\r", py + 1, H);
        std::fflush(stdout);
    }
    std::printf("\n");

    // Encode to 8-bit sRGB (exact piecewise sRGB curve) and write a P6 PPM.
    auto toU8 = [EXPOSURE](float x) -> unsigned char {
        x = std::clamp(x * EXPOSURE, 0.f, 1.f);
        x = (x <= 0.0031308f) ? 12.92f * x : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
        return (unsigned char)std::lround(x * 255.f);
    };

    std::FILE* f = std::fopen("output.ppm", "wb");
    if (!f) { std::perror("output.ppm"); return 1; }
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H; ++i) {
        unsigned char px[3] = {toU8(img[i].x), toU8(img[i].y), toU8(img[i].z)};
        std::fwrite(px, 1, 3, f);
    }
    std::fclose(f);
    std::printf("wrote output.ppm\n");
    return 0;
}
