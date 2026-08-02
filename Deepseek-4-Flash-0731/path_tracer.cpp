// ================================================================
//  A small, clearly-commented path tracer with direct lighting
//  (Next Event Estimation / NEE) and sRGB-correct output.
//
//  Scene: a specular (perfect-mirror) sphere of radius 1 inside a
//  diffuse box of radius 2. The box's front wall (z = +2) is open;
//  the camera looks in from the front. A white quad light hangs
//  from the ceiling, pointing straight down.
//
//  Output: output.ppm -- 512x512, 8-bit/channel, sRGB-encoded P6.
//
//  Build:  g++ -O3 -fopenmp path_tracer.cpp -o pt
//  Run:    ./pt [spp]     (default: 64 samples per pixel)
// ================================================================
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>

static const float PI = 3.14159265f;

// ---------------------------------------------------------------
// 3D vector (float)
// ---------------------------------------------------------------
struct Vec3 {
    float x, y, z;
    Vec3() {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& b) const { return Vec3(x+b.x, y+b.y, z+b.z); }
    Vec3 operator-(const Vec3& b) const { return Vec3(x-b.x, y-b.y, z-b.z); }
    Vec3 operator*(const Vec3& b) const { return Vec3(x*b.x, y*b.y, z*b.z); } // component-wise
    Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(float s) const { return Vec3(x/s, y/s, z/s); }
    Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
    Vec3& operator*=(const Vec3& b) { x*=b.x; y*=b.y; z*=b.z; return *this; }
    Vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
    Vec3& operator/=(float s) { x/=s; y/=s; z/=s; return *this; }
    float dot(const Vec3& b) const { return x*b.x + y*b.y + z*b.z; }
    Vec3 cross(const Vec3& b) const {
        return Vec3(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
    }
    Vec3 norm() const {
        float l = sqrtf(x*x + y*y + z*z);
        return Vec3(x/l, y/l, z/l);
    }
    float maxComp() const { return fmaxf(x, fmaxf(y, z)); }
};
inline Vec3 operator*(float s, const Vec3& v) { return v*s; }

struct Ray { Vec3 o, d; Ray() {} Ray(const Vec3& o_, const Vec3& d_) : o(o_), d(d_) {} };

// ---------------------------------------------------------------
// Tiny xorshift64* RNG: fast, deterministic, good enough
// ---------------------------------------------------------------
struct RNG {
    uint64_t s;
    RNG(uint64_t seed) { s = seed ? seed : 0x9E3779B97F4A7C15ULL; }
    uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
    float u() { return (float)(next() >> 40) * (1.f/16777216.f); } // [0,1)
};

// ---------------------------------------------------------------
// Materials
// ---------------------------------------------------------------
struct Material {
    Vec3  albedo;    // diffuse color, or tint multiplier for a mirror
    Vec3  emission;  // > 0  =>  area light
    bool  specular;  // true  =>  perfect mirror (delta BSDF)
};

// Geometry
struct Sphere { Vec3 c; float r; int m; };
struct Quad   { Vec3 c, u, v, n; float area; int m; }; // parallelogram: c ± u/2 ± v/2, n = normalize(u×v)

struct Scene {
    std::vector<Material> mats;
    std::vector<Sphere> spheres;
    std::vector<Quad>   quads;
    std::vector<int>    lights;    // quad indices with emission > 0
};

struct Hit { float t; Vec3 p, n; int m; };

// Quad factory: computes the outward-facing normal (u×v) and area
Quad makeQuad(const Vec3& c, const Vec3& u, const Vec3& v, int m) {
    Vec3 uv = u.cross(v);
    Quad q; q.c = c; q.u = u; q.v = v;
    q.n = uv.norm(); q.area = sqrtf(uv.dot(uv)); q.m = m;
    return q;
}

// ---------------------------------------------------------------
// Ray intersections
// ---------------------------------------------------------------

// Ray vs sphere (nearest positive root)
bool hitSphere(const Sphere& s, const Ray& r, float tmin, float tmax, Hit& h) {
    Vec3 oc = r.o - s.c;
    float b = oc.dot(r.d);
    float c = oc.dot(oc) - s.r*s.r;
    float disc = b*b - c;
    if (disc <= 0.f) return false;
    float t = -b - sqrtf(disc);                       // near root
    if (t < tmin || t > tmax) {                       // try far root
        t = -b + sqrtf(disc);
        if (t < tmin || t > tmax) return false;
    }
    h.t = t;
    h.p = r.o + r.d*t;
    h.n = (h.p - s.c) * (1.f/s.r);                    // unit sphere normal
    h.m = s.m;
    return true;
}

// Ray vs parallelogram (single-sided). u and v are orthogonal axes
bool hitQuad(const Quad& q, const Ray& r, float tmin, float tmax, Hit& h) {
    float den = r.d.dot(q.n);
    if (den > -1e-8f) return false;                   // only front faces
    float t = (q.c - r.o).dot(q.n) / den;
    if (t < tmin || t > tmax) return false;
    Vec3 d = r.o + r.d*t - q.c;
    float uu = d.dot(q.u) / q.u.dot(q.u);             // barycentric in [−½, ½]
    float vv = d.dot(q.v) / q.v.dot(q.v);
    if (uu < -0.5f || uu > 0.5f || vv < -0.5f || vv > 0.5f) return false;
    h.t = t; h.p = r.o + r.d*t; h.n = q.n; h.m = q.m;
    return true;
}

// Closest hit over the whole scene
bool closestHit(const Scene& s, const Ray& r, float tmin, float tmax, Hit& h) {
    float best = tmax; bool any = false;
    for (const Sphere& sp : s.spheres) { Hit tmp; if (hitSphere(sp, r, tmin, best, tmp)) { best = tmp.t; h = tmp; any = true; } }
    for (const Quad&  q  : s.quads)   { Hit tmp; if (hitQuad  (q,  r, tmin, best, tmp)) { best = tmp.t; h = tmp; any = true; } }
    return any;
}

// Shadow test: is anything opaque between a surface point and the light?
bool occluded(const Scene& s, const Ray& r, float maxT) {
    Hit tmp;
    for (const Sphere& sp : s.spheres) if (hitSphere(sp, r, 1e-4f, maxT, tmp)) return true;
    for (const Quad&  q  : s.quads)
        if (s.mats[q.m].emission.maxComp() == 0.f && hitQuad(q, r, 1e-4f, maxT, tmp)) return true; // lights never shadow
    return false;
}

// ---------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------

// NEE: sample a point on each area light and add its direct
// contribution. Estimator for pdf = 1/area:
//   f(ω)·L_e·A·cosθ·cosθl / d²   (the area cancels against the pdf)
Vec3 directLight(const Scene& s, const Hit& h, const Vec3& albedo, RNG& rng) {
    Vec3 L(0.f, 0.f, 0.f);
    for (int li : s.lights) {
        const Quad& q = s.quads[li];
        Vec3 p = q.c + (rng.u()-0.5f)*q.u + (rng.u()-0.5f)*q.v; // uniform point on the light
        Vec3 w = p - h.p;
        float d2 = w.dot(w);
        w = w.norm();
        float cosT =  h.n.dot(w);    // cosine at the receiving surface
        float cosL = -q.n.dot(w);    // cosine at the light
        if (cosT <= 0.f || cosL <= 0.f) continue;                     // back-facing sample
        if (occluded(s, Ray(h.p + h.n*1e-4f, w), sqrtf(d2)-1e-4f)) continue; // shadow ray
        // Lambert BRDF × emitted radiance × geometry term:
        L += albedo * (1.f/PI) * s.mats[q.m].emission * q.area * cosT * cosL / d2;
    }
    return L;
}

// Cosine-weighted hemisphere sample: cosθ = √r₁, φ = 2π·r₂ (pdf = cosθ/π)
Vec3 cosSample(const Vec3& n, RNG& rng) {
    float r1 = rng.u(), r2 = rng.u();
    float ct = sqrtf(r1), st = sqrtf(1.f-r1), phi = 2.f*PI*r2;
    Vec3 t = (fabsf(n.y) < 0.99f) ? Vec3(0.f,1.f,0.f).cross(n)
                                  : Vec3(1.f,0.f,0.f).cross(n);
    t = t.norm();
    Vec3 b = n.cross(t);
    return (t*(cosf(phi)*st) + b*(sinf(phi)*st) + n*ct).norm();
}

Vec3 reflect(const Vec3& d, const Vec3& n) { return d - n*(2.f*d.dot(n)); }

// ---------------------------------------------------------------
// Path trace: NEE at diffuse vertices, mirror recursion at the
// sphere, Russian-roulette termination, 8-bounce depth limit.
// ---------------------------------------------------------------
Vec3 radiance(const Scene& s, const Ray& ray, RNG& rng) {
    Vec3 L(0.f, 0.f, 0.f), T(1.f, 1.f, 1.f); // accumulated radiance & throughput
    bool spec = true;  // the camera is a "specular" vertex, so a direct view of
                       // the light is valid; diffuse paths already got it via NEE
    Ray r = ray;
    for (int bounce = 0; bounce < 8; bounce++) {
        Hit h;
        if (!closestHit(s, r, 1e-4f, 1e30f, h)) break;      // escaped through the open front
        const Material& m = s.mats[h.m];
        if (m.emission.maxComp() > 0.f) {                   // hit the light surface
            if (spec) L += T * m.emission;                  // only specular chains see it
            break;                                          // directly (avoids double counting)
        }
        if (m.specular) {                                   // perfect mirror: reflect
            T *= m.albedo;
            r = Ray(h.p + h.n*1e-4f, reflect(r.d, h.n));
            spec = true;
            continue;
        }
        // ---- diffuse surface ----
        L += T * directLight(s, h, m.albedo, rng);          // explicit direct lighting (NEE)
        Vec3 d = cosSample(h.n, rng);                       // implicit indirect: cosine-weighted
        T *= m.albedo;                                      // f·cosθ/pdf = albedo (Lambert)
        r = Ray(h.p + h.n*1e-4f, d);
        spec = false;
        if (bounce >= 2) {                                  // Russian roulette
            float p = T.maxComp();
            if (rng.u() > p) break;
            T /= p;
        }
    }
    return L;
}

// ---------------------------------------------------------------
// Main: build the scene, render, write output.ppm
// ---------------------------------------------------------------
int main(int argc, char** argv) {
    int spp = (argc > 1) ? atoi(argv[1]) : 64;
    const int W = 512, H = 512;
    const float halfTan = tanf(0.5f * 55.f * PI/180.f);
    const Vec3 eye(0.f, 0.f, 3.4f);

    Scene s;
    // materials: 0 back wall, 1 floor, 2 ceiling, 3 left, 4 right, 5 mirror, 6 light
    s.mats.push_back(Material{Vec3(0.60f,0.35f,0.30f), Vec3(0.f,0.f,0.f), false});
    s.mats.push_back(Material{Vec3(0.55f,0.55f,0.60f), Vec3(0.f,0.f,0.f), false});
    s.mats.push_back(Material{Vec3(0.80f,0.80f,0.85f), Vec3(0.f,0.f,0.f), false});
    s.mats.push_back(Material{Vec3(0.35f,0.55f,0.35f), Vec3(0.f,0.f,0.f), false});
    s.mats.push_back(Material{Vec3(0.35f,0.45f,0.60f), Vec3(0.f,0.f,0.f), false});
    s.mats.push_back(Material{Vec3(0.95f,0.98f,1.00f), Vec3(0.f,0.f,0.f), true}); // mirror sphere
    s.mats.push_back(Material{Vec3(0.f,0.f,0.f),       Vec3(15.f,15.f,15.f), false}); // light

    // Diffuse box of radius 2 (walls are 4×4 quads). Front wall at z=+2 is open.
    s.quads.push_back(makeQuad(Vec3(-2.f,0.f,0.f), Vec3(0.f,4.f,0.f),  Vec3(0.f,0.f,4.f),  3)); // left  wall (x=-2)
    s.quads.push_back(makeQuad(Vec3( 2.f,0.f,0.f), Vec3(0.f,4.f,0.f),  Vec3(0.f,0.f,-4.f), 4)); // right wall (x=+2)
    s.quads.push_back(makeQuad(Vec3(0.f,-2.f,0.f), Vec3(4.f,0.f,0.f),  Vec3(0.f,0.f,-4.f), 1)); // floor      (y=-2)
    s.quads.push_back(makeQuad(Vec3(0.f, 2.f,0.f), Vec3(4.f,0.f,0.f),  Vec3(0.f,0.f,4.f),  2)); // ceiling    (y=+2)
    s.quads.push_back(makeQuad(Vec3(0.f,0.f,-2.f), Vec3(4.f,0.f,0.f),  Vec3(0.f,4.f,0.f),  0)); // back wall  (z=-2)

    // Quad light on the ceiling, emitting straight down
    s.quads.push_back(makeQuad(Vec3(0.f,1.99f,0.f), Vec3(1.4f,0.f,0.f), Vec3(0.f,0.f,1.4f), 6));
    s.lights.push_back((int)s.quads.size()-1);

    // Specular sphere, radius 1, at the box center
    s.spheres.push_back(Sphere{Vec3(0.f,0.f,0.f), 1.f, 5});

    // ---- render ----
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Vec3> img(W*H);
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            Vec3 col(0.f,0.f,0.f);
            for (int n = 0; n < spp; n++) {
                RNG rng(px*0x9E3779B97F4A7C15ULL + py*0xC2B2AE3D27D4EB4FULL
                      + n*0x27D4EB2F165C7869ULL + 0x9E3779B97F4A7C15ULL);
                float u = (px + rng.u())/W;                       // pixel → camera space
                float v = (py + rng.u())/H;
                Vec3 d = Vec3((2.f*u-1.f)*halfTan, (1.f-2.f*v)*halfTan, -1.f).norm();
                col += radiance(s, Ray(eye, d), rng);
            }
            img[py*W+px] = col * (1.f/spp);
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    // ---- stats (exposure sanity check) ----
    float mn = 1e9f, mx = 0.f, avg = 0.f;
    for (const Vec3& c : img) { float l = c.maxComp(); mn = fminf(mn,l); mx = fmaxf(mx,l); avg += l; }
    avg /= img.size();
    printf("rendered %dx%d, %d spp, %.2fs\n", W, H, spp,
           std::chrono::duration<float>(t1-t0).count());
    printf("  luminance: min %.3f  avg %.3f  max %.3f\n", mn, avg, mx);

    // ---- write output.ppm (P6, top-down rows) ----
    FILE* f = fopen("output.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            Vec3 c = img[py*W+px];
            unsigned char b[3];
            for (int k = 0; k < 3; k++) {
                float x = (k==0)?c.x : (k==1)?c.y : c.z;
                x = fmaxf(x, 0.f);                                // clamp negatives
                x = (x <= 0.0031308f) ? 12.92f*x                  // linear → sRGB OETF
                    : 1.055f*powf(x, 1.f/2.4f) - 0.055f;
                b[k] = (unsigned char)(fminf(x, 1.f)*255.f + 0.5f);
            }
            fwrite(b, 1, 3, f);
        }
    }
    fclose(f);
    printf("wrote output.ppm\n");
    return 0;
}
