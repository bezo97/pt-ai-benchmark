// =============================================================================
//  pathtracer.cpp — compact C++17 path tracer with NEE direct lighting
// =============================================================================
//  Scene:
//    * Perfectly specular (mirror) sphere of radius 1 at the origin.
//    * Diffuse box [-2, 2]^3, open on the front face (z = -2) where the
//      camera sits, outside the box.
//    * 1.5 x 1.5 quad area light on the ceiling (y = +2), facing straight
//      down.
//  Output: output.ppm — 512 x 512, 8-bit sRGB.
//
//  Build:  g++ -O2 -std=c++17 -o pathtracer pathtracer.cpp
//  Run:    ./pathtracer [spp]        (default 256 samples per pixel)
//
//  Integration strategy:
//    * On every diffuse hit, one uniform sample on the light quad plus a
//      shadow ray yields the *direct* lighting term (next event estimation).
//    * The path then continues with a cosine-weighted bounce, which covers
//      the *indirect* (bounced) term.
//    * The mirror sphere is deterministic: a finite area light aligns with
//      its reflection direction only on a set of measure zero, so NEE adds
//      nothing there — instead, reflection rays that land on the light quad
//      produce its sharp specular highlight.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static constexpr double PI = 3.14159265358979323846;
static constexpr int    MAX_BOUNCES = 8;

// ---------------------------------------------------------------------------
//  Vec3 — minimal 3-component vector
// ---------------------------------------------------------------------------
struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3  operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3  operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3  operator*(double s)      const { return {x * s,   y * s,   z * s};   }
    Vec3  operator/(double s)      const { return {x / s,   y / s,   z / s};   }
    Vec3& operator+=(const Vec3& o){ x += o.x; y += o.y; z += o.z; return *this; }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3  cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double lengthSq() const { return dot(*this); }
    double length()   const { return std::sqrt(lengthSq()); }
    Vec3  normalized() const { double l = length(); return l > 0.0 ? *this / l : *this; }
};
static Vec3 operator*(double s, const Vec3& v) { return v * s; }

// ---------------------------------------------------------------------------
//  Ray and hit record
// ---------------------------------------------------------------------------
struct Ray {
    Vec3 o, d;  // origin, (unit) direction
};

enum class Material { Diffuse, Mirror, Light };

struct Surface {
    double t = 0;
    Vec3 p, n;      // hit point, geometric (inward) normal
    Material mat = Material::Diffuse;
    Vec3 albedo;    // Diffuse only
    Vec3 emission;  // Light only
};

// ---------------------------------------------------------------------------
//  Scene definition (all constants)
// ---------------------------------------------------------------------------
static const Vec3   SPHERE_C{0, 0, 0};
static const double SPHERE_R = 1.0;

static const double BOX_HALF = 2.0;   // box is [-2, 2]^3; front face z=-2 is open
static const Vec3   BOX_ALBEDO{0.9, 0.9, 0.9};

static const Vec3   LIGHT_POS{0, BOX_HALF, 0};  // center of the quad, on ceiling
static const double LIGHT_SIZE = 1.5;           // quad is LIGHT_SIZE x LIGHT_SIZE
static const Vec3   LIGHT_N{0, -1, 0};          // emission direction: straight down
static const Vec3   LIGHT_COL{8.0, 7.6, 7.0};   // uniform radiance (slightly warm)
static const double LIGHT_AREA = LIGHT_SIZE * LIGHT_SIZE;

// ---------------------------------------------------------------------------
//  Intersections
// ---------------------------------------------------------------------------
static double comp(const Vec3& v, int a) { return a == 0 ? v.x : a == 1 ? v.y : v.z; }

// Mirror sphere.
static bool intersectSphere(const Ray& r, Surface& s) {
    Vec3 oc = r.o - SPHERE_C;
    double b = 2.0 * oc.dot(r.d);
    double c = oc.lengthSq() - SPHERE_R * SPHERE_R;
    double disc = b * b - 4.0 * c;
    if (disc < 0.0) return false;
    double t = (-b - std::sqrt(disc)) * 0.5;
    if (t < 1e-4) return false;
    s.t = t;
    s.p = r.o + r.d * t;
    s.n = (s.p - SPHERE_C).normalized();
    s.mat = Material::Mirror;
    return true;
}

// Diffuse box built as five inward-facing planes (front face z = -BOX_HALF is
// open, so it is omitted). Each candidate hit must also lie within the finite
// quad of its face, which rejects rays that miss the box through the opening.
static bool intersectBox(const Ray& r, Surface& s) {
    struct Face { int axis; double sgn; };  // plane comp(axis) = sgn * BOX_HALF
    static const Face faces[5] = {
        {0, +1},  // x = +2, normal -X
        {0, -1},  // x = -2, normal +X
        {1, +1},  // y = +2 (ceiling), normal -Y
        {1, -1},  // y = -2 (floor), normal +Y
        {2, +1},  // z = +2 (back), normal -Z
    };

    double best = 1e30;
    for (const Face& f : faces) {
        double o = comp(r.o, f.axis), d = comp(r.d, f.axis);
        if (std::fabs(d) < 1e-12) continue;            // parallel to the plane
        double t = (f.sgn * BOX_HALF - o) / d;
        if (t < 1e-4 || t >= best) continue;

        Vec3 p = r.o + r.d * t;
        if (std::fabs(comp(p, (f.axis + 1) % 3)) > BOX_HALF + 1e-9) continue;
        if (std::fabs(comp(p, (f.axis + 2) % 3)) > BOX_HALF + 1e-9) continue;

        best = t;
        s.t = t;
        s.p = p;
        Vec3 n{0, 0, 0};
        if (f.axis == 0) n.x = -f.sgn;
        if (f.axis == 1) n.y = -f.sgn;
        if (f.axis == 2) n.z = -f.sgn;
        s.n = n;
        s.mat = Material::Diffuse;
        s.albedo = BOX_ALBEDO;
    }
    return best < 1e30;
}

// Quad light on the ceiling (hittable, so reflection rays can hit it and the
// mirror sphere shows its specular highlight).
static bool intersectLight(const Ray& r, Surface& s) {
    if (std::fabs(r.d.y) < 1e-12) return false;
    double t = (LIGHT_POS.y - r.o.y) / r.d.y;
    if (t < 1e-4) return false;
    Vec3 p = r.o + r.d * t;
    double h = LIGHT_SIZE * 0.5;
    if (std::fabs(p.x - LIGHT_POS.x) > h) return false;
    if (std::fabs(p.z - LIGHT_POS.z) > h) return false;
    s.t = t;
    s.p = p;
    s.n = LIGHT_N;
    s.mat = Material::Light;
    s.emission = LIGHT_COL;
    return true;
}

static bool intersectWorld(const Ray& r, Surface& s) {
    bool hit = false;
    Surface c;
    auto consider = [&](Surface& cand) {
        if (!hit || cand.t < s.t) { s = cand; hit = true; }
    };
    if (intersectLight(r, c))  consider(c);   // light takes precedence on ties
    if (intersectSphere(r, c)) consider(c);
    if (intersectBox(r, c))    consider(c);
    return hit;
}

// ---------------------------------------------------------------------------
//  Light sampling and occlusion (for NEE)
// ---------------------------------------------------------------------------
// Uniform point on the quad light.
static Vec3 sampleLight(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(-LIGHT_SIZE * 0.5, LIGHT_SIZE * 0.5);
    return LIGHT_POS + Vec3{u(rng), 0, u(rng)};
}

// True if any geometry (sphere or box, never the light itself) blocks the
// straight line from 'from' to 'to'.
static bool occluded(const Vec3& from, const Vec3& to) {
    Vec3 delta = to - from;
    double maxT = delta.length() - 1e-3;         // stop just short of the light
    Ray r{from, delta.normalized()};
    Surface s;
    if (intersectSphere(r, s) && s.t < maxT) return true;
    if (intersectBox(r, s) && s.t < maxT) return true;
    return false;
}

// ---------------------------------------------------------------------------
//  BSDF sampling
// ---------------------------------------------------------------------------
// Cosine-weighted direction in the hemisphere around n (the pdf cancels the
// Lambertian f_d = albedo/PI, so the bounce throughput is just albedo).
static Vec3 cosineHemisphere(const Vec3& n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double phi = 2.0 * PI * u(rng);
    double r = std::sqrt(u(rng));
    Vec3 v{r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0, 1.0 - r * r))};
    // Build an orthonormal basis with z-axis = n.
    Vec3 up = std::fabs(n.y) < 0.999 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 t = up.cross(n).normalized();
    Vec3 b = n.cross(t);
    return v.x * t + v.y * b + v.z * n;
}

// ---------------------------------------------------------------------------
//  Path integrator (with NEE)
// ---------------------------------------------------------------------------
static Vec3 trace(const Ray& ray, std::mt19937_64& rng) {
    Vec3 radiance{0, 0, 0};   // accumulated estimate for this path
    Vec3 throughput{1, 1, 1}; // product of BSDF values along the path
    Surface s;
    Ray r = ray;

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        if (!intersectWorld(r, s)) break;      // escaped the scene -> black

        // Path reached the light: add its emission and stop.
        if (s.mat == Material::Light) {
            radiance += throughput * s.emission;
            break;
        }

        if (s.mat == Material::Diffuse) {
            // --- NEE: direct lighting from the quad light (Lambertian) ---
            // L_dir = L_e * f_d * cos(theta) * cos(LIGHT_N, dir) * A / d^2
            Vec3 lp = sampleLight(rng);
            Vec3 ld = lp - s.p;
            double dist = ld.length();
            ld = ld / dist;
            double cosT = s.n.dot(ld);                 // surface-to-light
            double cosL = LIGHT_N.dot((s.p - lp).normalized()); // light-facing
            if (cosT > 0.0 && cosL > 0.0 && !occluded(s.p + s.n * 1e-3, lp)) {
                radiance += throughput * s.albedo * LIGHT_COL
                          * (cosT * cosL * LIGHT_AREA) / (PI * dist * dist);
            }

            // --- Indirect: cosine-weighted bounce ---
            r = Ray{s.p + s.n * 1e-3, cosineHemisphere(s.n, rng)};
            throughput = throughput * s.albedo;
        } else {
            // --- Perfect mirror: continue along the reflected direction ---
            // (NEE adds nothing for a deterministic BSDF; the reflection ray
            //  hitting the light quad gives the specular highlight.)
            Vec3 h = s.n * (2.0 * s.n.dot(r.d));
            r = Ray{s.p + s.n * 1e-3, (r.d - h).normalized()};
        }

        // Russian roulette: terminate low-contribution paths.
        if (bounce >= 2) {
            double p = std::min(0.95, std::max(throughput.x, std::max(throughput.y, throughput.z)));
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (u(rng) > p) break;
            throughput = throughput / p;   // compensate to stay unbiased
        }
    }
    return radiance;
}

// ---------------------------------------------------------------------------
//  Color management and output
// ---------------------------------------------------------------------------
static unsigned char toSrgb8(double c) {
    c = std::max(0.0, c);
    c = c / (1.0 + c);                          // Reinhard tone mapping
    c = (c <= 0.0031308) ? 12.92 * c            // linear -> sRGB
                         : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    return static_cast<unsigned char>(std::min(1.0, c) * 255.0 + 0.5);
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    constexpr int W = 512, H = 512;
    int spp = (argc > 1) ? std::atoi(argv[1]) : 256;
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<double> jitter(0.0, 1.0);

    // Pinhole camera looking through the open front face (z = -2) at the box.
    Vec3 camPos{0, 0.3, -6.5};
    Vec3 fwd  = (Vec3{0, 0, 0} - camPos).normalized();
    Vec3 right = fwd.cross(Vec3{0, 1, 0}).normalized();
    Vec3 up    = right.cross(fwd);
    double halfTan = std::tan(25.0 * PI / 180.0);  // 50-degree vertical FOV

    std::vector<Vec3> acc(W * H);
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            Vec3 sum{0, 0, 0};
            for (int s = 0; s < spp; ++s) {
                // Jitter the sample inside the pixel for anti-aliasing.
                double a = (px + jitter(rng)) / W * 2.0 - 1.0;
                double b = 1.0 - (py + jitter(rng)) / H * 2.0;
                Vec3 dir = (fwd + right * (a * halfTan) + up * (b * halfTan)).normalized();
                sum += trace(Ray{camPos, dir}, rng);
            }
            acc[py * W + px] = sum / spp;
        }
        if (py % 32 == 31)
            std::fprintf(stderr, "row %d/%d\n", py + 1, H);
    }

    std::FILE* out = std::fopen("output.ppm", "wb");
    if (!out) { std::perror("output.ppm"); return 1; }
    std::fprintf(out, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H; ++i) {
        unsigned char rgb[3] = {
            toSrgb8(acc[i].x), toSrgb8(acc[i].y), toSrgb8(acc[i].z)
        };
        std::fwrite(rgb, 1, 3, out);
    }
    std::fclose(out);
    std::fprintf(stderr, "wrote output.ppm (%dx%d, %d spp)\n", W, H, spp);
    return 0;
}
