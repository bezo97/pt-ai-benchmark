// Minimal path tracer with next-event estimation (direct light sampling).
// Scene: mirror sphere (r=1) inside a diffuse box (half-size 2), open at the
// front (+z), lit by a downward-facing quad light just below the ceiling.
// Output: 512x512 8-bit sRGB binary PPM (output.ppm).
//
// Build: g++ -O3 -std=c++17 -pthread pathtracer.cpp -o pathtracer

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

// ---------------------------------------------------------------- math ----
struct Vec {
    double x = 0, y = 0, z = 0;
    Vec operator+(Vec b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec operator-(Vec b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec operator*(Vec b) const { return {x * b.x, y * b.y, z * b.z}; }
    Vec operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec& operator+=(Vec b) { return *this = *this + b; }
};
double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec cross(Vec a, Vec b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec normalize(Vec v) { return v * (1.0 / std::sqrt(dot(v, v))); }

struct Ray { Vec o, d; };

constexpr double PI = 3.14159265358979323846;
constexpr double EPS = 1e-4;

// --------------------------------------------------------------- scene ----
enum Material { DIFFUSE, MIRROR, LIGHT };

struct Hit {
    double t;
    Vec p, n;      // hit point and surface normal (facing into the box / out of sphere)
    Vec albedo;
    Material mat;
};

constexpr double BOX = 2.0;                        // box half-size
const Vec SPHERE_C = {0, -1, 0};                   // sphere rests on the floor
constexpr double SPHERE_R = 1.0;
constexpr double LIGHT_Y = BOX - 0.01;             // light sits just below ceiling
constexpr double LIGHT_H = 0.5;                    // light half-size (1x1 quad)
constexpr double LIGHT_AREA = 4 * LIGHT_H * LIGHT_H;
const Vec LIGHT_N = {0, -1, 0};                    // emits downward
const Vec LIGHT_LE = {15, 15, 15};                 // emitted radiance

// Axis-aligned wall: plane coord[axis] == pos, with inward normal and color.
struct Wall { int axis; double pos; Vec n, albedo; };
const Wall WALLS[] = {
    {0, -BOX, {1, 0, 0},  {0.75, 0.15, 0.15}},     // left   (red)
    {0,  BOX, {-1, 0, 0}, {0.15, 0.75, 0.15}},     // right  (green)
    {1, -BOX, {0, 1, 0},  {0.75, 0.75, 0.75}},     // floor
    {1,  BOX, {0, -1, 0}, {0.75, 0.75, 0.75}},     // ceiling
    {2, -BOX, {0, 0, 1},  {0.75, 0.75, 0.75}},     // back   (front +z is open)
};

double comp(Vec v, int a) { return a == 0 ? v.x : a == 1 ? v.y : v.z; }

// Closest intersection along the ray; returns false if the ray escapes.
bool intersect(const Ray& r, Hit& h) {
    h.t = 1e30;

    // Walls: plane hit, then check the point lies inside the box face.
    for (const Wall& w : WALLS) {
        double d = comp(r.d, w.axis);
        if (std::fabs(d) < 1e-12) continue;
        double t = (w.pos - comp(r.o, w.axis)) / d;
        if (t < EPS || t >= h.t) continue;
        Vec p = r.o + r.d * t;
        bool inside = true;
        for (int a = 0; a < 3; ++a)
            if (a != w.axis && std::fabs(comp(p, a)) > BOX) inside = false;
        if (inside) h = {t, p, w.n, w.albedo, DIFFUSE};
    }

    // Sphere: solve |o + t d - c|^2 = R^2 (d is unit length).
    Vec oc = r.o - SPHERE_C;
    double b = dot(oc, r.d), c = dot(oc, oc) - SPHERE_R * SPHERE_R;
    double disc = b * b - c;
    if (disc > 0) {
        double s = std::sqrt(disc);
        double t = (-b - s > EPS) ? -b - s : -b + s;
        if (t > EPS && t < h.t) {
            Vec p = r.o + r.d * t;
            h = {t, p, normalize(p - SPHERE_C), {0.95, 0.95, 0.95}, MIRROR};
        }
    }

    // Light quad (horizontal plane y = LIGHT_Y).
    if (std::fabs(r.d.y) > 1e-12) {
        double t = (LIGHT_Y - r.o.y) / r.d.y;
        if (t > EPS && t < h.t) {
            Vec p = r.o + r.d * t;
            if (std::fabs(p.x) <= LIGHT_H && std::fabs(p.z) <= LIGHT_H)
                h = {t, p, LIGHT_N, {0, 0, 0}, LIGHT};
        }
    }
    return h.t < 1e30;
}

// ---------------------------------------------------------- integrator ----
using Rng = std::mt19937_64;
double rand01(Rng& g) { return std::uniform_real_distribution<double>(0, 1)(g); }

// Cosine-weighted direction in the hemisphere around n (pdf = cos/pi).
Vec sampleCosine(Vec n, Rng& g) {
    double r1 = 2 * PI * rand01(g), r2 = rand01(g), r = std::sqrt(r2);
    Vec u = normalize(cross(std::fabs(n.x) > 0.1 ? Vec{0, 1, 0} : Vec{1, 0, 0}, n));
    Vec v = cross(n, u);
    return normalize(u * (std::cos(r1) * r) + v * (std::sin(r1) * r) + n * std::sqrt(1 - r2));
}

Vec reflect(Vec d, Vec n) { return d - n * (2 * dot(d, n)); }

// Estimate radiance along a camera ray.
// Diffuse hits gather direct light via NEE, so emission found by a BSDF-sampled
// ray is only counted after camera or mirror bounces (avoids double counting).
Vec radiance(Ray r, Rng& g) {
    Vec L, beta = {1, 1, 1};
    bool countEmission = true;

    for (int depth = 0; depth < 64; ++depth) {
        Hit h;
        if (!intersect(r, h)) break;               // escaped through the open front

        if (h.mat == LIGHT) {
            if (countEmission && dot(r.d, h.n) < 0) L += beta * LIGHT_LE;  // front face only
            break;
        }

        if (h.mat == MIRROR) {                     // perfect specular reflection
            beta = beta * h.albedo;
            r = {h.p + h.n * EPS, reflect(r.d, h.n)};
            countEmission = true;
            continue;
        }

        // --- Diffuse: next-event estimation toward a uniform point on the light.
        Vec lp = {(2 * rand01(g) - 1) * LIGHT_H, LIGHT_Y, (2 * rand01(g) - 1) * LIGHT_H};
        Vec toL = lp - h.p;
        double dist2 = dot(toL, toL), dist = std::sqrt(dist2);
        Vec wi = toL * (1.0 / dist);
        double cosS = dot(h.n, wi), cosL = -dot(LIGHT_N, wi);
        if (cosS > 0 && cosL > 0) {
            Hit sh;
            Ray shadow = {h.p + h.n * EPS, wi};
            if (intersect(shadow, sh) && sh.mat == LIGHT) {
                // f * Le * G / pdf,  f = albedo/pi,  pdf_area = 1/A
                double G = cosS * cosL / dist2;
                L += beta * h.albedo * LIGHT_LE * (G * LIGHT_AREA / PI);
            }
        }

        // --- Indirect: cosine sampling makes the weight just the albedo.
        beta = beta * h.albedo;
        r = {h.p + h.n * EPS, sampleCosine(h.n, g)};
        countEmission = false;

        // Russian roulette after a few bounces keeps the estimator unbiased.
        if (depth > 3) {
            double p = std::max({beta.x, beta.y, beta.z});
            if (rand01(g) > p) break;
            beta = beta * (1.0 / p);
        }
    }
    return L;
}

// ---------------------------------------------------------------- output ----
// Linear [0,1] -> 8-bit sRGB (IEC 61966-2-1 transfer curve).
unsigned char toSRGB(double c) {
    c = std::clamp(c, 0.0, 1.0);
    c = c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1 / 2.4) - 0.055;
    return (unsigned char)std::lround(c * 255);
}

int main() {
    constexpr int W = 512, H = 512, SPP = 256;

    // Pinhole camera in front of the open face, looking down -z.
    const Vec eye = {0, 0, 6.8};
    const double tanHalf = std::tan(0.5 * 45.0 * PI / 180);

    std::vector<Vec> img(W * H);
    std::atomic<int> nextRow{0};

    auto worker = [&] {
        for (int y; (y = nextRow++) < H;) {
            Rng g(0x9E3779B97F4A7C15ull ^ (uint64_t)y);   // deterministic per row
            for (int x = 0; x < W; ++x) {
                Vec sum;
                for (int s = 0; s < SPP; ++s) {
                    // Jittered sample within the pixel, mapped to [-1,1] screen space.
                    double sx = (2 * (x + rand01(g)) / W - 1) * tanHalf;
                    double sy = (1 - 2 * (y + rand01(g)) / H) * tanHalf;
                    sum += radiance({eye, normalize({sx, sy, -1})}, g);
                }
                img[y * W + x] = sum * (1.0 / SPP);
            }
        }
    };

    std::vector<std::thread> pool;
    for (unsigned i = 0; i < std::max(1u, std::thread::hardware_concurrency()); ++i)
        pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    FILE* f = std::fopen("output.ppm", "wb");
    if (!f) { std::perror("output.ppm"); return 1; }
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (const Vec& c : img) {
        unsigned char px[3] = {toSRGB(c.x), toSRGB(c.y), toSRGB(c.z)};
        std::fwrite(px, 1, 3, f);
    }
    std::fclose(f);
    return 0;
}
