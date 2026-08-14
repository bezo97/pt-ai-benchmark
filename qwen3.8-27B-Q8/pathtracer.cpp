// ============================================================================
//  pathtracer.cpp — small single-file C++ path tracer with NEE
//
//  Scene
//   - Specular (perfect-mirror) sphere, radius 1, at the box center
//   - Diffuse (Lambertian) box of radius 2, i.e. half-extent 2 (walls at ±2)
//   - The front wall (z = -2) is removed: the box is viewed open from the
//     front, camera just outside at (0, 0.5, -5.8), 50 deg FOV
//   - Quad light: a square patch in the ceiling (y = +2), normal (0,-1,0),
//     pointing straight down, warm-white emission
//
//  Output
//   - output.ppm, 512x512, 8-bit sRGB (Reinhard tone map, then gamma 1/2.2)
//
//  Method
//   - Paths are traced from the camera. Diffuse vertices use cosine-weighted
//     BSDF sampling (Lambertian weight = albedo) plus Russian roulette.
//   - NEE (next event estimation): at every diffuse hit we also shoot a
//     shadow ray to a uniformly sampled point on the quad light and add
//         L = f_r * Le * cos(theta) * cos(theta_l) * A / d^2        (pdf = 1/A)
//     which makes direct lighting converge fast (low variance).
//   - The mirror sphere has a delta BSDF: for it, the NEE term *is* the
//     specular continuation, so no shadow ray is needed — the reflected ray
//     itself samples the light (and walls) exactly.
//   - To avoid double-counting direct light: a ray sampled from a diffuse
//     BSDF that happens to land on the emitter contributes 0, because that
//     "hit the light" path is exactly the one NEE at the parent vertex
//     already estimates. Primary (camera) and specular rays do count the
//     emission, so the light stays visible directly and in the mirror.
//
//  Build & run
//    g++ -O3 -std=c++17 pathtracer.cpp -o pt                (portable)
//    g++ -O3 -std=c++17 -fopenmp pathtracer.cpp -o pt       (multithreaded)
//    ./pt  ->  writes output.ppm
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static const double PI = 3.14159265358979323846;

// ------------------------------------------------------------------- vec3 --
struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3 operator + (const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator - (const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator * (double s)        const { return {x * s,   y * s,   z * s}; }
    Vec3 operator * (const Vec3& o)   const { return {x*o.x,   y*o.y,   z*o.z}; }
    Vec3 operator / (double s)        const { return {x / s,   y / s,   z / s}; }
    Vec3& operator *=(const Vec3& o)  { x *= o.x; y *= o.y; z *= o.z; return *this; }
    Vec3& operator +=(const Vec3& o)  { x += o.x; y += o.y; z += o.z; return *this; }
    double dot(const Vec3& o)  const { return x*o.x + y*o.y + z*o.z; }
    Vec3   cross(const Vec3& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    double len()  const { return std::sqrt(dot(*this)); }
    Vec3   norm() const { return *this / (len() + 1e-30); }
    double maxComp() const { return std::max(x, std::max(y, z)); }
};
static Vec3  operator*(double s, const Vec3& v) { return v * s; }
static Vec3  reflect(const Vec3& d, const Vec3& n) { return d - 2.0*d.dot(n)*n; }

struct Ray { Vec3 o, d; };

// ------------------------------------------------------------- scene setup --
static const double BOX         = 2.0;    // box half-extent (walls at ±2)
static const double SPHERE_R    = 1.0;    // mirror sphere radius
static const Vec3   SPHERE_C    {0, 0, 0};
static const Vec3   MIRROR_F    {0.92, 0.93, 0.95};  // sphere reflectance
static const Vec3   WALL_ALBEDO {0.75, 0.75, 0.75};  // box walls (Lambertian)
// Quad light: square patch in the ceiling plane y = +BOX, |x|,|z| <= HALF,
// emitting downward (normal (0,-1,0)).
static const double LIGHT_HALF  = 0.5;
static const double LIGHT_AREA  = (2*LIGHT_HALF) * (2*LIGHT_HALF);
static const Vec3   LIGHT_N     {0, -1, 0};
static const Vec3   LIGHT_E     {18, 17, 15.5};      // emitted radiance

// -------------------------------------------------------------- geometry --
// hit.kind: 0 = wall (diffuse), 1 = sphere (mirror), 2 = light quad (emitter)
struct Hit { double t = 1e30; Vec3 p, n; int kind = -1; };

// One-sided wall: plane n·p = d with n pointing into the box interior,
// clipped to the square |tangential coords| <= BOX. The ceiling passes
// quadHole = true so the quad's footprint belongs to the light, not the wall.
static bool hitWall(const Ray& r, const Vec3& n, double d, int axis,
                    bool quadHole, double tmax, Hit& h) {
    double denom = r.d.dot(n);
    if (std::fabs(denom) < 1e-12) return false;
    double t = (d - r.o.dot(n)) / denom;
    if (t <= 1e-4 || t > tmax) return false;
    Vec3 p = r.o + r.d * t;
    double a = p.x, b = p.y, c = p.z;
    if (axis == 0 && (std::fabs(b) > BOX || std::fabs(c) > BOX)) return false;
    if (axis == 1 && (std::fabs(a) > BOX || std::fabs(c) > BOX)) return false;
    if (axis == 2 && (std::fabs(a) > BOX || std::fabs(b) > BOX)) return false;
    if (quadHole && std::fabs(a) <= LIGHT_HALF && std::fabs(c) <= LIGHT_HALF)
        return false;  // this part of the ceiling is the light
    if (t < h.t) { h.t = t; h.p = p; h.n = n; h.kind = 0; }
    return true;
}

static bool hitSphere(const Ray& r, double tmax, Hit& h) {
    Vec3 m = r.o - SPHERE_C;
    double b = m.dot(r.d);
    double disc = b*b - (m.dot(m) - SPHERE_R*SPHERE_R);
    if (disc < 0) return false;
    double s = std::sqrt(disc);
    double t = -b - s;
    if (t <= 1e-4) t = -b + s;              // inside the sphere: use far root
    if (t <= 1e-4 || t > tmax) return false;
    Vec3 p = r.o + r.d * t;
    if (t < h.t) { h.t = t; h.p = p; h.n = (p - SPHERE_C).norm(); h.kind = 1; }
    return true;
}

// The quad light: pure emitter, no BSDF.
static bool hitLight(const Ray& r, double tmax, Hit& h) {
    if (std::fabs(r.d.y) < 1e-12) return false;
    double t = (BOX - r.o.y) / r.d.y;       // crossing of the ceiling plane
    if (t <= 1e-4 || t > tmax) return false;
    Vec3 p = r.o + r.d * t;
    if (std::fabs(p.x) > LIGHT_HALF || std::fabs(p.z) > LIGHT_HALF) return false;
    if (t < h.t) { h.t = t; h.p = p; h.n = LIGHT_N; h.kind = 2; }
    return true;
}

static bool intersect(const Ray& r, Hit& h) {
    // Inward normals, so every wall satisfies n·p = -BOX.
    hitSphere(r, h.t, h);
    hitWall(r, { 1, 0, 0}, -BOX, 0, false, h.t, h);  // left wall  (x = -2)
    hitWall(r, {-1, 0, 0}, -BOX, 0, false, h.t, h);  // right wall (x = +2)
    hitWall(r, { 0, 1, 0}, -BOX, 1, false, h.t, h);  // floor      (y = -2)
    hitWall(r, { 0,-1, 0}, -BOX, 1, true,  h.t, h);  // ceiling    (y = +2)
    hitWall(r, { 0, 0,-1}, -BOX, 2, false, h.t, h);  // back wall  (z = +2)
    hitLight (r, h.t, h);
    return h.kind >= 0;
    // NOTE: the front wall (z = -2) is deliberately absent — open front.
}

// Nearest occluder for shadow rays — everything except the light itself.
static double shadowT(const Ray& r) {
    Hit h;
    hitSphere(r, h.t, h);
    hitWall(r, { 1, 0, 0}, -BOX, 0, false, h.t, h);
    hitWall(r, {-1, 0, 0}, -BOX, 0, false, h.t, h);
    hitWall(r, { 0, 1, 0}, -BOX, 1, false, h.t, h);
    hitWall(r, { 0,-1, 0}, -BOX, 1, true,  h.t, h);
    hitWall(r, { 0, 0,-1}, -BOX, 2, false, h.t, h);
    return h.t;  // 1e30 if unoccluded
}

// -------------------------------------------------------------- sampling --
// Direction from the cosine lobe about normal n (pdf = cos(theta)/pi).
static Vec3 cosineDir(const Vec3& n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double phi = 2.0 * PI * U(rng);
    double s   = std::sqrt(U(rng));
    Vec3 local = {s*std::cos(phi), s*std::sin(phi), std::sqrt(1.0 - s*s)};
    Vec3 up = (std::fabs(n.y) < 0.99) ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 t = n.cross(up).norm();
    Vec3 b = t.cross(n);
    return t*local.x + b*local.y + n*local.z;
}

// NEE: direct light at point p (Lambertian surface, `albedo`).
// Uniform point on the quad (pdf_A = 1/A); solid-angle conversion gives the
// cos(theta_l) * A / d^2 factor, so the estimate is:
//     L = (albedo/pi) * Le * cos(theta) * cos(theta_l) * A / d^2
static Vec3 directLight(const Vec3& p, const Vec3& n, const Vec3& albedo,
                        std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Vec3 lp  = {(U(rng)*2.0 - 1.0) * LIGHT_HALF, BOX,
                (U(rng)*2.0 - 1.0) * LIGHT_HALF};
    Vec3 toL = lp - p;
    double d2 = toL.dot(toL);
    double d  = std::sqrt(d2);
    double cosP = n.dot(toL) / d;  // cos(theta) at the surface
    double cosL = toL.y / d;       // cos(theta_l) at the light, normal (0,-1,0)
    if (cosP <= 0.0 || cosL <= 0.0) return {};   // light behind surface/light
    if (shadowT({p + (toL/d)*1e-4, toL/d}) < d - 1e-4) return {};  // occluded
    return LIGHT_E * albedo / PI * (cosP * cosL * LIGHT_AREA / d2);
}

// ------------------------------------------------------------------ path --
static const int MAX_BOUNCES = 16;

// countEmission: may a light hit add its radiance?
//   true  — camera ray (the light must be visible) and rays leaving a
//           mirror (delta BSDF: the continuation *is* the NEE there);
//   false — rays sampled from a diffuse BSDF: that "hit the light" path was
//           already estimated by NEE at the parent vertex, counting it
//           again would double-count direct light.
static Vec3 trace(const Ray& ray0, bool countEmission, std::mt19937_64& rng) {
    Ray ray = ray0;
    Vec3 Li{}, throughput{1, 1, 1};
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int b = 0; b < MAX_BOUNCES; ++b) {
        Hit h;
        if (!intersect(ray, h)) break;          // escaped via the open front
        if (h.kind == 2) {                      // hit the emitter
            if (countEmission) Li += throughput * LIGHT_E;
            break;
        }
        if (h.kind == 1) {                      // perfect mirror: just reflect
            ray = {h.p + h.n*1e-4, reflect(ray.d, h.n)};
            countEmission = true;               // see note above
            throughput *= MIRROR_F;
            continue;
        }
        // diffuse wall: NEE term + cosine-weighted BSDF continuation
        Li += throughput * directLight(h.p, h.n, WALL_ALBEDO, rng);
        Vec3 nd = cosineDir(h.n, rng);
        throughput *= WALL_ALBEDO;  // (albedo/pi)*cos / (cos/pi) = albedo
        ray = {h.p + nd*1e-4, nd};
        countEmission = false;
        double s = std::min(0.9, throughput.maxComp());  // Russian roulette
        if (s <= 1e-8 || U(rng) >= s) break;
        throughput /= s;
    }
    return Li;
}

// ------------------------------------------------------------------ main --
int main() {
    const int W = 512, H = 512, SPP = 64;

    // Camera: in front of the open face, looking at the sphere center.
    Vec3 camPos{0, 0.5, -5.8}, camTarget{0, 0, 0};
    Vec3 w = (camTarget - camPos).norm();
    Vec3 u = w.cross({0, 1, 0}).norm();
    Vec3 v = u.cross(w);
    double tfov   = std::tan(50.0 * 0.5 * PI / 180.0);  // 50 deg vertical FOV
    double aspect = double(W) / H;

    std::vector<Vec3> img(W * H);

    auto renderRow = [&](int py, std::mt19937_64& rng) {
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int px = 0; px < W; ++px) {
            Vec3 col{};
            for (int s = 0; s < SPP; ++s) {
                // stratified sample inside the pixel
                double sx = (px + U(rng)) / W;
                double sy = (py + U(rng)) / H;
                Vec3 dir = (w + u*((2*sx - 1) * aspect * tfov)
                            + v*((1 - 2*sy) * tfov)).norm();
                col += trace({camPos, dir}, true, rng);
            }
            img[py*W + px] = col * (1.0 / SPP);
        }
    };

#ifdef _OPENMP
#pragma omp parallel
    {
        std::mt19937_64 rng(0x9E3779B97F4A7C15ULL
                             + 0x2545F4914F6CDD1DULL * omp_get_thread_num());
#pragma omp for schedule(dynamic, 8)
        for (int py = 0; py < H; ++py) renderRow(py, rng);
    }
#else
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    for (int py = 0; py < H; ++py) renderRow(py, rng);
#endif

    // 8-bit sRGB PPM: Reinhard tone map, then linear -> sRGB (gamma 1/2.2).
    std::vector<unsigned char> buf(W * H * 3);
    for (int i = 0; i < W * H; ++i) {
        const double c[3] = {img[i].x, img[i].y, img[i].z};
        for (int ch = 0; ch < 3; ++ch) {
            double x = c[ch] / (1.0 + c[ch]);            // Reinhard
            x = std::pow(std::clamp(x, 0.0, 1.0), 1.0/2.2);  // to sRGB
            buf[i*3 + ch] = (unsigned char)(x * 255.0 + 0.5);
        }
    }
    FILE* f = std::fopen("output.ppm", "wb");
    if (!f) { std::fprintf(stderr, "cannot open output.ppm\n"); return 1; }
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    std::printf("wrote output.ppm (%dx%d, %d spp)\n", W, H, SPP);
    return 0;
}
