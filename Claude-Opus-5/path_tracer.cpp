// path_tracer.cpp -- a compact unidirectional path tracer with next-event estimation.
//
// Scene: an open-fronted diffuse box (interior half-extent 2) holding a perfectly
// specular sphere of radius 1, lit by a single downward-facing quad light recessed
// into the ceiling.  The camera sits in front of the opening looking down -Z.
//
// Output: output.ppm -- 512x512, 8-bit sRGB, binary PPM (P6).
//
// Build: g++ -O2 -ffast-math -fopenmp -o path_tracer path_tracer.cpp
// Run:   ./path_tracer [samples_per_pixel]        (default 512)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------- math ------

static const double kPi = 3.14159265358979323846;

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator*(const Vec3& b) const { return {x * b.x, y * b.y, z * b.z}; }  // per-channel
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    Vec3& operator*=(const Vec3& b) { x *= b.x; y *= b.y; z *= b.z; return *this; }
    Vec3& operator/=(double s) { x /= s; y /= s; z /= s; return *this; }

    double maxComponent() const { return x > y ? (x > z ? x : z) : (y > z ? y : z); }
};

static inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline Vec3 normalize(const Vec3& v) { return v / std::sqrt(dot(v, v)); }

// Mirror d about n (d points *into* the surface, n outward).
static inline Vec3 reflect(const Vec3& d, const Vec3& n) { return d - n * (2.0 * dot(d, n)); }

// Any orthonormal tangent frame around n; used to orient local hemisphere samples.
static void makeBasis(const Vec3& n, Vec3& t, Vec3& b) {
    Vec3 a = std::fabs(n.x) > 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    t = normalize(cross(a, n));
    b = cross(n, t);
}

// ----------------------------------------------------------------- rng ------

// PCG32: small, fast, and cheap to seed independently per pixel.
struct Rng {
    uint64_t state, inc;
    explicit Rng(uint64_t seed, uint64_t seq = 1) : state(0), inc((seq << 1) | 1) {
        nextUInt();
        state += seed;
        nextUInt();
    }
    uint32_t nextUInt() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18) ^ old) >> 27);
        uint32_t rot = static_cast<uint32_t>(old >> 59);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }
    // Uniform in [0,1).
    double operator()() { return (nextUInt() >> 8) * 0x1p-24; }
};

// ------------------------------------------------------------- geometry -----

struct Ray {
    Vec3 o, d;  // d is normalized
};

enum MaterialType { DIFFUSE, SPECULAR };

struct Material {
    Vec3 albedo;              // reflectance: Lambertian tint, or mirror tint
    Vec3 emission;            // emitted radiance (non-zero only for the light)
    MaterialType type = DIFFUSE;
};

// Parallelogram: corner p plus edge vectors u, v.  Points are p + a*u + b*v, a,b in [0,1].
struct Quad {
    Vec3 p, u, v, normal;
    double area;
    Material material;

    Quad(const Vec3& p, const Vec3& u, const Vec3& v, const Material& m)
        : p(p), u(u), v(v), material(m) {
        Vec3 n = cross(u, v);
        area = std::sqrt(dot(n, n));  // |u x v| is the parallelogram area
        normal = n / area;
    }

    bool intersect(const Ray& r, double tMin, double tMax, double& t) const {
        double denom = dot(normal, r.d);
        if (std::fabs(denom) < 1e-12) return false;  // parallel to the plane
        t = dot(p - r.o, normal) / denom;
        if (t <= tMin || t >= tMax) return false;
        Vec3 q = r.o + r.d * t - p;                  // hit point in quad-local coords
        double a = dot(q, u) / dot(u, u);
        double b = dot(q, v) / dot(v, v);
        return a >= 0 && a <= 1 && b >= 0 && b <= 1;
    }
};

struct Sphere {
    Vec3 center;
    double radius;
    Material material;

    bool intersect(const Ray& r, double tMin, double tMax, double& t) const {
        Vec3 oc = r.o - center;                      // d is unit, so a == 1
        double b = dot(oc, r.d);
        double c = dot(oc, oc) - radius * radius;
        double disc = b * b - c;
        if (disc < 0) return false;
        double sd = std::sqrt(disc);
        for (double cand : {-b - sd, -b + sd}) {     // nearest valid root
            if (cand > tMin && cand < tMax) { t = cand; return true; }
        }
        return false;
    }
};

// ---------------------------------------------------------------- scene -----

// The light is a quad inset just below the ceiling, facing straight down (0,-1,0).
// Its albedo is black so it only emits; NEE below samples this quad exclusively.
static const Material kLightMaterial = {Vec3(0, 0, 0), Vec3(18, 17, 15), DIFFUSE};
static const Quad kLight(Vec3(-0.75, 1.98, -0.75), Vec3(1.5, 0, 0), Vec3(0, 0, 1.5),
                         kLightMaterial);

// Box interior: |x|,|y|,|z| <= 2, with the +Z face left open for the camera.
// Quad winding is irrelevant here -- normals are flipped toward the ray on hit.
static const std::vector<Quad> kQuads = {
    // floor, ceiling, back wall: neutral grey
    Quad(Vec3(-2, -2, -2), Vec3(4, 0, 0), Vec3(0, 0, 4), {Vec3(0.75, 0.75, 0.75), {}, DIFFUSE}),
    Quad(Vec3(-2, 2, -2), Vec3(4, 0, 0), Vec3(0, 0, 4), {Vec3(0.75, 0.75, 0.75), {}, DIFFUSE}),
    Quad(Vec3(-2, -2, -2), Vec3(4, 0, 0), Vec3(0, 4, 0), {Vec3(0.75, 0.75, 0.75), {}, DIFFUSE}),
    // side walls: coloured, to make interreflection and the mirror visible
    Quad(Vec3(-2, -2, -2), Vec3(0, 4, 0), Vec3(0, 0, 4), {Vec3(0.75, 0.20, 0.16), {}, DIFFUSE}),
    Quad(Vec3(2, -2, -2), Vec3(0, 4, 0), Vec3(0, 0, 4), {Vec3(0.20, 0.60, 0.28), {}, DIFFUSE}),
    kLight,
};

// Mirror sphere resting on the floor.
static const Sphere kSphere{Vec3(0, -1, 0), 1.0, {Vec3(0.97, 0.97, 0.99), {}, SPECULAR}};

struct Hit {
    double t;
    Vec3 point;
    Vec3 normal;             // always faces the incoming ray
    const Material* material;
};

static const double kEps = 1e-4;  // shadow-ray / self-intersection offset

static bool intersectScene(const Ray& r, double tMax, Hit& hit) {
    bool found = false;
    double t;
    for (const Quad& q : kQuads) {
        if (q.intersect(r, kEps, tMax, t)) {
            tMax = t;
            hit = {t, r.o + r.d * t, q.normal, &q.material};
            found = true;
        }
    }
    if (kSphere.intersect(r, kEps, tMax, t)) {
        hit = {t, r.o + r.d * t, (r.o + r.d * t - kSphere.center) / kSphere.radius,
               &kSphere.material};
        found = true;
    }
    if (found && dot(hit.normal, r.d) > 0) hit.normal = -hit.normal;
    return found;
}

static bool occluded(const Vec3& from, const Vec3& to) {
    Vec3 d = to - from;
    double dist = std::sqrt(dot(d, d));
    Ray r{from, d / dist};
    Hit ignored;
    // Stop just short of the target so the light quad itself doesn't self-shadow.
    return intersectScene(r, dist * (1 - 1e-3), ignored);
}

// ------------------------------------------------- direct lighting (NEE) ----

// One uniform-area sample of the light quad, returned as the factor that the
// surface albedo multiplies:  Le * cos_surface * cos_light * area / (pi * dist^2).
// (The 1/pi is the Lambertian BRDF; area is the reciprocal of the 1/area pdf.)
static Vec3 sampleLight(const Vec3& p, const Vec3& n, Rng& rng) {
    Vec3 lp = kLight.p + kLight.u * rng() + kLight.v * rng();
    Vec3 toLight = lp - p;
    double dist2 = dot(toLight, toLight);
    Vec3 wi = toLight / std::sqrt(dist2);

    double cosSurface = dot(n, wi);
    double cosLight = dot(kLight.normal, -wi);   // light emits along its normal only
    if (cosSurface <= 0 || cosLight <= 0) return {};
    if (occluded(p + n * kEps, lp)) return {};

    return kLight.material.emission * (cosSurface * cosLight * kLight.area / (kPi * dist2));
}

// ------------------------------------------------------------ path trace ----

static const int kMaxDepth = 24;
static const int kRouletteDepth = 5;

static Vec3 radiance(Ray r, Rng& rng) {
    Vec3 L(0, 0, 0);        // accumulated radiance
    Vec3 beta(1, 1, 1);     // path throughput (BRDF * cos / pdf so far)
    // Emission is only added on hits that NEE could not have accounted for:
    // the first hit, and hits reached through a specular (delta) bounce.
    bool includeEmission = true;

    for (int depth = 0; depth < kMaxDepth; ++depth) {
        Hit hit;
        if (!intersectScene(r, 1e30, hit)) break;  // escaped through the open face
        const Material& m = *hit.material;

        if (includeEmission) L += beta * m.emission;

        if (m.type == SPECULAR) {
            // Delta lobe: NEE cannot sample it, so the mirror direction carries
            // emission itself.  BRDF/pdf reduces to the mirror tint.
            r = Ray{hit.point + hit.normal * kEps, reflect(r.d, hit.normal)};
            beta *= m.albedo;
            includeEmission = true;
            continue;
        }

        // Diffuse: explicit light sample this bounce...
        L += beta * m.albedo * sampleLight(hit.point, hit.normal, rng);

        // ...and one cosine-weighted indirect bounce.  With pdf = cos/pi the
        // BRDF (albedo/pi) and cosine terms cancel down to just the albedo.
        beta *= m.albedo;
        double u1 = rng(), u2 = rng();
        double sinTheta = std::sqrt(u1), phi = 2 * kPi * u2;
        Vec3 t, b;
        makeBasis(hit.normal, t, b);
        Vec3 dir = t * (sinTheta * std::cos(phi)) + b * (sinTheta * std::sin(phi)) +
                   hit.normal * std::sqrt(1 - u1);
        r = Ray{hit.point + hit.normal * kEps, normalize(dir)};
        includeEmission = false;  // light hit along this ray is already in the NEE term

        // Russian roulette on throughput, keeping the estimator unbiased.
        if (depth >= kRouletteDepth) {
            double q = beta.maxComponent() < 1.0 ? beta.maxComponent() : 1.0;
            if (rng() >= q) break;
            beta /= q;
        }
    }
    return L;
}

// ---------------------------------------------------------------- output ----

// Linear -> sRGB transfer function (IEC 61966-2-1), then quantize to 8 bits.
static uint8_t toSrgb8(double c) {
    if (c <= 0) c = 0;
    if (c > 1) c = 1;
    double s = c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    return static_cast<uint8_t>(s * 255.0 + 0.5);
}

// ------------------------------------------------------------------ main ----

int main(int argc, char** argv) {
    const int width = 512, height = 512;
    int spp = argc > 1 ? std::atoi(argv[1]) : 512;
    if (spp < 1) spp = 1;

    // Pinhole camera in front of the open face, framing the box interior.
    const Vec3 eye(0, 0, 6.5);
    const Vec3 forward(0, 0, -1), right(1, 0, 0), up(0, 1, 0);
    const double tanHalfFov = std::tan(45.0 * 0.5 * kPi / 180.0);  // 45 deg vertical fov

    std::vector<Vec3> film(static_cast<size_t>(width) * height);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int y = 0; y < height; ++y) {
        Rng rng(0x9E3779B97F4A7C15ULL, static_cast<uint64_t>(y) + 1);  // per-row stream
        for (int x = 0; x < width; ++x) {
            Vec3 sum(0, 0, 0);
            for (int s = 0; s < spp; ++s) {
                // Jittered pixel position: box-filter antialiasing.
                double px = (2.0 * (x + rng()) / width - 1.0) * tanHalfFov;
                double py = (1.0 - 2.0 * (y + rng()) / height) * tanHalfFov;
                Ray r{eye, normalize(right * px + up * py + forward)};
                sum += radiance(r, rng);
            }
            film[static_cast<size_t>(y) * width + x] = sum / spp;
        }
        if (y % 32 == 0) fprintf(stderr, "\rrendering: row %d/%d", y, height);
    }
    fprintf(stderr, "\rrendering: done            \n");

    FILE* f = fopen("output.ppm", "wb");
    if (!f) { perror("output.ppm"); return 1; }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (const Vec3& c : film) {
        uint8_t rgb[3] = {toSrgb8(c.x), toSrgb8(c.y), toSrgb8(c.z)};
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
    printf("wrote output.ppm (%dx%d, %d spp)\n", width, height, spp);
    return 0;
}
