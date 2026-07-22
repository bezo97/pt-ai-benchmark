// ============================================================================
// Path Tracer with Next Event Estimation (Direct Lighting)
// Scene: Specular sphere (r=1) inside a diffuse box (r=2), open front,
//        quad light on ceiling pointing down.
// Output: 512x512 8-bit sRGB PPM (P6)
// Compile: cl pathtracer.cpp /O2 /EHsc /std:c++17 /Fe:pathtracer.exe
// ============================================================================

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// 3D Vector with common operations
// ---------------------------------------------------------------------------
struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    // Index access for axis-aligned operations
    double& operator[](int i) { return (&x)[i]; }
    double  operator[](int i) const { return (&x)[i]; }

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator-()              const { return {-x,    -y,    -z};     }
    Vec3 operator*(double s)      const { return {x * s,   y * s,   z * s}; }
    Vec3 operator/(double s)      const { return {x / s,   y / s,   z / s}; }
    Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }

    Vec3  abs()    const { return {std::abs(x), std::abs(y), std::abs(z)}; }
    Vec3  clamp0() const { return {std::max(x,0.), std::max(y,0.), std::max(z,0.)}; }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3  cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3  normalize() const { double l = length(); return l > 0 ? *this / l : Vec3{}; }
};

// ---------------------------------------------------------------------------
// Ray: origin + direction * t
// ---------------------------------------------------------------------------
struct Ray {
    Vec3 origin;
    Vec3 dir;
};

// ---------------------------------------------------------------------------
// Hit record: result of a ray-scene intersection
// ---------------------------------------------------------------------------
struct Hit {
    bool   hit = false;
    double t   = 1e30;   // distance along ray
    Vec3   p;            // hit point
    Vec3   normal;       // surface normal at hit point
    Vec3   albedo;       // surface colour (for diffuse/specular)
    bool   specular = false;  // true = mirror, false = diffuse
    bool   isLight  = false;  // true = emissive surface
    Vec3   lightEmission;     // emitted radiance (for lights)
};

// ---------------------------------------------------------------------------
// Scene constants
// ---------------------------------------------------------------------------
static constexpr double BOX_HALF      = 2.0;     // box: [-2, +2] on each axis
static constexpr double SPHERE_R      = 1.0;     // sphere radius
static constexpr double LIGHT_HALF_W  = 0.8;     // quad light half-width (1.6x1.6)
static constexpr double LIGHT_INTENSITY = 15.0;  // light brightness (tone-mapped)
static constexpr int    WIDTH         = 512;
static constexpr int    HEIGHT        = 512;
static constexpr int    MAX_BOUNCE    = 20;      // max path depth
static constexpr int    SAMPLES       = 256;     // samples per pixel
static constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Sphere intersection (quadratic formula)
// ---------------------------------------------------------------------------
Hit intersect_sphere(Ray ray, Vec3 center, double radius, Vec3 albedo, bool spec) {
    Vec3 oc = ray.origin - center;
    double a = ray.dir.dot(ray.dir);
    double b = 2.0 * oc.dot(ray.dir);
    double c = oc.dot(oc) - radius * radius;
    double disc = b * b - 4.0 * a * c;

    if (disc < 0) return {}; // no intersection

    double sqrtD = std::sqrt(disc);
    // Take the closer root, fallback to farther if too close
    double t = (-b - sqrtD) / (2.0 * a);
    if (t < 1e-4) t = (-b + sqrtD) / (2.0 * a);
    if (t < 1e-4) return {}; // behind camera or too close

    Vec3 p = ray.origin + ray.dir * t;
    Hit h;
    h.hit = true;
    h.t = t;
    h.p = p;
    h.normal = (p - center).normalize(); // outward normal
    h.albedo = albedo;
    h.specular = spec;
    return h;
}

// ---------------------------------------------------------------------------
// Box face intersection (axis-aligned plane within bounding square)
// axis: 0=x, 1=y, 2=z; side: -1 or +1
// ---------------------------------------------------------------------------
Hit intersect_box_face(Ray ray, int axis, int side, Vec3 albedo) {
    double planeCoord = side * BOX_HALF;
    double dirComp = ray.dir[axis];

    if (std::abs(dirComp) < 1e-10) return {}; // ray parallel to plane

    double t = (planeCoord - ray.origin[axis]) / dirComp;
    if (t < 1e-4) return {}; // behind camera

    Vec3 p = ray.origin + ray.dir * t;

    // Check hit is within the face's bounding square
    if (axis != 0 && std::abs(p.x) > BOX_HALF + 1e-4) return {};
    if (axis != 1 && std::abs(p.y) > BOX_HALF + 1e-4) return {};
    if (axis != 2 && std::abs(p.z) > BOX_HALF + 1e-4) return {};

    // Normal points inward (into the box)
    Vec3 normal{0, 0, 0};
    normal[axis] = -side;

    Hit h;
    h.hit = true;
    h.t = t;
    h.p = p;
    h.normal = normal;
    h.albedo = albedo;
    h.specular = false; // diffuse
    return h;
}

// ---------------------------------------------------------------------------
// Quad light intersection (ceiling, y = +BOX_HALF)
// Only visible from below (inside the box)
// ---------------------------------------------------------------------------
Hit intersect_light(Ray ray, double halfW) {
    double planeY = BOX_HALF;
    if (std::abs(ray.dir.y) < 1e-10) return {};

    double t = (planeY - ray.origin.y) / ray.dir.y;
    if (t < 1e-4) return {};

    Vec3 p = ray.origin + ray.dir * t;
    if (std::abs(p.x) > halfW + 1e-4 || std::abs(p.z) > halfW + 1e-4) return {};

    // Reject rays coming from above the light
    if (ray.dir.y < 0) return {};

    Hit h;
    h.hit = true;
    h.t = t;
    h.p = p;
    h.normal = {0, -1, 0}; // points down into the scene
    h.isLight = true;
    h.lightEmission = {LIGHT_INTENSITY, LIGHT_INTENSITY, LIGHT_INTENSITY};
    h.specular = false;
    return h;
}

// ---------------------------------------------------------------------------
// Full scene intersection: sphere + 5 box faces + light
// Returns the nearest hit
// ---------------------------------------------------------------------------
Hit intersect_scene(Ray ray) {
    Hit best;

    // Specular sphere (silver)
    Hit s = intersect_sphere(ray, {0, 0, 0}, SPHERE_R, {0.9, 0.9, 0.95}, true);
    if (s.hit && s.t < best.t) best = s;

    // Diffuse box faces (warm white) -- 5 faces, front (z=-2) is open
    Vec3 boxAlbedo{0.73, 0.65, 0.55};
    for (int ax : {0, 1, 2}) {
        for (int side : {-1, 1}) {
            if (ax == 2 && side == -1) continue; // skip open front face
            Hit f = intersect_box_face(ray, ax, side, boxAlbedo);
            if (f.hit && f.t < best.t) best = f;
        }
    }

    // Quad light on ceiling (overrides box face at same position)
    Hit l = intersect_light(ray, LIGHT_HALF_W);
    if (l.hit && l.t <= best.t) best = l;

    return best;
}

// ---------------------------------------------------------------------------
// Light sample for NEE: uniform point on quad + visibility check
// ---------------------------------------------------------------------------
struct LightSample {
    Vec3  p;       // sampled point on light
    Vec3  emitted; // L * cos(theta_light) / area
    double dist;   // distance from hit point to light sample
    bool  visible; // true if no occluder between hit point and light
};

LightSample sample_light(Vec3 hitPoint, double halfW, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    LightSample ls;
    ls.visible = false;
    ls.dist = 0;

    // Uniform sample on the quad [-halfW, +halfW] x [-halfW, +halfW]
    double u = dist01(rng) * 2.0 - 1.0;
    double v = dist01(rng) * 2.0 - 1.0;
    ls.p = {u * halfW, BOX_HALF, v * halfW};

    Vec3 toLight = ls.p - hitPoint;
    ls.dist = toLight.length();
    if (ls.dist < 1e-6) return ls;

    Vec3 toLightN = toLight / ls.dist;
    Vec3 lightNormal = {0, -1, 0}; // points down

    // cos(theta_light) = |N_light . (-toLight)|
    double cosL = std::abs(lightNormal.dot(-toLightN));
    if (cosL < 1e-6) return ls;

    // Shadow ray: check if light is visible from hit point
    Ray shadowRay;
    shadowRay.origin = hitPoint;
    shadowRay.dir = toLightN;

    Hit sh = intersect_scene(shadowRay);
    if (sh.hit && sh.isLight && std::abs(sh.t - ls.dist) < 0.01) {
        ls.visible = true;
    }

    // pdf = 1/area for uniform sampling
    double area = 4.0 * halfW * halfW;
    Vec3 radiance = {LIGHT_INTENSITY, LIGHT_INTENSITY, LIGHT_INTENSITY};
    ls.emitted = radiance * cosL / area;

    return ls;
}

// ---------------------------------------------------------------------------
// Next Event Estimation: direct lighting contribution for a diffuse surface
// Formula: integral of f * L * cos(theta) dA / r^2
// With uniform area sampling: f * L * cosH * cosL / r^2
// ---------------------------------------------------------------------------
Vec3 estimate_direct_lighting(Vec3 hitPoint, Vec3 normal, double halfW, std::mt19937& rng) {
    LightSample ls = sample_light(hitPoint, halfW, rng);
    if (!ls.visible) return {};

    Vec3 wo = (ls.p - hitPoint).normalize(); // direction toward light

    // cos(theta_hit) = N . wo (light must be in front of surface)
    double cosH = normal.dot(wo);
    if (cosH < 1e-6) return {};

    double area = 4.0 * halfW * halfW;
    // ls.emitted = L * cosL / area
    // Full term: L * cosH * cosL / r^2 = ls.emitted * cosH * area / dist^2
    return ls.emitted * cosH * area / (ls.dist * ls.dist);
}

// ---------------------------------------------------------------------------
// Cosine-weighted hemisphere sampling for diffuse bounce
// Returns a direction in the hemisphere centered on 'normal'
// ---------------------------------------------------------------------------
Vec3 sample_cosine_hemisphere(Vec3 normal, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    // Build orthonormal basis (u, v, normal)
    Vec3 up = std::abs(normal.y) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 u = (up.cross(normal)).normalize();
    Vec3 v = normal.cross(u);

    // Cosine-weighted sample: pdf = cos(theta) / pi
    double phi = dist01(rng) * 2.0 * PI;
    double cosTheta = dist01(rng);
    double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));

    return u * sinTheta * std::cos(phi) +
           v * sinTheta * std::sin(phi) +
           normal * cosTheta;
}

// ---------------------------------------------------------------------------
// Perfect mirror reflection
// ---------------------------------------------------------------------------
Vec3 reflect(Vec3 dir, Vec3 normal) {
    return dir - normal * (2.0 * dir.dot(normal));
}

// ---------------------------------------------------------------------------
// Recursive path tracing with NEE
// ---------------------------------------------------------------------------
Vec3 trace_ray(Ray ray, int depth, std::mt19937& rng) {
    if (depth > MAX_BOUNCE) return {};

    Hit h = intersect_scene(ray);
    if (!h.hit) return {}; // escaped to background (black)

    // Direct hit on light: return its emission
    if (h.isLight) return h.lightEmission;

    Vec3 throughput{1.0, 1.0, 1.0};
    Vec3 total{0.0, 0.0, 0.0};

    // --- Direct lighting via NEE (diffuse surfaces only) ---
    if (!h.specular) {
        Vec3 direct = estimate_direct_lighting(h.p, h.normal, LIGHT_HALF_W, rng);
        Vec3 bsdf = h.albedo / PI; // Lambertian reflectance
        total += throughput * (direct * bsdf);
    }

    // --- Indirect bounce ---
    Ray nextRay;
    nextRay.origin = h.p + h.normal * 1e-4; // bias to avoid self-intersection

    if (h.specular) {
        // Mirror reflection: deterministic direction
        nextRay.dir = reflect(ray.dir, h.normal);
        Vec3 Li = trace_ray(nextRay, depth + 1, rng);
        total += throughput * (h.albedo * Li);
    } else {
        // Diffuse bounce: cosine-weighted hemisphere sampling
        nextRay.dir = sample_cosine_hemisphere(h.normal, rng);
        double cosTheta = h.normal.dot(nextRay.dir);
        if (cosTheta > 1e-6) {
            Vec3 bsdf = h.albedo / PI; // Lambertian reflectance
            Vec3 Li = trace_ray(nextRay, depth + 1, rng);
            total += throughput * (bsdf * Li * cosTheta);
        }
    }

    return total;
}

// ---------------------------------------------------------------------------
// HDR to 8-bit sRGB: Reinhard tone mapping + gamma 2.2
// ---------------------------------------------------------------------------
int to_srgb(double v) {
    if (v < 0.0) v = 0.0;
    v = v / (1.0 + v);          // Reinhard tone mapping
    v = std::pow(v, 1.0 / 2.2); // gamma correction
    return static_cast<int>(std::round(v * 255.0));
}

// ---------------------------------------------------------------------------
// Main: camera setup, render loop, PPM output
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Path Tracer with NEE\n";
    std::cout << "  Resolution: " << WIDTH << "x" << HEIGHT << "\n";
    std::cout << "  Samples/pixel: " << SAMPLES << "\n";
    std::cout << "  Scene: specular sphere in diffuse box, quad light on ceiling\n\n";

    // Camera: slightly offset so sphere shows interesting reflections
    Vec3 camPos{0.3, 0.2, -BOX_HALF - 0.5};
    Vec3 camTarget{0.0, 0.0, 0.0};
    Vec3 camForward = (camTarget - camPos).normalize();
    Vec3 camRight = camForward.cross({0, 1, 0}).normalize();
    Vec3 camUp = camRight.cross(camForward).normalize();

    // Image plane: width=2 world units at distance 1 from camera
    double aspect = (double)WIDTH / HEIGHT;
    double imageWidth = 2.0;
    Vec3 pixelDeltaU = camRight * imageWidth / (WIDTH - 1);
    Vec3 pixelDeltaV = -camUp * imageWidth / aspect / (HEIGHT - 1); // negative = top-to-bottom

    Vec3 topLeft = camPos + camForward * 1.0
                   - camRight * imageWidth / 2.0
                   + camUp * imageWidth / aspect / 2.0;

    std::mt19937 rng(42); // seeded for reproducibility
    std::vector<Vec3> pixels(WIDTH * HEIGHT);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Vec3 sum{0.0, 0.0, 0.0};

            for (int s = 0; s < SAMPLES; s++) {
                // Jittered sampling for anti-aliasing
                std::uniform_real_distribution<double> dist01(-0.5, 0.5);
                double u = x + dist01(rng);
                double v = y + dist01(rng);

                Vec3 dir = (topLeft + pixelDeltaU * u + pixelDeltaV * v - camPos).normalize();
                Ray ray{camPos, dir};

                sum += trace_ray(ray, 0, rng);
            }

            pixels[y * WIDTH + x] = sum / SAMPLES;
        }

        if (y % 64 == 0) {
            std::cout << "  Rendering row " << y << "/" << HEIGHT << "...\n";
        }
    }

    // Write PPM (P6 = raw binary RGB)
    std::ofstream ppm("output.ppm", std::ios::binary);
    ppm << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        ppm.put(static_cast<char>(to_srgb(pixels[i].x)));
        ppm.put(static_cast<char>(to_srgb(pixels[i].y)));
        ppm.put(static_cast<char>(to_srgb(pixels[i].z)));
    }
    ppm.close();

    std::cout << "\nDone! Output written to output.ppm\n";
    return 0;
}
