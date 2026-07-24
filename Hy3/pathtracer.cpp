// ---------------------------------------------------------------------------
// Minimal path tracer with explicit (NEE) direct lighting.
//
// Scene:  a perfect-mirror sphere (radius 1) sits at the origin inside a
//         diffuse axis-aligned "box" (half-size 2, open at the front) that is
//         viewed from outside/front. A downward-facing quad light is mounted
//         on the ceiling. Output is an 8-bit sRGB PPM, 512x512.
//
// Lighting model:
//   - Diffuse surfaces are shaded with NEE (sample the area light, shadow ray).
//   - The mirror sphere only reflects (no NEE); it "sees" the light by the
//     continuation ray hitting the light directly.
//   - To avoid double-counting the direct component, light emission is only
//     counted when reached by the primary ray or after a specular bounce.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

using F = double;
static const F PI = 3.141592653589793;

// ---- small vec3 -------------------------------------------------------------
struct V {
    F x, y, z;
    V operator+(const V& o) const { return {x+o.x, y+o.y, z+o.z}; }
    V operator-(const V& o) const { return {x-o.x, y-o.y, z-o.z}; }
    V operator*(F s)             const { return {x*s, y*s, z*s}; }
    V operator*(const V& o)      const { return {x*o.x, y*o.y, z*o.z}; } // comp-wise
    V operator/(F s)            const { return {x/s, y/s, z/s}; }
    V operator-()              const { return {-x, -y, -z}; }   // unary minus
    F  dot(const V& o)    const { return x*o.x + y*o.y + z*o.z; }
    V  cross(const V& o)  const { return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x}; }
    F  len()              const { return std::sqrt(x*x + y*y + z*z); }
    V  norm()             const { F l = len(); return {x/l, y/l, z/l}; }
};

// ---- ray --------------------------------------------------------------------
struct Ray { V ro; V rd; };

// ---- hit record -------------------------------------------------------------
enum Mat { DIFFUSE, MIRROR };

struct Hit {
    F  t   = 1e30;   // distance along ray (1e30 = miss)
    V  n;            // geometric normal
    V  albedo   = {0,0,0};
    V  emission = {0,0,0};
    int mat = DIFFUSE;
};

// ---- scene constants --------------------------------------------------------
static const V  SPHERE_C   = {0, 0, 0};
static const F  SPHERE_R   = 1.0;

static const F  BOX        = 2.0;   // box half-size: interior in [-2,2]^3
static const V  WALL_ALB   = {0.6, 0.6, 0.6}; // gray diffuse walls

// ceiling quad light (faces down). y just below ceiling to avoid self-touch.
static const F  LIGHT_Y    = BOX - 0.001;          // ~1.999
static const F  LIGHT_HALF = 1.2;                  // half-width in x and z
static const F  LIGHT_AREA = (2*LIGHT_HALF)*(2*LIGHT_HALF);
static const V  LIGHT_LE   = {1.6, 1.5, 1.4};      // emission radiance (slightly warm)

// ---- RNG -------------------------------------------------------------------
static std::mt19937 rng(12345);
static F randU() { return std::uniform_real_distribution<F>(0,1)(rng); }

// ---------------------------------------------------------------------------
// Intersection: returns the closest hit in the scene.
// ---------------------------------------------------------------------------
static Hit intersect(const V& ro, const V& rd) {
    Hit h; // default = miss (t = 1e30)

    // --- mirror sphere ------------------------------------------------------
    {
        F a = rd.dot(rd);
        F b = 2 * rd.dot(ro - SPHERE_C);
        F c = (ro - SPHERE_C).dot(ro - SPHERE_C) - SPHERE_R*SPHERE_R;
        F disc = b*b - 4*a*c;
        if (disc >= 0) {
            F t = (-b - std::sqrt(disc)) / (2*a);
            if (t > 1e-4 && t < h.t) {
                V p = ro + rd*t;
                h = Hit{t, (p - SPHERE_C).norm(), {0,0,0}, {0,0,0}, MIRROR};
            }
        }
    }

    // --- ceiling quad light (emissive, downward facing) ---------------------
    {
        if (std::fabs(rd.y) > 1e-8) {
            F t = (LIGHT_Y - ro.y) / rd.y;
            V p = ro + rd*t;
            if (t > 1e-4 && t < h.t &&
                std::fabs(p.x) <= LIGHT_HALF && std::fabs(p.z) <= LIGHT_HALF) {
                h = Hit{t, {0,-1,0}, {0,0,0}, LIGHT_LE, DIFFUSE};
            }
        }
    }

    // --- diffuse box walls (front face is OPEN, so no z=+BOX plane) --------
    // Each wall is a bounded axis-aligned plane. Normal points inward.
    // axis: 0=x-plane, 1=y-plane, 2=z-plane. Tangential bounds are the
    // two coordinates other than `axis`, each clamped to [-BOX, BOX].
    auto wall = [&](int axis, F val, const V& nrm) {
        F denom = (axis==0 ? rd.x : (axis==1 ? rd.y : rd.z));
        if (std::fabs(denom) < 1e-8) return;
        F t = (val - (axis==0 ? ro.x : (axis==1 ? ro.y : ro.z))) / denom;
        if (t <= 1e-4 || t >= h.t) return;
        V p = ro + rd*t;
        F a, b;
        if      (axis==0) { a=p.y; b=p.z; }
        else if (axis==1) { a=p.x; b=p.z; }
        else            { a=p.x; b=p.y; } // back wall: tangential x,y
        if (a >= -BOX && a <= BOX && b >= -BOX && b <= BOX)
            h = Hit{t, nrm, WALL_ALB, {0,0,0}, DIFFUSE};
    };

    wall(1, -BOX, {0, 1,0});            // floor  (y = -2)
    // ceiling (y = +2) is diffuse EXCEPT where the light quad sits:
    {
        F tc = (BOX - ro.y)/rd.y;       // ceiling hit distance (recomputed by wall())
        V pc = ro + rd*tc;
        if (tc > 1e-4 && tc < h.t &&
            (std::fabs(pc.x) > LIGHT_HALF || std::fabs(pc.z) > LIGHT_HALF))
            wall(1, BOX, {0,-1,0});     // outside the light quad -> diffuse ceiling
    }
    wall(0, -BOX, { 1,0,0});            // left  wall (x = -2)
    wall(0,  BOX, {-1,0,0});            // right wall (x = +2)
    wall(2, -BOX, {0,0, 1});            // back wall (z = -2); front is open

    return h;
}

// ---------------------------------------------------------------------------
// Cosine-weighted hemisphere sampling (for diffuse continuation rays).
// ---------------------------------------------------------------------------
static V cosineSample(const V& n) {
    F u1 = randU(), u2 = randU();
    F r = std::sqrt(u1), phi = 2*PI*u2;
    F x = r*std::cos(phi), y = r*std::sin(phi), z = std::sqrt(1 - u1);
    V t = (std::fabs(n.x) < 0.9) ? V{1,0,0} : V{0,1,0};
    V b1 = (t - n*(n.dot(t))).norm();
    V b2 = n.cross(b1);
    return b1*x + b2*y + n*z;
}

// ---------------------------------------------------------------------------
// Path tracing with NEE.
// ---------------------------------------------------------------------------
static V trace(V ro, V rd) {
    V thr = {1,1,1};   // throughput
    V col = {0,0,0};   // accumulated radiance
    bool countLight = true; // primary ray / post-specular may "see" the light

    for (int depth = 0; depth < 12; ++depth) {
        Hit h = intersect(ro, rd);
        if (h.t >= 1e30) break;             // escaped through the open front -> black

        V p = ro + rd*h.t;
        V n = h.n;
        if (rd.dot(n) > 0) n = -n;          // face the ray

        // Emission (only counted to avoid double-counting direct light)
        if (h.emission.len() > 0 && countLight)
            col = col + thr * h.emission;
        if (h.emission.len() > 0) break;    // lights are terminal (no bounce)

        if (h.mat == MIRROR) {              // perfect mirror: reflect, keep going
            V r = rd - n * (2 * rd.dot(n));
            ro = p + n * 1e-4; rd = r;
            thr = thr * V{1,1,1};
            countLight = true;
            continue;
        }

        // ---- DIFFUSE surface: explicit direct lighting (NEE) --------------
        {
            // uniform sample a point on the ceiling quad
            V lp = V{ LIGHT_HALF*(2*randU()-1), LIGHT_Y, LIGHT_HALF*(2*randU()-1) };
            V toL = lp - p;
            F dist = toL.len();
            V wi = toL / dist;

            F cosSurf  = std::max(F(0), n.dot(wi));                 // surface term
            F cosLight = std::max(F(0), (lp.y - p.y) / dist);       // light faces down

            if (cosSurf > 0 && cosLight > 0) {
                // shadow ray: is the light visible (not blocked by the sphere)?
                Hit s = intersect(p + n*1e-4, wi);
                bool visible = (s.t >= 1e30) ||           // nothing in between
                               (s.emission.len() > 0 && s.t <= dist + 1e-2);
                if (visible) {
                    V brdf = h.albedo * (1 / PI);
                    // area-light NEE estimator: brdf * Le * cosL * cosS / dist^2 * area
                    V contrib = thr * brdf * LIGHT_LE * cosLight * cosSurf
                                 / (dist*dist) * LIGHT_AREA;
                    col = col + contrib;
                }
            }
        }

        // ---- continue the path with a cosine-weighted diffuse bounce -------
        V wi = cosineSample(n);
        ro = p + n * 1e-4;
        rd = wi;
        thr = thr * h.albedo;   // (cosine pdf cancels the 1/pi and cos terms)
        countLight = false;     // diffuse continuation must NOT re-count direct light
    }
    return col;
}

// ---------------------------------------------------------------------------
// sRGB (gamma 2.2) 8-bit encoding.
// ---------------------------------------------------------------------------
static int to8bit(F v) {
    v = std::pow(std::clamp(v, F(0), F(1)), F(1)/F(2.2));
    int i = int(v * 255 + 0.5);
    return std::min(255, std::max(0, i));
}

int main() {
    const int W = 512, H = 512;
    const int SPP = 64;          // samples per pixel
    const F fovScale = std::tan(F(40*PI/180) / 2); // ~40deg vertical fov

    // camera: outside the open front, looking toward -z
    V cam = {0, 0, 5.0};

    FILE* fp = std::fopen("output.ppm", "wb");
    std::fprintf(fp, "P6\n%d %d\n255\n", W, H);

    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            V c{0,0,0};
            for (int s = 0; s < SPP; ++s) {
                // jittered normalized device coords, y flipped for image up
                F u = (F(i) + randU()) / W  * 2 - 1;
                F v = (F(j) + randU()) / H  * 2 - 1;
                V dir = V{ u*fovScale, v*fovScale, -1 }.norm(); // forward = -z
                c = c + trace(cam, dir);
            }
            c = c / F(SPP);
            std::fputc(to8bit(c.x), fp);
            std::fputc(to8bit(c.y), fp);
            std::fputc(to8bit(c.z), fp);
        }
    }
    std::fclose(fp);
    std::printf("wrote output.ppm (%dx%d, %d spp)\n", W, H, SPP);
    return 0;
}
