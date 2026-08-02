// Simple Path Tracer with Direct Lighting / NEE
// Renders a specular sphere in a diffuse box with a quad area light

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>

// Vector operations
struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double e0, double e1, double e2) : x(e0), y(e1), z(e2) {}

    inline const Vec3& operator+() const { return *this; }
    inline Vec3 operator-() const { return Vec3(-x, -y, -z); }
    inline Vec3& operator+=(const Vec3 &v);
    inline Vec3& operator*=(const double t);
    inline Vec3& operator/=(const double t);

    inline double length() const { return sqrt(x*x + y*y + z*z); }
    inline double squared_length() const { return x*x + y*y + z*z; }
};

inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

inline Vec3 operator*(double t, const Vec3 &v) {
    return Vec3(t*v.x, t*v.y, t*v.z);
}

inline Vec3 operator*(const Vec3 &v, double t) {
    return t * v;
}

inline Vec3 operator/(Vec3 v, double t) {
    return (1.0/t) * v;
}

inline Vec3& Vec3::operator+=(const Vec3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

inline Vec3& Vec3::operator*=(const double t) {
    x *= t;
    y *= t;
    z *= t;
    return *this;
}

inline Vec3& Vec3::operator/=(const double t) {
    double k = 1.0/t;
    x *= k;
    y *= k;
    z *= k;
    return *this;
}

inline double dot(const Vec3 &u, const Vec3 &v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}

inline Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}

// Ray class
struct Ray {
    Vec3 origin, direction;

    Ray() {}
    Ray(const Vec3 &a, const Vec3 &b) : origin(a), direction(b) {}

    Vec3 at(double t) const { return origin + t * direction; }
};

// Material types
enum MaterialType { DIFFUSE, SPECULAR };

// Sphere structure
struct Sphere {
    Vec3 center;
    double radius;
    MaterialType type;
    Vec3 color;
    double fuzz;
};

// Light structure
struct Light {
    Vec3 position;
    Vec3 normal;
    double size;
    Vec3 emission;
};

// Scene definition
std::vector<Sphere> spheres;
std::vector<Light> lights;

// Random number generation
double random_double() {
    return rand() / (RAND_MAX + 1.0);
}

double random_double(double min, double max) {
    return min + (max-min)*random_double();
}

// Generate random point on quad light
Vec3 random_point_on_light(const Light& light) {
    double u = random_double(-1.0, 1.0) * light.size;
    double v = random_double(-1.0, 1.0) * light.size;
    Vec3 point(u, 0, v);
    // Rotate to align with light normal (pointing down)
    return light.position + point;
}

// Hit detection for spheres
bool hit_sphere(const Sphere& sphere, const Ray& r, double t_min, double t_max, double& t) {
    Vec3 oc = r.origin - sphere.center;
    double a = dot(r.direction, r.direction);
    double b = 2.0 * dot(oc, r.direction);
    double c = dot(oc, oc) - sphere.radius * sphere.radius;
    double discriminant = b*b - 4*a*c;
    if (discriminant < 0) return false;
    
    double t1 = (-b - sqrt(discriminant)) / (2.0 * a);
    double t2 = (-b + sqrt(discriminant)) / (2.0 * a);
    
    if (t1 > t2) std::swap(t1, t2);
    if (t1 < t_min || t1 > t_max) {
        t = t1;
        return true;
    }
    return false;
}

// Check if ray hits any sphere
bool hit_any_sphere(const Ray& r, double t_min, double t_max, int& hit_index, double& closest_t) {
    closest_t = t_max;
    bool hit = false;
    for (int i = 0; i < spheres.size(); i++) {
        double t;
        if (hit_sphere(spheres[i], r, t_min, closest_t, t)) {
            if (t < closest_t) {
                closest_t = t;
                hit_index = i;
                hit = true;
            }
        }
    }
    return hit;
}

// Estimate direct lighting using NEE
Vec3 estimate_direct_lighting(const Ray& r, const Vec3& intersection_point, const Vec3& normal, int hit_index) {
    const Sphere& sphere = spheres[hit_index];
    Vec3 direct_light(0, 0, 0);
    
    for (const Light& light : lights) {
        Vec3 light_point = random_point_on_light(light);
        Vec3 light_dir = unit_vector(light_point - intersection_point);
        
        // Check for shadow ray occlusion
        Ray shadow_ray(intersection_point, light_dir);
        int shadow_hit_index;
        double shadow_t;
        if (hit_any_sphere(shadow_ray, 0.001, std::numeric_limits<double>::infinity(), shadow_hit_index, shadow_t)) {
            // Occluded - no contribution
            continue;
        }
        
        // Calculate direct lighting contribution
        double cos_theta = std::max(0.0, dot(normal, light_dir));
        if (cos_theta > 0) {
            double distance_squared = (light_point - intersection_point).squared_length();
            double light_cos_theta = std::max(0.0, dot(-light_dir, light.normal));
            
            Vec3 BRDF;
            if (sphere.type == DIFFUSE) {
                BRDF = sphere.color / M_PI;
            } else { // SPECULAR
                Vec3 view_dir = -unit_vector(r.direction);
                Vec3 reflect_dir = view_dir - 2.0 * dot(view_dir, normal) * normal;
                double cos_alpha = std::pow(std::max(0.0, dot(light_dir, reflect_dir)), 1.0 / (sphere.fuzz + 0.001));
                BRDF = sphere.color * cos_alpha / M_PI;
            }
            
            Vec3 contribution = BRDF * light.emission * cos_theta * light_cos_theta / 
                               (distance_squared * light.size * light.size * M_PI);
            direct_light += contribution;
        }
    }
    
    return direct_light;
}

// Trace ray and compute color
Vec3 trace_ray(const Ray& r, int depth = 0) {
    if (depth > 5) return Vec3(0, 0, 0);

    int hit_index;
    double t;
    if (hit_any_sphere(r, 0.001, std::numeric_limits<double>::infinity(), hit_index, t)) {
        Vec3 intersection_point = r.at(t);
        Vec3 normal = unit_vector(intersection_point - spheres[hit_index].center);
        
        Vec3 direct_light = estimate_direct_lighting(r, intersection_point, normal, hit_index);
        Vec3 indirect_light(0, 0, 0);
        
        if (depth < 4) {
            Vec3 scattered_direction;
            Ray scattered(intersection_point, scattered_direction);
            
            if (spheres[hit_index].type == DIFFUSE) {
                // Lambertian diffuse reflection
                Vec3 target = intersection_point + normal + Vec3(random_double(), random_double(), random_double()).cross(normal);
                scattered_direction = target - intersection_point;
            } else { // SPECULAR
                Vec3 view_dir = -unit_vector(r.direction);
                Vec3 reflect_dir = view_dir - 2.0 * dot(view_dir, normal) * normal;
                // Add fuzziness
                Vec3 fuzz_effect(random_double(-1, 1), random_double(-1, 1), random_double(-1, 1));
                scattered_direction = reflect_dir + spheres[hit_index].fuzz * fuzz_effect;
            }
            
            Ray scattered(intersection_point, scattered_direction);
            indirect_light = trace_ray(scattered, depth + 1);
        }
        
        return direct_light + indirect_light;
    }

    // Background color
    Vec3 unit_direction = unit_vector(r.direction);
    double t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
}

// Gamma correction
double gamma_correction(double value) {
    return pow(value, 1.0 / 2.2);
}

int main() {
    // Initialize random seed
    srand(time(NULL));
    
    // Image settings
    const int image_width = 512;
    const int image_height = 512;
    const int samples_per_pixel = 100;
    
    // Camera setup
    Vec3 lookfrom(0, 0, 0);
    Vec3 lower_left_corner(-2.0, -1.0, -1.0);
    Vec3 horizontal(4.0, 0.0, 0.0);
    Vec3 vertical(0.0, 2.0, 0.0);
    
    // Setup scene
    // Specular sphere of radius 1
    spheres.push_back({Vec3(0, 0, -1), 1, SPECULAR, Vec3(0.8, 0.8, 0.8), 0.1});
    
    // Diffuse box of radius 2
    spheres.push_back({Vec3(0, 0, -1), 2, DIFFUSE, Vec3(0.8, 0.3, 0.3), 0.0});
    
    // Quad light source pointing down from the ceiling
    lights.push_back({Vec3(0, 1.95, -1), Vec3(0, -1, 0), 0.5, Vec3(20, 20, 20)});
    
    // Output image
    std::ofstream file("output.ppm");
    file << "P3\n" << image_width << " " << image_height << "\n255\n";
    
    for (int j = image_height-1; j >= 0; j--) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            Vec3 color(0, 0, 0);
            
            // Anti-aliasing with multiple samples
            for (int s = 0; s < samples_per_pixel; s++) {
                double u = (i + random_double()) / image_width;
                double v = (j + random_double()) / image_height;
                
                Vec3 direction = lower_left_corner + u*horizontal + v*vertical - lookfrom;
                Ray r(lookfrom, direction);
                color += trace_ray(r);
            }
            
            color /= samples_per_pixel;
            
            // Apply gamma correction
            color.x = gamma_correction(color.x);
            color.y = gamma_correction(color.y);
            color.z = gamma_correction(color.z);
            
            // Clamp values
            int ir = static_cast<int>(256 * std::clamp(color.x, 0.0, 0.999));
            int ig = static_cast<int>(256 * std::clamp(color.y, 0.0, 0.999));
            int ib = static_cast<int>(256 * std::clamp(color.z, 0.0, 0.999));
            
            file << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    
    file.close();
    std::cerr << "\nDone.\n";
    return 0;
}