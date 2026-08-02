#!/usr/bin/env python3
"""
Simple Path Tracer with Direct Lighting / NEE in Python
Renders a specular sphere in a diffuse box with a quad area light
"""

import math
import random
import sys

# Vector operations
class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, t):
        if isinstance(t, Vec3):
            return Vec3(self.x * t.x, self.y * t.y, self.z * t.z)
        return Vec3(self.x * t, self.y * t, self.z * t)
    
    def __rmul__(self, t):
        return self.__mul__(t)
    
    def __truediv__(self, t):
        if isinstance(t, Vec3):
            return Vec3(self.x / t.x, self.y / t.y, self.z / t.z)
        return Vec3(self.x / t, self.y / t, self.z / t)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def squared_length(self):
        return self.x*self.x + self.y*self.y + self.z*self.z
    
    def normalize(self):
        length = self.length()
        return Vec3(self.x / length, self.y / length, self.z / length)
    
    def __repr__(self):
        return f"Vec3({self.x}, {self.y}, {self.z})"

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
    
    def at(self, t):
        return self.origin + self.direction * t

# Random number generation
def random_double():
    return random.random()

def random_double_range(min_val, max_val):
    return min_val + (max_val - min_val) * random_double()

# Material types
DIFFUSE = 0
SPECULAR = 1

# Sphere structure
class Sphere:
    def __init__(self, center, radius, material_type, color, fuzz=0.0):
        self.center = center
        self.radius = radius
        self.type = material_type
        self.color = color
        self.fuzz = fuzz

# Light structure
class Light:
    def __init__(self, position, normal, size, emission):
        self.position = position
        self.normal = normal
        self.size = size
        self.emission = emission

# Hit detection for spheres
def hit_sphere(sphere, r, t_min, t_max):
    oc = r.origin - sphere.center
    a = r.direction.dot(r.direction)
    b = 2.0 * oc.dot(r.direction)
    c = oc.dot(oc) - sphere.radius * sphere.radius
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    
    t1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2.0 * a)
    
    if t1 < t_min or t1 > t_max:
        return None
    return t1

# Check if ray hits any sphere
def hit_any_sphere(spheres, r, t_min, t_max):
    closest_t = t_max
    hit_index = -1
    for i, sphere in enumerate(spheres):
        t = hit_sphere(sphere, r, t_min, closest_t)
        if t is not None and t < closest_t:
            closest_t = t
            hit_index = i
    return hit_index, closest_t if hit_index != -1 else None

# Estimate direct lighting using NEE
def estimate_direct_lighting(r, intersection_point, normal, sphere, light):
    direct_light = Vec3(0, 0, 0)
    
    # Generate random point on quad light
    u = random_double_range(-1.0, 1.0) * light.size
    v = random_double_range(-1.0, 1.0) * light.size
    point = Vec3(u, 0, v)
    light_point = light.position + point
    
    light_dir = (light_point - intersection_point).normalize()
    
    # Check for shadow ray occlusion
    shadow_ray = Ray(intersection_point, light_dir)
    shadow_hit, _ = hit_any_sphere(spheres, shadow_ray, 0.001, float('inf'))
    if shadow_hit != -1:
        # Occluded - no contribution
        return direct_light
    
    # Calculate direct lighting contribution
    cos_theta = max(0.0, normal.dot(light_dir))
    if cos_theta > 0:
        distance_squared = (light_point - intersection_point).squared_length()
        light_cos_theta = max(0.0, (-light_dir).dot(light.normal))
        
        if sphere.type == DIFFUSE:
            brdf = sphere.color * (1.0 / math.pi)
        else:  # SPECULAR
            view_dir = (-r.direction).normalize()
            reflect_dir = view_dir - 2.0 * view_dir.dot(normal) * normal
            cos_alpha = pow(max(0.0, light_dir.dot(reflect_dir)), 1.0 / (sphere.fuzz + 0.001))
            brdf = sphere.color * cos_alpha / math.pi
        
        contribution = brdf * light.emission * cos_theta * light_cos_theta / \
                      (distance_squared * light.size * light.size * math.pi)
        direct_light = direct_light + contribution
    
    return direct_light

# Trace ray and compute color
def trace_ray(r, spheres, lights, depth=0):
    if depth > 5:
        return Vec3(0, 0, 0)
    
    hit_index, t = hit_any_sphere(spheres, r, 0.001, float('inf'))
    
    if hit_index is not None and t is not None:
        sphere = spheres[hit_index]
        intersection_point = r.at(t)
        normal = (intersection_point - sphere.center).normalize()
        
        direct_light = Vec3(0, 0, 0)
        for light in lights:
            direct_light = direct_light + estimate_direct_lighting(r, intersection_point, normal, sphere, light)
        
        indirect_light = Vec3(0, 0, 0)
        
        if depth < 4:
            scattered_direction = Vec3(0, 0, 0)
            
            if sphere.type == DIFFUSE:
                # Lambertian diffuse reflection
                target = intersection_point + normal + Vec3(random_double(), random_double(), random_double()).cross(normal)
                scattered_direction = (target - intersection_point).normalize()
            else:  # SPECULAR
                view_dir = (-r.direction).normalize()
                reflect_dir = view_dir - 2.0 * view_dir.dot(normal) * normal
                # Add fuzziness
                fuzz_effect = Vec3(random_double_range(-1, 1), random_double_range(-1, 1), random_double_range(-1, 1))
                scattered_direction = (reflect_dir + sphere.fuzz * fuzz_effect).normalize()
            
            scattered = Ray(intersection_point, scattered_direction)
            indirect_light = trace_ray(scattered, spheres, lights, depth + 1)
        
        return direct_light + indirect_light
    
    # Background color
    unit_direction = r.direction.normalize()
    t = 0.5 * (unit_direction.y + 1.0)
    return Vec3(1.0, 1.0, 1.0) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t

# Gamma correction
def gamma_correction(value):
    return pow(max(0.0, min(1.0, value)), 1.0 / 2.2)

def main():
    # Initialize random seed
    random.seed()
    
    # Image settings
    image_width = 512
    image_height = 512
    samples_per_pixel = 50  # Reduced for Python performance
    
    # Camera setup
    lookfrom = Vec3(0, 0, 0)
    lower_left_corner = Vec3(-2.0, -1.0, -1.0)
    horizontal = Vec3(4.0, 0.0, 0.0)
    vertical = Vec3(0.0, 2.0, 0.0)
    
    # Setup scene
    spheres = [
        Sphere(Vec3(0, 0, -1), 1, SPECULAR, Vec3(0.8, 0.8, 0.8), 0.1),  # Specular sphere
        Sphere(Vec3(0, 0, -1), 2, DIFFUSE, Vec3(0.8, 0.3, 0.3))         # Diffuse box
    ]
    
    lights = [
        Light(Vec3(0, 1.95, -1), Vec3(0, -1, 0), 0.5, Vec3(20, 20, 20))  # Quad light
    ]
    
    # Output image
    with open("output.ppm", "w") as file:
        file.write(f"P3\n{image_width} {image_height}\n255\n")
        
        for j in range(image_height-1, -1, -1):
            print(f"\rScanlines remaining: {j}", end='', flush=True)
            for i in range(image_width):
                color = Vec3(0, 0, 0)
                
                # Anti-aliasing with multiple samples
                for s in range(samples_per_pixel):
                    u = (i + random_double()) / image_width
                    v = (j + random_double()) / image_height
                    
                    direction = lower_left_corner + horizontal * u + vertical * v - lookfrom
                    r = Ray(lookfrom, direction)
                    color = color + trace_ray(r, spheres, lights)
                
                color = color * (1.0 / samples_per_pixel)
                
                # Apply gamma correction
                ir = int(256 * gamma_correction(color.x))
                ig = int(256 * gamma_correction(color.y))
                ib = int(256 * gamma_correction(color.z))
                
                # Clamp values
                ir = max(0, min(255, ir))
                ig = max(0, min(255, ig))
                ib = max(0, min(255, ib))
                
                file.write(f"{ir} {ig} {ib}\n")
        
        print("\nDone.")
    
    print("Image saved as output.ppm")

if __name__ == "__main__":
    main()