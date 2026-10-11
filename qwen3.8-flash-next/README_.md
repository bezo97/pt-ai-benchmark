# A small path tracer with next-event estimation

Single-file, dependency-free C++17 path tracer (~470 lines, mostly comments). It
renders the scene below at 512x512 into an 8-bit sRGB binary PPM.

```
        +---------------------------+   y
        |     [ quad light ]        |   ^
        |     points straight down  |   |
        |                           |   +----> x
        |        .-"```"-._         |  /
   box  |      /( mirror )  \       | v
  walls |      | sphere r=1 |       | z (out of screen, towards camera)
        |      '-....,..--'         |
        |   floor  shadow           |
        +---- open front ------------+  <- camera looks in here
```

## Build & run

| platform | command |
| --- | --- |
| Windows / MSVC | `build.bat` (locates VS via `vswhere`), produces `pt.exe` |
| gcc / clang | `g++ -O3 -std=c++17 -pthread main.cpp -o pt` |
| make | `make` then `./pt` |

```
./pt                 # writes output.ppm (256 spp)
./pt 32 small.ppm    # [spp] [out path]
```

Everything tunable (resolution, samples, fov, light power/placement, sphere
placement, exposure, max depth) is a `k*` constant at the top of `main.cpp`.

## Scene

| item | definition |
| --- | --- |
| box | axis-aligned, half-extent 2 (`radius 2`) centred on the origin, matte |
| open face | the `+Z` wall is simply not created; the camera looks through the hole |
| sphere | radius 1, centred `(0, -1, -0.85)` so it rests on the floor, perfect mirror |
| light | 2x2 quad at `y = 1.98` (just below the ceiling), normal `(0,-1,0)`, radiance 10 |
| camera | `(0, 0.4, 4.5)` looking down `-Z`, 52 deg vertical fov |

The fov/frustum combination is chosen so that **every** primary ray passes through
the open face, so the frame is filled with the interior and never shows the black
void outside. Wall albedos are tinted (warm left, cool right) so colour bleeding
between bounces is visible; the mirror sphere shows the lamp and both side walls.

## Integrator

Classic unidirectional path tracing, one BSDF sample per bounce, cosine-weighted
for the matte walls. Each vertex does exactly one of:

* **diffuse** - *next-event estimation* (one uniform area sample of the quad light,
  plus one shadow ray) **and** one cosine-weighted continuation ray for indirect
  light;
* **mirror** - one reflected ray. NEE is deliberately skipped here: a Dirac BRDF
  has zero response to an area-light sample, so sampling the light is wasted work.
* **emissive** - terminate; add `Le` (see the rule below).

The NEE term at a rough vertex `x` with light point `q` is the rendering equation
evaluated at a single sample,

```
L += Le * (albedo/pi) * cos(x->q) * cos(light) / (pdf_area * |q - x|^2),
pdf_area = 1 / lightArea
```

which is unbiased with a single sample because we sample the *light*, not the
BRDF - so nothing cancels against the path's incoming pdf and the term is simply
added on top of the path throughput.

**Double-counting rule.** A hit on the emitter is only added when the path reached
it through specular vertices only (`deltaChain`), or on a primary ray - see
`trace()`. So every direct-lighting event is counted exactly once:

* rough vertex -> by its own NEE term; a random bounce that happens to land on the
  light contributes nothing (that transport is already integrated by NEE),
* delta vertex -> by the reflected ray physically reaching the light, the only
  mechanism it has, and the reason the sphere shows the lamp.

Because the direct term is handled in closed form per vertex and BSDF sampling
covers the rest, no multiple-importance-sampling weight is needed; Veach's balance
heuristic would only move variance between the two strategies.

**Colour pipeline:** linear radiance -> exposure -> Narkowicz/ACES-fit filmic curve
(the blown-out lamp and its specular highlight roll off instead of clipping) ->
sRGB OETF -> 8-bit channels, written as PPM `P6`.

**Other details:** Russian roulette after 3 bounces, per-path xorshift32 streams
seeded from a splitmix64 hash of `(pixel, sample)` (no shared state, so the render
is bit-reproducible at any thread count), pixel jitter for antialiasing, linear
intersection search over 7 primitives (no BVH - pointless at this size).

## Verification

A/B builds produced by patching one line each, 256 spp, mean 8-bit channel value:

| build | whole image | back wall (diffuse) | sphere (specular) |
| --- | --- | --- | --- |
| shipped | 0.546 | 0.547 | 0.468 |
| NEE call removed | 0.095 | **0.000** (pure black) | 0.076 |
| emitter hits only on primary rays | 0.541 | 0.547 | **0.393** |
| Russian roulette disabled | 0.546 | 0.547 | 0.468 |

Reading: the matte box is lit almost entirely through NEE (without it a random
cosine sample rarely finds the small light, so the interior goes black); the
`deltaChain` term is what puts the lamp into the mirror sphere; disabling Russian
roulette changes the image by 0.003 %, i.e. it costs nothing in accuracy.

## Performance

512x512, 256 spp, max depth 8: **~1.5 s** on a 24-thread desktop CPU
(~0.08 s at 16 spp). Scale with `./pt <spp>`.

## Known simplifications

Because NEE picks a random point on the quad light and then tests visibility, the
direct shadows already have correct (if grainy) area-light penumbrae; the indirect
shadows are as good as the paths that carry them. Remaining simplifications: no
Fresnel or roughness on the mirror, one emitter, purely Lambertian walls, no
light-vs-BRDF MIS, no NEE on specular vertices.

The black core of the sphere is not a bug: the point of a convex mirror that faces
the camera reflects straight back along the view ray, so it shows the unlit void
outside the open front. Every surface point that reflects within ~35 deg of the
view axis escapes through the opening, which is why the dark region is sizeable.
