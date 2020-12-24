# Raytracer

Following [Ray Tracing In One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html), transposing to rust.

## Running it

Outputs PPM format by default:

    cargo run --release > out.ppm

For png output (easier to actually view!):

    apt install netpbm
    cargo run --release | pnmtopng > test.png
