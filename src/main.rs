use pbr::ProgressBar;
use rand::Rng;
use std::io::stderr;

struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

impl Vec3 {
    fn zero() -> Vec3 {
        Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }

    fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x: x, y: y, z: z }
    }

    fn mul(&self, t: f64) -> Vec3 {
        Vec3::new(self.x * t, self.y * t, self.z * t)
    }

    fn add(&self, v: &Vec3) -> Vec3 {
        Vec3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }

    fn sub(&self, v: &Vec3) -> Vec3 {
        Vec3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }

    fn write(&self, rgb_range: u16, samples_per_pixel: i8) -> () {
        let scaled = self.mul(1. / (samples_per_pixel as f64));

        let rgbf = rgb_range as f64;

        let ir = (rgbf * clamp(scaled.x.sqrt(), 0., 0.999)) as u8;
        let ig = (rgbf * clamp(scaled.y.sqrt(), 0., 0.999)) as u8;
        let ib = (rgbf * clamp(scaled.z.sqrt(), 0., 0.999)) as u8;

        println!("{} {} {}", ir, ig, ib)
    }

    fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    fn unit(&self) -> Vec3 {
        let l = self.length();
        self.mul(1.0 / l)
    }

    fn dot(&self, v: &Vec3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    fn rand_in_unit_sphere(rng: &mut rand::prelude::ThreadRng) -> Vec3 {
        let test = Vec3::new(rng.gen(), rng.gen(), rng.gen());
        if test.length_squared() >= 1. {
            Vec3::rand_in_unit_sphere(rng)
        } else {
            test
        }
    }
}

type Point = Vec3;
type Color = Vec3;

struct Ray<'a> {
    origin: &'a Point,
    direction: Vec3,
}

impl<'a> Ray<'a> {
    fn at(&self, t: f64) -> Vec3 {
        self.origin.add(&self.direction.mul(t))
    }
}

struct Hit {
    p: Point,
    normal: Vec3,
    t: f64,
    front_face: bool,
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

struct Sphere {
    x: Point,
    r: f64,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let oc = ray.origin.sub(&self.x);
        let a = ray.direction.dot(&ray.direction);
        let half_b = oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.r * self.r;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            None
        } else {
            let sqrtd = discriminant.sqrt();
            let root = (-half_b - sqrtd) / a;
            if root < t_min || t_max < root {
                None
            } else {
                let p = ray.at(root);
                let outward_normal = p.sub(&self.x).mul(1. / self.r);
                let front_face = ray.direction.dot(&outward_normal) < 0.;
                Some(Hit {
                    t: root,
                    p: p,
                    normal: if front_face {
                        outward_normal
                    } else {
                        outward_normal.mul(-1.)
                    },
                    front_face: front_face,
                })
            }
        }
    }
}

type World = Vec<Box<dyn Hittable>>;

fn any_hit(world: &World, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
    let mut ret: Option<Hit> = None;

    for x in world.iter() {
        let hit = match &ret {
            None => x.hit(ray, t_min, t_max),
            Some(h) => x.hit(ray, t_min, h.t),
        };

        match hit {
            None => (),
            Some(h) => ret = Some(h),
        }
    }

    ret
}

struct Camera<'a> {
    origin: &'a Vec3,
    lower_left_corner: Point,
    horizontal: Vec3,
    vertical: Vec3,
}

impl<'a> Camera<'a> {
    fn new(origin: &Vec3, width: f64, height: f64, focal_length: f64) -> Camera {
        let horizontal = Vec3::new(width, 0., 0.);
        let vertical = Vec3::new(0., height, 0.);
        Camera {
            origin: &origin,
            lower_left_corner: origin
                .sub(&horizontal.mul(1. / 2.))
                .sub(&vertical.mul(1. / 2.))
                .sub(&Vec3::new(0., 0., focal_length)),
            horizontal: horizontal,
            vertical: vertical,
        }
    }

    fn get_ray(&self, u: f64, v: f64) -> Ray {
        Ray {
            origin: self.origin,
            direction: self
                .lower_left_corner
                .add(&self.horizontal.mul(u))
                .add(&self.vertical.mul(v))
                .sub(self.origin),
        }
    }
}

fn ray_color(
    rng: &mut rand::prelude::ThreadRng,
    remaining_bounces: u8,
    world: &World,
    r: &Ray,
) -> Color {
    if remaining_bounces <= 0 {
        Vec3::new(0., 0., 0.)
    } else {
        match any_hit(world, r, 0., f64::MAX) {
            Some(hit) => {
                let target = hit.p.add(&hit.normal).add(&Vec3::rand_in_unit_sphere(rng));
                let c = ray_color(
                    rng,
                    remaining_bounces - 1,
                    world,
                    &Ray {
                        origin: &hit.p,
                        direction: target.sub(&hit.p),
                    },
                );
                c.mul(0.5)
            }
            None => {
                let u = r.direction.unit();
                let t = 0.5 * (u.y + 1.0);
                let v1 = Vec3::new(1.0, 1.0, 1.0).mul(1.0 - t);
                let v2 = Vec3::new(0.5, 0.7, 1.0).mul(t);

                v1.add(&v2)
            }
        }
    }
}

fn main() {
    let aspect_ratio: f64 = 16.0 / 9.0;
    let image_width: u16 = 400;
    let image_height: u16 = (image_width as f64 / aspect_ratio) as u16;
    let rgb_range = 256;
    let samples_per_pixel = 100;

    //camera
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let camera = Camera::new(&origin, viewport_width, viewport_height, focal_length);

    //world
    let world: Vec<Box<dyn Hittable>> = vec![
        Box::new(Sphere {
            x: Vec3::new(0.0, 0.0, -1.0),
            r: 0.5,
        }),
        Box::new(Sphere {
            x: Vec3::new(0.0, -100.5, -1.0),
            r: 100.,
        }),
    ];

    let total_steps = image_height as u64;
    let mut pb = ProgressBar::on(stderr(), total_steps);

    println!("P3");
    println!("{} {}", image_width, image_height);
    println!("{}", rgb_range - 1);

    let mut rng = rand::thread_rng();
    let max_bounces = 50;

    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let mut pixel_color = Vec3::zero();

            for _s in 0..samples_per_pixel {
                let du: f64 = rng.gen();
                let dv: f64 = rng.gen();

                let u = (i as f64 + du) / ((image_width - 1) as f64);
                let v = (j as f64 + dv) / ((image_height - 1) as f64);

                let ray = camera.get_ray(u, v);
                pixel_color = pixel_color.add(&ray_color(&mut rng, max_bounces, &world, &ray));
            }

            pixel_color.write(rgb_range, samples_per_pixel);
        }

        pb.inc();
    }
}
