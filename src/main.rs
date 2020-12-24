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

    fn scale(&self, t: f64) -> Vec3 {
        Vec3::new(self.x * t, self.y * t, self.z * t)
    }

    fn add(&self, v: &Vec3) -> Vec3 {
        Vec3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }

    fn sub(&self, v: &Vec3) -> Vec3 {
        Vec3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }

    fn mul(&self, v: &Vec3) -> Vec3 {
        Vec3::new(self.x * v.x, self.y * v.y, self.z * v.z)
    }

    fn write(&self, rgb_range: u16, samples_per_pixel: i8) -> () {
        let scaled = self.scale(1. / (samples_per_pixel as f64));

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
        self.scale(1.0 / l)
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

    fn rand_unit_vector(rng: &mut rand::prelude::ThreadRng) -> Vec3 {
        Vec3::rand_in_unit_sphere(rng).unit()
    }

    fn near_zero(&self) -> bool {
        let s: f64 = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    fn reflect(&self, normal: &Vec3) -> Vec3 {
        self.sub(&normal.scale(2. * self.dot(normal)))
    }

    fn refract(&self, normal: &Vec3, eta_before: f64, eta_after: f64) -> Vec3 {
        let eta_ratio = eta_before / eta_after;
        let cos_theta = f64::min(self.scale(-1.).dot(&normal), 1.);
        let r_out_perp = self.add(&normal.scale(cos_theta)).scale(eta_ratio);
        let r_out_parallel = normal.scale((-r_out_perp.length_squared()).abs().sqrt());
        r_out_perp.add(&r_out_parallel)
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
        self.origin.add(&self.direction.scale(t))
    }
}

trait Material {
    fn scatter<'a>(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        pos: &'a Vec3,
        normal: Vec3,
        r: &Ray,
    ) -> (&Color, Ray<'a>);
}

struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    fn new(albedo: Color) -> Lambertian {
        Lambertian { albedo: albedo }
    }
}

impl Material for Lambertian {
    fn scatter<'a>(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        pos: &'a Vec3,
        normal: Vec3,
        _r: &Ray,
    ) -> (&Color, Ray<'a>) {
        let d = normal.add(&Vec3::rand_unit_vector(rng));

        let scatter_direction = if d.near_zero() { normal } else { d };

        let scattered = Ray {
            origin: &pos,
            direction: scatter_direction,
        };

        (&self.albedo, scattered)
    }
}

struct Metal {
    albedo: Color,
    fuzz: f64,
}

impl Metal {
    fn new(albedo: Color, fuzz: f64) -> Metal {
        let fuzz = if fuzz < 1. {
            if fuzz < 0. {
                0.
            } else {
                fuzz
            }
        } else {
            1.
        };
        Metal {
            albedo: albedo,
            fuzz: fuzz,
        }
    }
}

impl Material for Metal {
    fn scatter<'a>(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        pos: &'a Vec3,
        normal: Vec3,
        r: &Ray,
    ) -> (&Color, Ray<'a>) {
        let reflected = r.direction.reflect(&normal);

        let scattered = Ray {
            origin: &pos,
            direction: reflected.add(&Vec3::rand_in_unit_sphere(rng).scale(self.fuzz)),
        };

        (&self.albedo, scattered)
    }
}

struct Hit<'a> {
    p: Point,
    normal: Vec3,
    t: f64,
    front_face: bool,
    material: &'a (dyn Material + 'a),
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

struct Sphere<'a> {
    x: Point,
    r: f64,
    material: &'a (dyn Material + 'a),
}

impl<'a> Hittable for Sphere<'a> {
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
                let outward_normal = p.sub(&self.x).scale(1. / self.r);
                let front_face = ray.direction.dot(&outward_normal) < 0.;
                Some(Hit {
                    t: root,
                    p: p,
                    normal: if front_face {
                        outward_normal
                    } else {
                        outward_normal.scale(-1.)
                    },
                    front_face: front_face,
                    material: self.material,
                })
            }
        }
    }
}

type World<'a> = Vec<&'a (dyn Hittable + 'a)>;

fn any_hit<'a>(world: &World<'a>, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit<'a>> {
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

struct Camera {
    origin: Vec3,
    lower_left_corner: Point,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new(origin: Vec3, width: f64, height: f64, focal_length: f64) -> Camera {
        let horizontal = Vec3::new(width, 0., 0.);
        let vertical = Vec3::new(0., height, 0.);
        let lower_left_corner = origin
            .sub(&horizontal.scale(1. / 2.))
            .sub(&vertical.scale(1. / 2.))
            .sub(&Vec3::new(0., 0., focal_length));

        Camera {
            origin: origin,
            lower_left_corner: lower_left_corner,
            horizontal: horizontal,
            vertical: vertical,
        }
    }

    fn get_ray(&self, u: f64, v: f64) -> Ray {
        Ray {
            origin: &self.origin,
            direction: self
                .lower_left_corner
                .add(&self.horizontal.scale(u))
                .add(&self.vertical.scale(v))
                .sub(&self.origin),
        }
    }
}

fn ray_color<'a>(
    rng: &mut rand::prelude::ThreadRng,
    remaining_bounces: u8,
    world: &World<'a>,
    r: &Ray,
) -> Color {
    if remaining_bounces <= 0 {
        Vec3::new(0., 0., 0.)
    } else {
        match any_hit(world, r, 0.001, f64::MAX) {
            Some(hit) => {
                let (attenuation, scattered) = hit.material.scatter(rng, &hit.p, hit.normal, r);
                let c = ray_color(rng, remaining_bounces - 1, world, &scattered);
                c.mul(attenuation)
            }
            None => {
                let u = r.direction.unit();
                let t = 0.5 * (u.y + 1.0);
                let v1 = Vec3::new(1.0, 1.0, 1.0).scale(1.0 - t);
                let v2 = Vec3::new(0.5, 0.7, 1.0).scale(t);

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
    let camera = Camera::new(origin, viewport_width, viewport_height, focal_length);

    //world
    let material_ground = Lambertian::new(Vec3::new(0.8, 0.8, 0.0));
    let material_center = Lambertian::new(Vec3::new(0.7, 0.3, 0.3));
    let material_left = Metal::new(Vec3::new(0.8, 0.8, 0.8), 0.3);
    let material_right = Metal::new(Vec3::new(0.8, 0.6, 0.2), 1.0);

    let s1 = Sphere {
        x: Vec3::new(0.0, -100.5, -1.0),
        r: 100.,
        material: &material_ground,
    };

    let s2 = Sphere {
        x: Vec3::new(0.0, 0.0, -1.0),
        r: 0.5,
        material: &material_center,
    };

    let s3 = Sphere {
        x: Vec3::new(-1.0, 0.0, -1.0),
        r: 0.5,
        material: &material_left,
    };

    let s4 = Sphere {
        x: Vec3::new(1.0, 0.0, -1.0),
        r: 0.5,
        material: &material_right,
    };

    let world: World = vec![&s1, &s2, &s3, &s4];

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
