use pbr::ProgressBar;
use rand::Rng;
use std::io::stderr;

const PI: f64 = std::f64::consts::PI;

#[derive(Clone)]
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

    fn mul(&self, v: &Vec3) -> Vec3 {
        Vec3::new(self.x * v.x, self.y * v.y, self.z * v.z)
    }

    fn write(&self, rgb_range: u16, samples_per_pixel: i8) -> () {
        let scaled = self / (samples_per_pixel as f64);

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
        self / self.length()
    }

    fn dot(&self, v: &Vec3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    fn cross(&self, v: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * v.z - self.z * v.y,
            y: self.z * v.x - self.x * v.z,
            z: self.x * v.y - self.y * v.x,
        }
    }

    fn rand_in_unit_sphere(rng: &mut rand::prelude::ThreadRng) -> Vec3 {
        let mut test = Vec3::new(rng.gen(), rng.gen(), rng.gen());
        while test.length_squared() >= 1. {
            test = Vec3::rand_in_unit_sphere(rng)
        }
        test
    }

    fn rand_unit_vector(rng: &mut rand::prelude::ThreadRng) -> Vec3 {
        Vec3::rand_in_unit_sphere(rng).unit()
    }

    fn near_zero(&self) -> bool {
        let s: f64 = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    fn reflect(&self, normal: &Vec3) -> Vec3 {
        self - (normal * 2. * self.dot(normal))
    }

    fn refract(&self, normal: &Vec3, refraction_ratio: f64) -> Vec3 {
        let unit = self.unit();
        let cos_theta = f64::min((-&unit).dot(&normal), 1.);
        let r_out_perp = (unit + normal * cos_theta) * refraction_ratio;
        let r_out_parallel = -normal * (((1. - r_out_perp.length_squared()).abs()).sqrt());
        r_out_perp + r_out_parallel
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Neg for &Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Add<Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Add<&Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Add<Vec3> for &Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Sub<&Vec3> for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Sub<Vec3> for &Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Sub<&Vec3> for &Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl std::ops::Mul<f64> for &Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: rhs.x * self,
            y: rhs.y * self,
            z: rhs.z * self,
        }
    }
}

impl std::ops::Mul<&Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: rhs.x * self,
            y: rhs.y * self,
            z: rhs.z * self,
        }
    }
}

impl std::ops::Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl std::ops::Div<f64> for &Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl std::ops::Div<Vec3> for f64 {
    type Output = Vec3;
    fn div(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: rhs.x / self,
            y: rhs.y / self,
            z: rhs.z / self,
        }
    }
}

impl std::ops::Div<&Vec3> for f64 {
    type Output = Vec3;
    fn div(self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: rhs.x / self,
            y: rhs.y / self,
            z: rhs.z / self,
        }
    }
}

type Point = Vec3;
type Color = Vec3;

struct Ray {
    origin: Point,
    direction: Vec3,
}

impl Ray {
    fn at(&self, t: f64) -> Vec3 {
        &self.origin + &self.direction * t
    }
}

trait Material {
    fn scatter<'a>(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        pos: Vec3,
        normal: Vec3,
        front_face: bool,
        r: &Ray,
    ) -> (&Color, Ray);
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
        pos: Vec3,
        normal: Vec3,
        _front_face: bool,
        _r: &Ray,
    ) -> (&Color, Ray) {
        let d = &normal + Vec3::rand_unit_vector(rng);

        let scatter_direction = if d.near_zero() { normal } else { d };

        let scattered = Ray {
            origin: pos,
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
        pos: Vec3,
        normal: Vec3,
        _front_face: bool,
        r: &Ray,
    ) -> (&Color, Ray) {
        let reflected = r.direction.reflect(&normal);

        let scattered = Ray {
            origin: pos,
            direction: reflected + (Vec3::rand_in_unit_sphere(rng) * self.fuzz),
        };

        (&self.albedo, scattered)
    }
}

struct Dielectric {
    refractive_index: f64,
    albedo: Vec3,
}

impl Dielectric {
    fn new(refractive_index: f64) -> Dielectric {
        Dielectric {
            refractive_index: refractive_index,
            albedo: Vec3::new(1., 1., 1.),
        }
    }

    fn reflectance(cosine: f64, refraction_ratio: f64) -> f64 {
        let r0 = (1. - refraction_ratio) / (1. + refraction_ratio);
        let r0 = r0 * r0;
        r0 + (1. - r0) * ((1. - cosine).powf(5.))
    }
}

impl Material for Dielectric {
    fn scatter<'a>(
        &self,
        rng: &mut rand::prelude::ThreadRng,
        pos: Vec3,
        normal: Vec3,
        front_face: bool,
        r: &Ray,
    ) -> (&Color, Ray) {
        let refraction_ratio = if front_face {
            1. / self.refractive_index
        } else {
            self.refractive_index
        };

        let unit_direction = r.direction.unit();
        let cos_theta = f64::min(-unit_direction.dot(&normal), 1.);
        let sin_theta = (1. - cos_theta * cos_theta).sqrt();

        let cannot_refract = (refraction_ratio * sin_theta) > 1.;

        let direction =
            if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > rng.gen() {
                unit_direction.reflect(&normal)
            } else {
                unit_direction.refract(&normal, refraction_ratio)
            };

        let scattered = Ray {
            origin: pos,
            direction: direction,
        };

        (&self.albedo, scattered)
    }
}

struct Hit<'a> {
    p: Point,
    normal: Vec3,
    t: f64,
    front_face: bool,
    material: &'a dyn Material,
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

struct Sphere<'a> {
    center: Point,
    r: f64,
    material: &'a dyn Material,
}

impl<'a> Hittable for Sphere<'a> {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let oc = &ray.origin - &self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(&ray.direction);
        let c = oc.length_squared() - self.r * self.r;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        } else {
            let sqrtd = discriminant.sqrt();
            let mut root = (-half_b - sqrtd) / a;
            if root < t_min || t_max < root {
                root = (-half_b + sqrtd) / a;
                if root < t_min || t_max < root {
                    return None;
                }
            }

            let p = ray.at(root);
            let outward_normal = (&p - &self.center) / self.r;
            let front_face = ray.direction.dot(&outward_normal) < 0.0;
            Some(Hit {
                t: root,
                p: p,
                normal: if front_face {
                    outward_normal
                } else {
                    -outward_normal
                },
                front_face: front_face,
                material: self.material,
            })
        }
    }
}

type World<'a> = Vec<&'a dyn Hittable>;

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
    fn new(look_from: Point, look_at: Point, vup: Vec3, vfov: f64, aspect_ratio: f64) -> Camera {
        let theta = vfov * PI / 180.;
        let h = f64::tan(theta / 2.);
        let viewport_height = 2. * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (&look_from - look_at).unit();
        let u = (vup.cross(&w)).unit();
        let v = w.cross(&u);

        let origin = look_from;
        let horizontal = viewport_width * u;
        let vertical = viewport_height * v;
        let lower_left_corner = &origin - &horizontal / 2. - &vertical / 2. - w;

        Camera {
            origin: origin,
            lower_left_corner: lower_left_corner,
            horizontal: horizontal,
            vertical: vertical,
        }
    }

    fn get_ray(&self, u: f64, v: f64) -> Ray {
        let direction =
            &self.lower_left_corner + &self.horizontal * u + &self.vertical * v - &self.origin;
        Ray {
            direction: direction,
            origin: self.origin.clone(),
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
                let (attenuation, scattered) =
                    hit.material
                        .scatter(rng, hit.p, hit.normal, hit.front_face, r);
                let c = ray_color(rng, remaining_bounces - 1, world, &scattered);
                c.mul(attenuation)
            }
            None => {
                let u = r.direction.unit();
                let t = 0.5 * (u.y + 1.0);
                let v1 = Vec3::new(0.7, 1.0, 1.0) * (1.0 - t);
                let v2 = Vec3::new(0.5, 0.7, 1.0) * t;

                let v3 = if t % 0.1 < 0.05 {
                    Vec3::new(0.3, 0.0, 0.0)
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                };

                v1 + v2 + v3
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
    let camera = Camera::new(
        Vec3::new(-2., 2., 1.),
        Vec3::new(0., 0., -1.),
        Vec3::new(0., 1., 0.),
        20.,
        aspect_ratio,
    );

    //world
    let material_ground = Lambertian::new(Vec3::new(0.8, 0.8, 0.0));
    let material_center = Lambertian::new(Vec3::new(0.1, 0.2, 0.5));
    let material_left = Dielectric::new(1.5);
    let material_right = Metal::new(Vec3::new(0.8, 0.6, 0.2), 0.0);

    let s1 = Sphere {
        center: Vec3::new(0.0, -100.5, -1.0),
        r: 100.,
        material: &material_ground,
    };

    let s2 = Sphere {
        center: Vec3::new(0.0, 0.0, -1.0),
        r: 0.5,
        material: &material_center,
    };

    let s3 = Sphere {
        center: Vec3::new(-1.0, 0.0, -1.0),
        r: 0.5,
        material: &material_left,
    };

    let s4 = Sphere {
        center: Vec3::new(-1.0, 0.0, -1.0),
        r: -0.45,
        material: &material_left,
    };

    let s5 = Sphere {
        center: Vec3::new(1.0, 0.0, -1.0),
        r: 0.5,
        material: &material_right,
    };

    let world: World = vec![&s1, &s2, &s3, &s4, &s5];

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
                pixel_color = pixel_color + ray_color(&mut rng, max_bounces, &world, &ray);
            }

            pixel_color.write(rgb_range, samples_per_pixel);
        }

        pb.inc();
    }
}
