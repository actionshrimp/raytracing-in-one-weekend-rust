struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
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

    fn write(&self) -> () {
        let ir = (255.999 * self.x) as i32;
        let ig = (255.999 * self.y) as i32;
        let ib = (255.999 * self.z) as i32;
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
}

type Point = Vec3;
type Color = Vec3;

struct Ray<'a> {
    origin: &'a Point,
    direction: &'a Vec3,
}

impl<'a> Ray<'a> {
    fn at(&'a self, t: f64) -> Vec3 {
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
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

struct Sphere {
    x: Point,
    r: f64,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let oc = ray.origin.sub(&self.x);
        let a = ray.direction.dot(ray.direction);
        let half_b = oc.dot(ray.direction);
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

fn ray_color(r: &Ray) -> Color {
    let sphere = Sphere {
        x: Vec3::new(0.0, 0.0, -1.0),
        r: 0.5,
    };

    match sphere.hit(r, 0., 1.) {
        Some(hit) => {
            let n = r.at(hit.t).sub(&Vec3::new(0.0, 0.0, -1.0)).unit();
            Vec3::new(n.x + 1.0, n.y + 1.0, n.z + 1.0).mul(0.5)
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

fn main() {
    let aspect_ratio: f64 = 16.0 / 9.0;
    let image_width: u16 = 400;
    let image_height: u16 = (image_width as f64 / aspect_ratio) as u16;
    let rgb_max: u8 = 255;

    //camera
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = Vec3::new(0.0, 0.0, 0.0);
    let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
    let vertical = Vec3::new(0.0, viewport_height, 0.0);
    let deep = Vec3::new(0.0, 0.0, focal_length);
    let lower_left_corner = origin
        .sub(&horizontal.mul(1.0 / 2.0))
        .sub(&vertical.mul(1.0 / 2.0))
        .sub(&deep);

    println!("P3");
    println!("{} {}", image_width, image_height);
    println!("{}", rgb_max);

    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let u = (i as f64) / ((image_width - 1) as f64);
            let v = (j as f64) / ((image_height - 1) as f64);
            let direction = lower_left_corner
                .add(&horizontal.mul(u))
                .add(&vertical.mul(v))
                .sub(&origin);

            let r = Ray {
                origin: &origin,
                direction: &direction,
            };
            let pixel_color = ray_color(&r);
            pixel_color.write()
        }
    }
}
