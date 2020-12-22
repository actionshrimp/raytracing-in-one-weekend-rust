struct Vec {
    x: f64,
    y: f64,
    z: f64
}

impl Vec {
    fn zero() -> Vec {
        Vec { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn new(x: f64, y: f64, z: f64) -> Vec {
        Vec { x: x, y: y, z: z }
    }

    fn write(&self) -> () {
        let ir = (255.999 * self.x).round() as i32;
        let ig = (255.999 * self.y).round() as i32;
        let ib = (255.999 * self.z).round() as i32;
        println!("{} {} {}", ir, ig, ib)
    }

}

fn main() {
    let image_width: u16 = 256;
    let image_height: u16 = 256;
    let rgb_max: u8 = 255;


    println!("P3");
    println!("{} {}", image_width, image_height);
    println!("{}", rgb_max);


    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let r = (i as f64) / ((image_width - 1) as f64);
            let g = (j as f64) / ((image_height - 1) as f64);
            let b: f64 = 0.25;

            Vec::new(r, g, b).write();
        }
    }
}
