fn main() {
    let image_width: u16 = 256;
    let image_height: u16 = 256;
    let rgb_max: u8 = 255;


    println!("P3");
    println!("{} {}", image_width, image_height);
    println!("{}", rgb_max);


    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let r = (i as f32) / ((image_width - 1) as f32);
            let g = (j as f32) / ((image_height - 1) as f32);
            let b: f32 = 0.25;

            let ir = (255.999 * r).round() as i32;
            let ig = (255.999 * g).round() as i32;
            let ib = (255.999 * b).round() as i32;

            println!("{} {} {}", ir, ig, ib);
        }
    }
}
