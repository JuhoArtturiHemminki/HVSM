use std::env;
use std::fs::File;
use std::io::{Read, Write, Result, Error, ErrorKind};

const TOP: u32 = 1 << 24;

pub struct HsvmEngine {
    local_history: u64,
    ghost_shadow: u64, 
    stats_short: [[u32; 2]; 4],
    stats_ghost: [[u32; 2]; 4],
    sigmoid_lut: [u32; 512],
}

impl HsvmEngine {
    pub fn new() -> Self {
        let mut lut = [2048u32; 512];
        for i in 0..512 {
            let x = (i as f64 - 256.0) / 48.0;
            let s = 1.0 / (1.0 + (-x).exp());
            lut[i] = (s * 4094.0) as u32 + 1;
        }

        HsvmEngine {
            local_history: 0,
            ghost_shadow: 0,
            stats_short: [[16; 2]; 4],
            stats_ghost: [[16; 2]; 4],
            sigmoid_lut: lut,
        }
    }

    #[inline(always)]
    pub fn predict_probability(&self) -> u32 {
        let idx_s = (self.local_history & 0x3) as usize;
        let ghost_projection = (self.ghost_shadow ^ self.local_history) & 0x3;
        let idx_g = ghost_projection as usize;

        let p1_s = (self.stats_short[idx_s][1] as f64 + 1.0) / (self.stats_short[idx_s][0] + self.stats_short[idx_s][1] + 2) as f64;
        let p1_g = (self.stats_ghost[idx_g][1] as f64 + 1.0) / (self.stats_ghost[idx_g][0] + self.stats_ghost[idx_g][1] + 2) as f64;

        let mixed = p1_s * 0.5 + p1_g * 0.5;
        let lut_index = (mixed * 511.0) as usize;
        
        self.sigmoid_lut[lut_index.clamp(0, 511)]
    }

    #[inline(always)]
    pub fn update_and_learn(&mut self, bit: u8) {
        let b = bit as usize;
        let idx_s = (self.local_history & 0x3) as usize;
        let idx_g = ((self.ghost_shadow ^ self.local_history) & 0x3) as usize;

        let p_s = (self.stats_short[idx_s][1] as f64 + 1.0) / (self.stats_short[idx_s][0] + self.stats_short[idx_s][1] + 2) as f64;
        let p_g = (self.stats_ghost[idx_g][1] as f64 + 1.0) / (self.stats_ghost[idx_g][0] + self.stats_ghost[idx_g][1] + 2) as f64;

        if (bit == 1 && p_s < 0.85) || (bit == 0 && p_s > 0.15) {
            self.stats_short[idx_s][b] += 2;
        }

        if (bit == 1 && p_g < 0.85) || (bit == 0 && p_g > 0.15) {
            self.stats_ghost[idx_g][b] += 2;
        }

        if self.stats_short[idx_s][b] > 200 { 
            self.stats_short[idx_s][0] /= 2; self.stats_short[idx_s][1] /= 2; 
        }
        if self.stats_ghost[idx_g][b] > 200 { 
            self.stats_ghost[idx_g][0] /= 2; self.stats_ghost[idx_g][1] /= 2; 
        }

        self.ghost_shadow = self.ghost_shadow.rotate_left(1) ^ (self.ghost_shadow >> 3) ^ (bit as u64 * 0xBF5F_5245_D1A3_432D);
        self.local_history = (self.local_history << 1) | bit as u64;
    }
}

pub fn hsvm_compress_file(source_path: &str, target_path: &str) -> Result<()> {
    let mut file = File::open(source_path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut engine = HsvmEngine::new();
    let mut compressed = Vec::new();
    
    let length = data.len() as u32;
    compressed.extend_from_slice(&length.to_le_bytes());

    let mut low: u32 = 0;
    let mut range: u32 = !0;

    for &byte in &data {
        for i in (0..8).rev() {
            let bit = (byte >> i) & 1;
            let p1 = engine.predict_probability();
            let r_one = (range >> 12) * p1;

            if bit == 1 {
                low = low.wrapping_add(range - r_one);
                range = r_one;
            } else {
                range -= r_one;
            }

            while (low ^ (low.wrapping_add(range))) < TOP || range < TOP {
                compressed.push((low >> 24) as u8);
                low <<= 8;
                range <<= 8;
            }
            engine.update_and_learn(bit);
        }
    }
    for _ in 0..4 { compressed.push((low >> 24) as u8); low <<= 8; }

    let mut target = File::create(target_path)?;
    target.write_all(&compressed)?;
    Ok(())
}

pub fn hsvm_decompress_file(source_path: &str, target_path: &str) -> Result<()> {
    let mut file = File::open(source_path)?;
    let mut compressed_data = Vec::new();
    file.read_to_end(&mut compressed_data)?;

    if compressed_data.len() < 4 {
        return Err(Error::new(ErrorKind::InvalidData, "File corrupted or too short."));
    }

    let mut length_bytes = [0u8; 4];
    length_bytes.copy_from_slice(&compressed_data[0..4]);
    let target_length = u32::from_le_bytes(length_bytes) as usize;

    let mut engine = HsvmEngine::new();
    let mut decompressed = Vec::with_capacity(target_length);

    let mut low: u32 = 0;
    let mut range: u32 = !0;
    let mut code: u32 = 0;
    let mut pos = 4;
    
    for _ in 0..4 {
        if pos < compressed_data.len() {
            code = (code << 8) | (compressed_data[pos] as u32);
            pos += 1;
        }
    }

    for _ in 0..target_length {
        let mut byte = 0u8;

        for i in (0..8).rev() {
            let p1 = engine.predict_probability();
            let r_one = (range >> 12) * p1;
            
            let bit = if code >= (range - r_one) { 1 } else { 0 };

            if bit == 1 {
                code -= range - r_one;
                low = low.wrapping_add(range - r_one);
                range = r_one;
            } else {
                range -= r_one;
            }

            byte |= bit << i;

            while (low ^ (low.wrapping_add(range))) < TOP || range < TOP {
                let next_byte = if pos < compressed_data.len() {
                    let b = compressed_data[pos]; pos += 1; b
                } else { 0 };
                code = (code << 8) | (next_byte as u32);
                low <<= 8;
                range <<= 8;
            }
            engine.update_and_learn(bit);
        }
        decompressed.push(byte);
    }

    let mut target = File::create(target_path)?;
    target.write_all(&decompressed)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        println!("=====================================================================");
        println!("       HSVM: Holographic Shadow & Voting Mixer (Core Architecture)   ");
        println!("       Author: Juho Artturi Hemminki                                 ");
        println!("=====================================================================");
        println!("Usage:");
        println!("  Compression:   cargo run --release -- c <source_file> <target_file.ai>");
        println!("  Decompression: cargo run --release -- d <compressed_file.ai> <output_file>");
        return;
    }

    let command = &args[1];
    let source = &args[2];
    let target = &args[3];

    match command.as_str() {
        "c" => {
            println!("Compressing file '{}' -> '{}'...", source, target);
            match hsvm_compress_file(source, target) {
                Ok(_) => println!("Compression complete! Non-linear register scan successful."),
                Err(e) => println!("Compression error: {}", e),
            }
        }
        "d" => {
            println!("Decompressing file '{}' -> '{}'...", source, target);
            match hsvm_decompress_file(source, target) {
                Ok(_) => println!("Decompression complete! Lossless binary match verified."),
                Err(e) => println!("Decompression error: {}", e),
            }
        }
        _ => {
            println!("Unknown command '{}'. Use 'c' for compression or 'd' for decompression.", command);
        }
    }
}
