use crate::kmath::*;


// Audio system
// PlayHold (UID, sd)
// Release (UID)

#[derive(Debug, Clone, Copy)]
pub enum AudioCommand {
    PlayHold(u64, SoundDesc),
    Release(u64),
    SetVol(f32),
}

pub fn db_to_vol(db: f32) -> f32 {
    10.0f32.powf(0.05 * db)
}

pub fn vol_to_db(vol: f32) -> f32 {
    20.0f32 * vol.log10()
}

#[derive(Clone, Copy, Debug)]
pub struct SoundDesc {
    pub f: f32,
    pub n: f32,
    pub troll: f32,
    pub ea: f32,
    pub ed: f32,
    pub es: f32,
    pub er: f32,
    pub detune: f32,
    pub voices: f32,
    pub amp: f32,
    pub cut: f32,
    pub cur: f32,
    pub cdt: f32,
    pub cdr: f32,
    pub hard_clip: f32,
}



pub struct Channel {
    pub sd: SoundDesc,
    pub birth: u64,
    pub age: u64,
    pub release_time: Option<u64>,
    pub phases: Vec<f32>,
    pub id: u64,
}

impl Channel {
    pub fn tick(&mut self) -> f32 {
        self.age += 1;

        let sd = self.sd;

        let voices_len = sd.voices.floor() as usize;
        let n_len = sd.n.floor() as usize;

        // pre compression
        let mut acc = 0.0;

        let a_vol = db_to_vol(sd.amp);

        let a_env = env_amplitude(self.sd.ea, self.sd.ed, self.sd.es, self.sd.er, self.age, 44100, self.release_time.map(|x| x - self.birth));

        let a_voices = 1.0 / voices_len as f32;

        for detune_voice_num in 0..voices_len {
            for n in 0..n_len {
                let a_roll = 1.0 / ((n+1) as f32).powf(sd.troll);

                let detune_interval = 2.0f32.powf(sd.detune / 1200.0);
                let f = sd.f * (n + 1) as f32;
                let f = f * detune_interval.powf(detune_voice_num as f32);

                let idx = detune_voice_num * n_len + n;
                self.phases[idx] = (self.phases[idx] + f / 44100.0).fract();
                acc += a_voices * a_env * a_roll * a_vol * (2.0 * PI * self.phases[idx]).sin();
            }
        }


        // now do compression
        // change db value or amplitude value?
        let comp = {
            // let cut_vol = sd.cut;
            // let cdt_vol = sd.cdt;

            let cut_vol = db_to_vol(sd.cut);
            let cdt_vol = db_to_vol(sd.cdt);
            if acc < cut_vol {
                let gain = lerp(sd.cur, 1.0, (sd.cur - acc)/sd.cur);
                gain * acc
            } else if acc > cdt_vol {
                cdt_vol + (acc - cdt_vol) / sd.cdr
            } else {
                acc
            }
        };
        
        let out = comp;

        // Hard clip
        let hc_vol = db_to_vol(self.sd.hard_clip);

        let hclipped = if out > hc_vol {
            hc_vol
        } else if out < -hc_vol {
            -hc_vol
        } else {
            out
        };

        hclipped
    }
}

pub struct Mixer {
    pub out_vol: f32,
    pub sample_count: u64,
    pub channels: Vec<Channel>,
}

impl Default for Mixer {
    fn default() -> Self {
        Mixer {
            out_vol: db_to_vol(-10.0),
            sample_count: 0,
            channels: vec![],
        }
    }
}

impl Mixer {
    // We assume only one playing at a time and unique
    pub fn handle_command(&mut self, com: AudioCommand) {
        println!("handle command {:?}", com);

        match com {
            AudioCommand::PlayHold(id, sd) => {
                let seed = khash(self.sample_count as u32);
                let mut phases = vec![];
                let voices_len = sd.voices.floor() as usize;
                let n_len = sd.n.floor() as usize;
                for i in 0..voices_len {
                    for j in 0..n_len {
                        phases.push(krand(seed + 13414177 * i as u32 + 123997 * j as u32) * 2.0 * PI)
                    }
                }
                self.channels.push(Channel {
                    sd,
                    id,
                    age: 0,
                    birth: self.sample_count,
                    phases,
                    release_time: None,
                })
            },
            AudioCommand::Release(id) => {
                for i in 0..self.channels.len() {
                    if self.channels[i].id == id {
                        self.channels[i].release_time = Some(self.sample_count);
                    }
                }
            },
            AudioCommand::SetVol(v) => self.out_vol = v,
        }
    }

    pub fn tick(&mut self) -> f32 {
        self.sample_count += 1;

        let mut i = self.channels.len();
        if i == 0 { return 0.0 }
        i -= 1;
        let mut acc = 0.0;
        loop {
            acc += self.channels[i].tick();
            if let Some(release_time) = self.channels[i].release_time {
                let n = self.channels[i].age;
                let n_since_release = n - release_time + self.channels[i].birth;
                if n_since_release > (self.channels[i].sd.er * 44100.0) as u64 {
                    println!("removing {}, n since release {}, release samples: {}, n {} releasetime {}", i, n_since_release, (self.channels[i].sd.er * 44100.0) as u64, n, release_time);
                    self.channels.swap_remove(i);

                }
            }

            if i == 0 { break; }
            i -= 1;
        }
        acc * self.out_vol
    }
}



pub fn env_amplitude(a: f32, d: f32, s: f32, r: f32, curr_sample: u64, sample_rate: u64, released_sample: Option<u64>) -> f32 {
    // +1 for useful recursion
    let A = a * sample_rate as f32;
    let D = d * sample_rate as f32;
    let S = s;
    let R = r * sample_rate as f32;

    if let Some(released_on) = released_sample {
        let num_released = curr_sample - released_on;
        let release_value = env_amplitude(a, d, s, r, released_on, sample_rate, None);
        return lerp(release_value, 0.0, num_released as f32 / R).max(0.0);
    }
    if curr_sample as f32 <= A {
        return lerp(0.0, 1.0, curr_sample as f32 / A);
    }
    if curr_sample as f32 <= D + A {
        return lerp(1.0, S, (curr_sample as f32 - A)/D);
    }
    S
}