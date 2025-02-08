use serde::{Serialize, Deserialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nurse {
    capacity: u32,
    current_load: u32,
    current_travel_time: f64,
    current_total_time: f64
}

impl Nurse {
    pub fn new(capacity: u32, current_load: u32, current_travel_time: f64, current_total_time: f64) -> Nurse {
        Nurse {
            capacity,
            current_load,
            current_travel_time,
            current_total_time
        }
    }

    pub fn get_capacity(&self) -> u32 {
        self.capacity
    }

    pub fn get_current_load(&self) -> u32 {
        self.current_load
    }

    pub fn set_current_load(&mut self, current_load: u32) {
        self.current_load = current_load;
    }

    pub fn get_current_travel_time(&self) -> f64 {
        self.current_travel_time
    }

    pub fn set_current_travel_time(&mut self, current_travel_time: f64) {
        self.current_travel_time = current_travel_time;
    }
}