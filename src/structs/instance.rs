use std::collections::HashMap;

use super::{depot::Depot, nurse::Nurse, patient::Patient};
use serde::{Serialize, Deserialize};
#[derive(Debug, Serialize, Deserialize)]
pub struct Instance {
    pub instance_name: String,         // The name of the instance.
    pub nbr_nurses: u32,               // The number of nurses available.
    pub capacity_nurse: u32,           // The capacity of a nurse.
    pub benchmark: f64,                // The benchmark objective value for the instance.
    pub depot: Depot,                // Depot-related information.
    pub patients: HashMap<String, Patient>,       // List of patients.
    pub travel_times: Vec<Vec<f64>>,   // The travel time matrix.
    #[serde(default)]
    pub nurses: Vec<Nurse>,            // List of nurses.
}

impl Instance {
    // Add nbr_nurses of nurses to the nurses vector when nbr_nurses is greater than 0
    pub fn add_nurses(&mut self) {
        if self.nbr_nurses > 0 {
            for _ in 0..self.nbr_nurses {
                self.nurses.push(Nurse::new(self.capacity_nurse, 0, 0.0, 0.0));
            }
        }
    }
}
