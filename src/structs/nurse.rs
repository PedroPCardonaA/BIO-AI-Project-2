use serde::{Serialize, Deserialize};

/// Represents a nurse in the vehicle routing problem.
/// 
/// This struct stores information about a nurse including capacity, current load, and timing metrics
/// such as the current travel time and the total time accumulated.
/// 
/// # Fields
/// - `capacity`: The maximum load the nurse can handle.
/// - `current_load`: The current load assigned to the nurse.
/// - `current_travel_time`: The cumulative travel time incurred by the nurse.
/// - `current_total_time`: The total time (including travel, waiting, and service time) accumulated by the nurse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nurse {
    capacity: u32,
    current_load: u32,
    current_travel_time: f64,
    current_total_time: f64,
}

impl Nurse {
    /// Creates a new `Nurse` with the specified capacity, current load, travel time, and total time.
    ///
    /// # Parameters
    /// - `capacity`: The maximum capacity of the nurse.
    /// - `current_load`: The initial load of the nurse.
    /// - `current_travel_time`: The initial travel time (typically 0.0).
    /// - `current_total_time`: The initial total time (typically 0.0).
    ///
    /// # Returns
    /// A new instance of `Nurse`.
    pub fn new(capacity: u32, current_load: u32, current_travel_time: f64, current_total_time: f64) -> Nurse {
        Nurse {
            capacity,
            current_load,
            current_travel_time,
            current_total_time,
        }
    }

    /// Returns the capacity of the nurse.
    pub fn get_capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns the current load of the nurse.
    pub fn get_current_load(&self) -> u32 {
        self.current_load
    }

    /// Sets the current load of the nurse.
    ///
    /// # Parameters
    /// - `current_load`: The new current load value.
    pub fn set_current_load(&mut self, current_load: u32) {
        self.current_load = current_load;
    }
}