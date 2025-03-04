use serde::{Serialize, Deserialize};

/// Represents a patient in the vehicle routing problem.
/// 
/// This struct contains details about a patient's service requirements and location.
/// 
/// # Fields
/// - `demand`: The load or demand required by the patient.
/// - `start_time`: The earliest time the patient can be serviced.
/// - `end_time`: The latest time by which the patient must be serviced.
/// - `care_time`: The duration of care provided to the patient.
/// - `x_coord`: The X coordinate of the patient's location.
/// - `y_coord`: The Y coordinate of the patient's location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patient {
    pub demand: f64,
    pub start_time: f64,
    pub end_time: f64,
    pub care_time: f64,
    pub x_coord: f64,
    pub y_coord: f64,
}
