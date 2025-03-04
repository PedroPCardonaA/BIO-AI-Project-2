use serde::{Serialize, Deserialize};

/// Represents the depot in the vehicle routing problem.
/// 
/// This struct stores the depot's return time along with its geographical coordinates.
/// 
/// # Fields
/// - `return_time`: The last time allowed for returning to the depot.
/// - `x_coord`: The X coordinate for the depot's placement.
/// - `y_coord`: The Y coordinate for the depot's placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Depot {
    pub return_time: f64,
    pub x_coord: f64,
    pub y_coord: f64,
}