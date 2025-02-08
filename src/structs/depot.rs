use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Depot {
    pub return_time: f64,
    pub x_coord: f64,
    pub y_coord: f64,
}