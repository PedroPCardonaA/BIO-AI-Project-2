use serde::{Serialize, Deserialize};
#[derive(Debug,Clone, Serialize, Deserialize)]
pub struct Patient {
    pub demand: f64,                 
    pub start_time: f64,             
    pub end_time: f64,              
    pub care_time: f64,              
    pub x_coord: f64,            
    pub y_coord: f64,                
}