use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use serde_json;
use crate::structs::{instance::Instance, depot::Depot, patient::Patient};

/// Configuration for generating random instances
#[derive(Debug, Clone)]
pub struct InstanceConfig {
    // Instance metadata
    pub instance_name: String,
    
    // Nurse configuration
    pub nbr_nurses_range: (u32, u32),          // (min, max) number of nurses
    pub capacity_nurse_range: (u32, u32),      // (min, max) capacity per nurse
    
    // Depot configuration
    pub depot_return_time_range: (f64, f64),   // (min, max) return time
    pub depot_x_range: (f64, f64),             // (min, max) x coordinate
    pub depot_y_range: (f64, f64),             // (min, max) y coordinate
    
    // Patient configuration
    pub nbr_patients_range: (usize, usize),    // (min, max) number of patients
    pub patient_x_range: (f64, f64),           // (min, max) x coordinate
    pub patient_y_range: (f64, f64),           // (min, max) y coordinate
    pub patient_demand_range: (f64, f64),      // (min, max) demand
    pub patient_care_time_range: (f64, f64),   // (min, max) care time
    
    // Time window configuration
    pub time_window_start_range: (f64, f64),   // (min, max) start time
    pub time_window_duration_range: (f64, f64),// (min, max) duration of time window
    
    // Benchmark (optional, can be calculated or set to 0.0)
    pub benchmark: Option<f64>,
}

impl Default for InstanceConfig {
    fn default() -> Self {
        InstanceConfig {
            instance_name: "random_instance".to_string(),
            nbr_nurses_range: (20, 30),
            capacity_nurse_range: (150, 250),
            depot_return_time_range: (1000.0, 1500.0),
            depot_x_range: (30.0, 50.0),
            depot_y_range: (40.0, 60.0),
            nbr_patients_range: (50, 150),
            patient_x_range: (0.0, 100.0),
            patient_y_range: (0.0, 100.0),
            patient_demand_range: (10.0, 40.0),
            patient_care_time_range: (60.0, 120.0),
            time_window_start_range: (0.0, 800.0),
            time_window_duration_range: (100.0, 1200.0),
            benchmark: None,
        }
    }
}

/// Generates a random instance based on the provided configuration
pub fn generate_random_instance(config: InstanceConfig) -> Instance {
    let mut rng = rand::rng();
    
    // Generate random number of nurses
    let nbr_nurses = rng.random_range(config.nbr_nurses_range.0..=config.nbr_nurses_range.1);
    
    // Generate random capacity
    let capacity_nurse = rng.random_range(config.capacity_nurse_range.0..=config.capacity_nurse_range.1);
    
    // Generate random depot with integer coordinates and return time
    let depot = Depot {
        return_time: rng.random_range(config.depot_return_time_range.0 as i32..=config.depot_return_time_range.1 as i32) as f64,
        x_coord: rng.random_range(config.depot_x_range.0 as i32..=config.depot_x_range.1 as i32) as f64,
        y_coord: rng.random_range(config.depot_y_range.0 as i32..=config.depot_y_range.1 as i32) as f64,
    };
    
    // Generate random number of patients
    let nbr_patients = rng.random_range(config.nbr_patients_range.0..=config.nbr_patients_range.1);
    
    // Generate random patients with integer values
    let mut patients = HashMap::new();
    for i in 1..=nbr_patients {
        let start_time = rng.random_range(config.time_window_start_range.0 as i32..=config.time_window_start_range.1 as i32) as f64;
        let duration = rng.random_range(config.time_window_duration_range.0 as i32..=config.time_window_duration_range.1 as i32) as f64;
        let care_time = rng.random_range(config.patient_care_time_range.0 as i32..=config.patient_care_time_range.1 as i32) as f64;
        
        let patient = Patient {
            x_coord: rng.random_range(config.patient_x_range.0 as i32..=config.patient_x_range.1 as i32) as f64,
            y_coord: rng.random_range(config.patient_y_range.0 as i32..=config.patient_y_range.1 as i32) as f64,
            demand: rng.random_range(config.patient_demand_range.0 as i32..=config.patient_demand_range.1 as i32) as f64,
            start_time,
            end_time: start_time + duration,
            care_time,
        };
        
        patients.insert(i.to_string(), patient);
    }
    
    // Calculate travel times matrix (Euclidean distance)
    let travel_times = calculate_travel_times(&depot, &patients, nbr_patients);
    
    Instance {
        instance_name: config.instance_name,
        nbr_nurses,
        capacity_nurse,
        benchmark: config.benchmark.unwrap_or(0.0),
        depot,
        patients,
        travel_times,
        nurses: Vec::new(), // Will be populated by add_nurses()
    }
}

/// Calculates the travel time matrix using Euclidean distance
/// Index 0 is the depot, indices 1..=n are patients
fn calculate_travel_times(depot: &Depot, patients: &HashMap<String, Patient>, nbr_patients: usize) -> Vec<Vec<f64>> {
    let n = nbr_patients + 1; // +1 for depot
    let mut matrix = vec![vec![0.0; n]; n];
    
    // Create a vector of all locations (depot + patients)
    let mut locations = vec![(depot.x_coord, depot.y_coord)];
    for i in 1..=nbr_patients {
        let patient = &patients[&i.to_string()];
        locations.push((patient.x_coord, patient.y_coord));
    }
    
    // Calculate distances
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let (x1, y1) = locations[i];
                let (x2, y2) = locations[j];
                matrix[i][j] = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
            }
        }
    }
    
    matrix
}

/// Generates a random instance similar to train_0 with slight variations
pub fn generate_train_like_instance(variation_percent: f64) -> Instance {
    let base_config = InstanceConfig {
        instance_name: format!("train_variant_{}", rand::rng().random_range(0..1000)),
        nbr_nurses_range: (
            (25.0 * (1.0 - variation_percent)).max(1.0) as u32,
            (25.0 * (1.0 + variation_percent)) as u32,
        ),
        capacity_nurse_range: (
            (200.0 * (1.0 - variation_percent)).max(50.0) as u32,
            (200.0 * (1.0 + variation_percent)) as u32,
        ),
        depot_return_time_range: (
            1236.0 * (1.0 - variation_percent),
            1236.0 * (1.0 + variation_percent),
        ),
        depot_x_range: (35.0, 45.0),
        depot_y_range: (45.0, 55.0),
        nbr_patients_range: (
            (100.0 * (1.0 - variation_percent)).max(10.0) as usize,
            (100.0 * (1.0 + variation_percent)) as usize,
        ),
        patient_x_range: (0.0, 100.0),
        patient_y_range: (0.0, 100.0),
        patient_demand_range: (10.0, 40.0),
        patient_care_time_range: (85.0, 95.0),
        time_window_start_range: (0.0, 800.0),
        time_window_duration_range: (150.0, 1220.0),
        benchmark: Some(827.3 * (1.0 + variation_percent * 0.5)),
    };
    
    generate_random_instance(base_config)
}

/// Saves an instance to a JSON file with sorted patient keys and without nurses field
pub fn save_instance_to_json(instance: &Instance, output_path: &str) -> std::io::Result<()> {
    use serde_json::{Value, Map};
    
    // First convert to JSON value
    let mut json_value = serde_json::to_value(instance)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    // Remove the nurses field and sort patients
    if let Some(obj) = json_value.as_object_mut() {
        // Remove nurses field
        obj.remove("nurses");
        
        // Sort the patients object by key (numerically)
        if let Some(Value::Object(patients)) = obj.get_mut("patients") {
            let mut sorted_patients = Map::new();
            let mut keys: Vec<_> = patients.keys().cloned().collect();
            
            // Sort keys numerically (not alphabetically)
            keys.sort_by_key(|k| k.parse::<usize>().unwrap_or(0));
            
            for key in keys {
                if let Some(value) = patients.get(&key) {
                    sorted_patients.insert(key, value.clone());
                }
            }
            
            *patients = sorted_patients;
        }
    }
    
    let json = serde_json::to_string_pretty(&json_value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    let mut file = File::create(output_path)?;
    file.write_all(json.as_bytes())?;
    
    println!("Instance saved to: {}", output_path);
    Ok(())
}

/// Generates a random instance and saves it to a JSON file, optionally creating a visualization
pub fn generate_and_save_instance(
    config: InstanceConfig,
    output_json_path: &str,
    visualize: bool,
    output_image_path: Option<&str>,
) -> std::io::Result<Instance> {
    // Generate the instance
    let mut instance = generate_random_instance(config);
    instance.add_nurses();
    
    // Save to JSON
    save_instance_to_json(&instance, output_json_path)?;
    
    // Optionally create a visualization of patient locations
    if visualize {
        let image_path = output_image_path.unwrap_or("output/instance_map.png");
        plot_instance_map(&instance, image_path);
    }
    
    Ok(instance)
}

/// Plots the instance map showing depot and all patient locations
pub fn plot_instance_map(instance: &Instance, output_path: &str) {
    use crate::utils::plot_map::plot_map_with_path;
    
    // Create a dummy solution with all patients in the first nurse's route
    // This ensures all patients are visible on the map
    let mut dummy_solution = vec![Vec::new(); instance.nurses.len()];
    if !dummy_solution.is_empty() {
        // Add all patient IDs to the first route
        dummy_solution[0] = (1..=instance.patients.len()).collect();
    }
    
    plot_map_with_path(
        &dummy_solution,
        &instance.patients,
        &instance.depot,
        output_path,
    );
    
    println!("Instance map saved to: {}", output_path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_instance() {
        let config = InstanceConfig::default();
        let instance = generate_random_instance(config);
        
        assert!(instance.nbr_nurses >= 20 && instance.nbr_nurses <= 30);
        assert!(instance.capacity_nurse >= 150 && instance.capacity_nurse <= 250);
        assert!(!instance.patients.is_empty());
    }

    #[test]
    fn test_travel_times_matrix() {
        let config = InstanceConfig {
            nbr_patients_range: (10, 10),
            ..Default::default()
        };
        let instance = generate_random_instance(config);
        
        // Matrix should be (n+1) x (n+1) where n is number of patients
        assert_eq!(instance.travel_times.len(), 11);
        assert_eq!(instance.travel_times[0].len(), 11);
        
        // Diagonal should be zero
        for i in 0..11 {
            assert_eq!(instance.travel_times[i][i], 0.0);
        }
    }

    #[test]
    fn test_train_like_instance() {
        let instance = generate_train_like_instance(0.2);
        
        // Should be within 20% of the base values
        assert!(instance.nbr_nurses >= 20 && instance.nbr_nurses <= 30);
        assert!(instance.patients.len() >= 80 && instance.patients.len() <= 120);
    }
}
