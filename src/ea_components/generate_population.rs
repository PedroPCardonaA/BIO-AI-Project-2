// Not used, better to use the heuristic approach
pub fn generate_population(population_size: usize, instance: &Instance) -> Vec<Vec<Vec<usize>>> {
    let mut population = Vec::new();
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();
    let mut rng = rand::rng();
    
    for _ in 0..population_size {
        let mut patients: Vec<usize> = (1..(patient_count + 1)).collect();
        patients.shuffle(&mut rng);

        let mut solution = vec![Vec::new(); nurse_count];

        // Randomly distribute patients to nurses
        for &patient in &patients {
            let nurse_index = rng.random_range(0..nurse_count);
            solution[nurse_index].push(patient);
        }

        population.push(solution);
    }
    
    population
}

use std::f64;

use rand::{seq::SliceRandom, Rng};

use crate::structs::instance::Instance;

// Assume Instance, Patient, Depot, and Nurse are defined as in your project.

pub fn generate_population_heuristic(population_size: usize, instance: &Instance) -> Vec<Vec<Vec<usize>>> {
    let mut population = Vec::with_capacity(population_size);
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();
    let mut rng = rand::rng();
    
    // Parameter to penalize nurses that already have many patients.
    let load_penalty: f64 = 1.0; // Tune this value as needed.
    
    for _ in 0..population_size {
        // Create a shuffled list of patient IDs.
        let mut patient_ids: Vec<usize> = (1..=patient_count).collect();
        patient_ids.shuffle(&mut rng);
        
        // Each solution is a vector of routes (each route is a vector of patient IDs)
        // and each nurse's route starts and ends at the depot (index 0).
        let mut solution = vec![Vec::new(); nurse_count];
        
        // First, ensure that every nurse gets one patient if possible.
        for i in 0..nurse_count {
            if let Some(patient) = patient_ids.pop() {
                solution[i].push(patient);
            }
        }
        
        // For the remaining patients, assign each to the nurse that minimizes the balanced cost.
        while let Some(patient) = patient_ids.pop() {
            let mut best_nurse_index = 0;
            let mut best_balanced_increase = f64::MAX;
            
            for (i, route) in solution.iter().enumerate() {
                // Calculate the extra travel time of appending this patient.
                // If the route is empty (should not occur now because of the initial assignment),
                // use depot -> patient + patient -> depot.
                let increase = if route.is_empty() {
                    instance.travel_times[0][patient] + instance.travel_times[patient][0]
                } else {
                    // For a non-empty route, the additional cost is:
                    // travel time from the last patient in the route to the new patient,
                    // plus travel time from the new patient back to the depot,
                    // minus the current travel time from the last patient to the depot.
                    let last_patient = *route.last().unwrap();
                    instance.travel_times[last_patient][patient] 
                        + instance.travel_times[patient][0] 
                        - instance.travel_times[last_patient][0]
                };
                
                // Add a penalty proportional to the current number of patients in the nurse's route.
                let balanced_increase = increase + load_penalty * (route.len() as f64);
                
                if balanced_increase < best_balanced_increase {
                    best_balanced_increase = balanced_increase;
                    best_nurse_index = i;
                }
            }
            // Assign the patient to the nurse with the minimal balanced cost.
            solution[best_nurse_index].push(patient);
        }
        
        population.push(solution);
    }
    
    population
}

pub fn generate_population_heuristic_with_workload(
    population_size: usize,
    instance: &Instance,
) -> Vec<Vec<Vec<usize>>> {
    let mut population = Vec::with_capacity(population_size);
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();
    let mut rng = rand::rng();

    for _ in 0..population_size {
        // Create a shuffled list of patient IDs (assumed to be 1-based).
        let mut patient_ids: Vec<usize> = (1..=patient_count).collect();
        patient_ids.shuffle(&mut rng);

        // Clone the nurse list to track each nurse's current load.
        let mut nurses = instance.nurses.clone();
        // Each solution is represented as a vector of routes (each route is a Vec of patient IDs).
        let mut solution = vec![Vec::new(); nurse_count];

        // First, assign one patient to each nurse (if available and if capacity allows).
        for i in 0..nurse_count {
            if let Some(patient) = patient_ids.pop() {
                {
                    let current_load = nurses[i].get_current_load();
                    let capacity = nurses[i].get_capacity();
                    if current_load < capacity {
                        solution[i].push(patient);
                        let new_load = current_load + 1;
                        nurses[i].set_current_load(new_load);
                    }
                }
            }
        }

        // For each remaining patient, choose the nurse that minimizes the balanced increase.
        while let Some(patient) = patient_ids.pop() {
            let mut best_nurse_index = None;
            let mut best_balanced_increase = f64::MAX;

            // Consider only nurses with available capacity.
            for (i, route) in solution.iter().enumerate() {
                let current_load = nurses[i].get_current_load();
                let capacity = nurses[i].get_capacity();
                if current_load >= capacity {
                    continue;
                }

                // Compute the extra travel time cost of appending the patient.
                let increase = if route.is_empty() {
                    // For an empty route, cost = depot -> patient + patient -> depot.
                    instance.travel_times[0][patient] + instance.travel_times[patient][0]
                } else {
                    // For a non-empty route, cost = (last patient -> new patient + new patient -> depot)
                    // minus (last patient -> depot) already accounted for.
                    let last_patient = *route.last().unwrap();
                    instance.travel_times[last_patient][patient]
                        + instance.travel_times[patient][0]
                        - instance.travel_times[last_patient][0]
                };

                // Instead of a fixed penalty, add the current load as a cost.
                let balanced_increase = increase + (current_load as f64);

                if balanced_increase < best_balanced_increase {
                    best_balanced_increase = balanced_increase;
                    best_nurse_index = Some(i);
                }
            }

            if let Some(i) = best_nurse_index {
                // Assign the patient to the chosen nurse.
                solution[i].push(patient);
                let new_load = nurses[i].get_current_load() + 1;
                nurses[i].set_current_load(new_load);
            } else {
                // If no nurse has available capacity (all reached capacity),
                // assign the patient to the nurse with the smallest overload.
                let mut min_overload_index = 0;
                let mut min_overload = f64::MAX;
                for (i, nurse) in nurses.iter().enumerate() {
                    let overload = nurse.get_current_load() as f64 - nurse.get_capacity() as f64;
                    if overload < min_overload {
                        min_overload = overload;
                        min_overload_index = i;
                    }
                }
                solution[min_overload_index].push(patient);
                let new_load = nurses[min_overload_index].get_current_load() + 1;
                nurses[min_overload_index].set_current_load(new_load);
            }
        }

        population.push(solution);
    }

    population
}
