use rand::{seq::SliceRandom, Rng};

use crate::structs::instance::Instance;

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

pub fn generate_population_combined(
    population_size: usize,
    instance: &Instance,
) -> Vec<Vec<Vec<usize>>> {
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();

    // Determine the sizes for the random and heuristic portions.
    let random_population_size = ((population_size as f64) * 0.8).round() as usize;
    let heuristic_population_size = population_size - random_population_size;

    let mut population = Vec::with_capacity(population_size);

    // Generate random population.
    {
        let mut rng = rand::thread_rng();
        for _ in 0..random_population_size {
            // Create a shuffled list of patient IDs (assumed to be 1-based).
            let mut patients: Vec<usize> = (1..=patient_count).collect();
            patients.shuffle(&mut rng);

            let mut solution = vec![Vec::new(); nurse_count];

            // Randomly distribute patients to nurses.
            for patient in patients {
                let nurse_index = rng.gen_range(0..nurse_count);
                solution[nurse_index].push(patient);
            }
            population.push(solution);
        }
    }

    // Generate heuristic population.
    {
        let mut rng = rand::thread_rng();
        for _ in 0..heuristic_population_size {
            // Create a shuffled list of patient IDs.
            let mut patient_ids: Vec<usize> = (1..=patient_count).collect();
            patient_ids.shuffle(&mut rng);

            // Clone the nurse list to track current loads.
            let mut nurses = instance.nurses.clone();
            // Each solution is represented as a vector of routes (one per nurse).
            let mut solution = vec![Vec::new(); nurse_count];

            // First, assign one patient to each nurse (if available and within capacity).
            for i in 0..nurse_count {
                if let Some(patient) = patient_ids.pop() {
                    let current_load = nurses[i].get_current_load();
                    let capacity = nurses[i].get_capacity();
                    if current_load < capacity {
                        solution[i].push(patient);
                        nurses[i].set_current_load(current_load + 1);
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

                    // Compute the extra travel time cost for appending the patient.
                    let increase = if route.is_empty() {
                        // For an empty route: depot -> patient + patient -> depot.
                        instance.travel_times[0][patient] + instance.travel_times[patient][0]
                    } else {
                        // For a non-empty route: cost for (last patient -> new patient + new patient -> depot)
                        // minus the depot return cost of the last patient.
                        let last_patient = *route.last().unwrap();
                        instance.travel_times[last_patient][patient]
                            + instance.travel_times[patient][0]
                            - instance.travel_times[last_patient][0]
                    };

                    // Add current load as a penalty.
                    let balanced_increase = increase + (current_load as f64);

                    if balanced_increase < best_balanced_increase {
                        best_balanced_increase = balanced_increase;
                        best_nurse_index = Some(i);
                    }
                }

                if let Some(i) = best_nurse_index {
                    solution[i].push(patient);
                    let current_load = nurses[i].get_current_load();
                    nurses[i].set_current_load(current_load + 1);
                } else {
                    // If no nurse has available capacity, assign the patient to the nurse with the smallest overload.
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
                    let current_load = nurses[min_overload_index].get_current_load();
                    nurses[min_overload_index].set_current_load(current_load + 1);
                }
            }
            population.push(solution);
        }
    }

    // Optionally, shuffle the entire population.
    {
        let mut rng = rand::thread_rng();
        population.shuffle(&mut rng);
    }

    population
}
