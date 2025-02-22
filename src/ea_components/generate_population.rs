use rand::{seq::SliceRandom, Rng};
use crate::structs::instance::Instance;

/// Generates a population of solutions.
/// 
/// Each solution is a vector of routes, where each route is a vector of patient IDs.
/// 
/// # Arguments
/// 
/// * `population_size` - The number of solutions to generate.
/// * `instance` - A reference to the problem instance.
/// 
/// # Returns
/// 
/// A vector of solutions.
pub fn generate_population(population_size: usize, instance: &Instance) -> Vec<Vec<Vec<usize>>> {
    let mut population = Vec::new();
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();
    let mut rng = rand::rng();
    
    for _ in 0..population_size {
        let mut patients: Vec<usize> = (1..=patient_count).collect();
        patients.shuffle(&mut rng);

        let mut solution = vec![Vec::new(); nurse_count];
        for patient in patients {
            let nurse_index = rng.random_range(0..nurse_count);
            solution[nurse_index].push(patient);
        }

        population.push(solution);
    }
    
    population
}

/// Generates a population of solutions using a heuristic that takes into account the workload of each nurse.
/// 
/// Each solution is a vector of routes, where each route is a vector of patient IDs.
/// 
/// # Arguments
/// 
/// * `population_size` - The number of solutions to generate.
/// * `instance` - A reference to the problem instance.
/// 
/// # Returns
/// 
/// A vector of solutions.
pub fn generate_population_heuristic_with_workload(
    population_size: usize,
    instance: &Instance,
) -> Vec<Vec<Vec<usize>>> {
    let mut population = Vec::with_capacity(population_size);
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();
    let mut rng = rand::rng();

    for _ in 0..population_size {
        let mut patient_ids: Vec<usize> = (1..=patient_count).collect();
        patient_ids.shuffle(&mut rng);
        let mut nurses = instance.nurses.clone();
        let mut solution = vec![Vec::new(); nurse_count];
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
        while let Some(patient) = patient_ids.pop() {
            let mut best_nurse_index = None;
            let mut best_balanced_increase = f64::MAX;
            for (i, route) in solution.iter().enumerate() {
                let current_load = nurses[i].get_current_load();
                let capacity = nurses[i].get_capacity();
                if current_load >= capacity {
                    continue;
                }
                let increase = if route.is_empty() {
                    instance.travel_times[0][patient] + instance.travel_times[patient][0]
                } else {
                    let last_patient = *route.last().unwrap();
                    instance.travel_times[last_patient][patient]
                        + instance.travel_times[patient][0]
                        - instance.travel_times[last_patient][0]
                };
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

    population
}


/// Generates a population of solutions using a heuristic that takes into account the workload of each nurse.
/// 
/// Each solution is a vector of routes, where each route is a vector of patient IDs.
/// 
/// # Arguments
/// 
/// * `population_size` - The number of solutions to generate.
/// * `instance` - A reference to the problem instance.
/// 
/// # Returns
/// 
/// A vector of solutions.
pub fn generate_population_combined(
    population_size: usize,
    instance: &Instance,
) -> Vec<Vec<Vec<usize>>> {
    let random_population_size = ((population_size as f64) * 0.5).round() as usize;
    let heuristic_population_size = population_size - random_population_size;

    let mut population = generate_population(random_population_size, instance);
    let mut heuristic_population =
        generate_population_heuristic_with_workload(heuristic_population_size, instance);
    population.append(&mut heuristic_population);
    let mut rng = rand::rng();
    population.shuffle(&mut rng);

    population
}
