use rand::{seq::SliceRandom, Rng};

use crate::structs::instance::Instance;

use rayon::prelude::*;
use std::sync::Arc;

use super::fitness::fitness;

/// Performs the select-delete-fix crossover on two parent solutions.
/// 
/// This function randomly selects one route from each parent, then removes from each child's routes
/// any patients found in the selected route of the opposite parent. The removed patients are then
/// reinserted into the child's routes at positions that yield the best fitness. If the generated
/// random value exceeds `crossover_rate`, the function returns the parents unchanged.
/// 
/// # Parameters
/// - `parent1`: The first parent's solution represented as a vector of routes.
/// - `parent2`: The second parent's solution represented as a vector of routes.
/// - `instance`: A reference to the problem instance.
/// - `crossover_rate`: The probability threshold for applying the crossover.
/// 
/// # Returns
/// A tuple containing two offspring solutions, each represented as a vector of routes.
pub fn select_delete_fix_crossover(
    parent1: &Vec<Vec<usize>>,
    parent2: &Vec<Vec<usize>>,
    instance: &Instance,
    crossover_rate: f64,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {

    // STEP 1: If the crossover rate is not met, return clones of the parents.
    let mut rng = rand::rng();
    if rng.random::<f64>() > crossover_rate {
        return (parent1.clone(), parent2.clone());
    }

    // STEP 2: Randomly select one nurse route index from each parent.
    let nurse_count = parent1.len();
    let route_idx1 = rng.random_range(0..nurse_count);
    let route_idx2 = rng.random_range(0..nurse_count);
    let selected_route_parent1 = parent1[route_idx1].clone();
    let selected_route_parent2 = parent2[route_idx2].clone();

    // STEP 3: Create children as clones of the parents.
    let mut child1 = parent1.clone();
    let mut child2 = parent2.clone();

    // STEP 4: In child1, remove any patient that appears in parent2's selected route and store them.
    let mut missing_from_child1 = Vec::new();
    for route in child1.iter_mut() {
        let mut i = 0;
        while i < route.len() {
            if selected_route_parent2.contains(&route[i]) {
                missing_from_child1.push(route.remove(i));
            } else {
                i += 1;
            }
        }
    }

    // STEP 5: In child2, remove any patient that appears in parent1's selected route and store them.
    let mut missing_from_child2 = Vec::new();
    for route in child2.iter_mut() {
        let mut i = 0;
        while i < route.len() {
            if selected_route_parent1.contains(&route[i]) {
                missing_from_child2.push(route.remove(i));
            } else {
                i += 1;
            }
        }
    }

    // Prepare the instance for parallel evaluations.
    let instance_arc = Arc::new(instance.clone());

    // STEP 6  - Fix child1: For each missing patient in child1, reinsert at the best position (lowest fitness) using parallel evaluation.
    for patient in missing_from_child1 {
        // Build candidate insertion positions as (route_index, insertion_position).
        let candidate_positions: Vec<(usize, usize)> = (0..child1.len())
            .flat_map(|r_idx| {
                // Allow insertion at any position in route, including at the beginning or end.
                (0..=child1[r_idx].len()).map(move |pos| (r_idx, pos))
            })
            .collect();

        let best_insertion = candidate_positions
            .par_iter()
            .map(|&(r_idx, pos)| {
                let mut candidate = child1.clone();
                candidate[r_idx].insert(pos, patient);
                let candidate_fit = fitness(&candidate, &instance_arc);
                ((r_idx, pos), candidate_fit)
            })
            .min_by(|(_, fit_a), (_, fit_b)| fit_a.partial_cmp(fit_b).unwrap())
            .map(|(insertion, _)| insertion);

        if let Some((best_route_idx, best_position)) = best_insertion {
            child1[best_route_idx].insert(best_position, patient);
        }
    }

    // STEP 7 - Fix child2: For each missing patient in child2, reinsert at the best position using parallel evaluation.
    for patient in missing_from_child2 {
        let candidate_positions: Vec<(usize, usize)> = (0..child2.len())
            .flat_map(|r_idx| {
                (0..=child2[r_idx].len()).map(move |pos| (r_idx, pos))
            })
            .collect();

        let best_insertion = candidate_positions
            .par_iter()
            .map(|&(r_idx, pos)| {
                let mut candidate = child2.clone();
                candidate[r_idx].insert(pos, patient);
                let candidate_fit = fitness(&candidate, &instance_arc);
                ((r_idx, pos), candidate_fit)
            })
            .min_by(|(_, fit_a), (_, fit_b)| fit_a.partial_cmp(fit_b).unwrap())
            .map(|(insertion, _)| insertion);

        if let Some((best_route_idx, best_position)) = best_insertion {
            child2[best_route_idx].insert(best_position, patient);
        }
    }

    (child1, child2)
}

/// Performs a random-route-mixing crossover between two parent solutions.
/// 
/// With a probability defined by `crossover_rate`, this function generates an offspring by:
/// - Flattening the routes of one parent into a single list of patients.
/// - Shuffling the patient list to disrupt the original route structure.
/// - Partitioning the shuffled list into new routes based on the average route sizes derived from both parents.
/// 
/// If the random chance does not meet the crossover rate, the function returns a clone of `parent1`.
/// 
/// # Parameters
/// - `parent1`: The first parent's solution represented as a vector of routes (each route is a vector of patient IDs).
/// - `parent2`: The second parent's solution represented as a vector of routes.
/// - `crossover_rate`: The probability threshold for performing the crossover.
/// 
/// # Returns
/// A new solution represented as a vector of routes, ensuring that all patients appear exactly once.
pub fn random_route_mixing_crossover(
    parent1: &Vec<Vec<usize>>, 
    parent2: &Vec<Vec<usize>>,
    crossover_rate: f64,
) -> Vec<Vec<usize>> {
    let mut rng = rand::rng();

    // STEP 1: Decide whether to perform crossover or return a clone of parent1.
    if rng.random::<f64>() > crossover_rate {
        return parent1.clone();
    }

    // STEP 2: Flatten parent's routes into a list of all patients and shuffle it.
    let mut all_patients: Vec<usize> = parent1.iter().flatten().cloned().collect();
    all_patients.shuffle(&mut rng);

    // STEP 3: Compute new route sizes for the offspring.
    //   3a: Determine the number of nurses (routes) and the total number of patients.
    let nurse_count = parent1.len();
    let total_patients = all_patients.len();
    //   3b: Compute the average route size for each nurse from both parents.
    let mut avg_sizes = Vec::with_capacity(nurse_count);
    for i in 0..nurse_count {
        let size = (parent1[i].len() + parent2[i].len()) / 2;
        avg_sizes.push(size);
    }

    //   3c: Calculate the sum of average sizes, compute proportions, and determine the number
    //       of patients for each nurse in the offspring.
    let sum_sizes: usize = avg_sizes.iter().sum();
    // Compute proportions for each nurse.
    let proportions: Vec<f64> = avg_sizes.iter().map(|&s| s as f64 / sum_sizes as f64).collect();
    // Determine the number of patients for each nurse in the offspring.
    let mut route_sizes: Vec<usize> = proportions
        .iter()
        .map(|&p| (p * total_patients as f64).round() as usize)
        .collect();
    // Adjust in case of rounding errors so that the total equals total_patients.
    let diff = total_patients as isize - route_sizes.iter().sum::<usize>() as isize;
    if diff != 0 {
        // Adjust the first route arbitrarily.
        route_sizes[0] = (route_sizes[0] as isize + diff) as usize;
    }

    // STEP 4: Partition the shuffled patient list into new routes according to the computed route sizes.
    let mut new_solution = Vec::with_capacity(nurse_count);
    let mut index = 0;
    for &size in &route_sizes {
        if index + size <= total_patients {
            new_solution.push(all_patients[index..index + size].to_vec());
            index += size;
        } else {
            new_solution.push(all_patients[index..].to_vec());
            index = total_patients;
        }
    }
    
    // STEP 5: Ensure the new solution has exactly `nurse_count` routes by adding empty routes if needed.
    while new_solution.len() < nurse_count {
        new_solution.push(Vec::new());
    }

    new_solution
}

/// Performs a meta crossover operation between two parent solutions.
/// 
/// This function chooses between two crossover strategies based on a probability check:
/// - If a generated random value is below 0.7, it uses the select-delete-fix crossover method.
/// - Otherwise, it uses the random-route-mixing crossover method and duplicates the resulting offspring.
/// 
/// # Parameters
/// - `parent1`: The first parent's solution represented as a vector of routes.
/// - `parent2`: The second parent's solution represented as a vector of routes.
/// - `instance`: A reference to the problem instance providing necessary context for the crossover.
/// - `crossover_rate`: A parameter influencing the crossover probability for the chosen method.
/// 
/// # Returns
/// A tuple containing two offspring solutions, each represented as a vector of routes.
pub fn meta_crossover(
    parent1: &Vec<Vec<usize>>,
    parent2: &Vec<Vec<usize>>,
    instance: &Instance,
    crossover_rate: f64,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    
    // STEP 1: Determine which alternative crossover strategy to use based on a random threshold.
    if rng.random::<f64>() < 0.7 {
        // ALT 1: Use the select-delete-fix crossover strategy.
        select_delete_fix_crossover(parent1, parent2, instance, crossover_rate)
    } else {
        // ALT 2: Use the random-route-mixing crossover strategy.
        let offspring = random_route_mixing_crossover(parent1, parent2, crossover_rate);
        // Duplicate the offspring to form a pair.
        (offspring.clone(), offspring)
    }
}
