use rand::{seq::SliceRandom, Rng};
use crate::structs::instance::Instance;
use super::fitness::fitness;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Local Search Improvement Mutation operator.
/// 
/// This operator iterates over each nurse's route within the solution and, with probability `mutation_rate`,
/// removes a randomly selected patient from the route and reinserts that patient at the position that minimizes
/// the overall solution fitness (cost). Candidate insertion positions are evaluated in parallel to determine the best move.
/// 
/// # Parameters
/// - `individual` - A mutable reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
/// - `mutation_rate` - The probability threshold for applying the mutation to a given route.
/// - `instance` - A reference to the problem instance containing travel times, patient data, and depot information.
pub fn mutation_local_search(
    individual: &mut Vec<Vec<usize>>,
    mutation_rate: f64,
    instance: &Instance,
) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();
    if num_nurses < 2 {
        return;
    }
    for i in 0..num_nurses {
        if !individual[i].is_empty() && rng.random::<f64>() < mutation_rate {
            // STEP 1: Select a random patient from the current nurse's route.
            let patient_index = rng.random_range(0..individual[i].len());
            let patient = individual[i][patient_index];
            let original_fitness = fitness(&individual, instance);

            // STEP 2: Build candidate moves: each candidate is (target_nurse, insertion_index).
            let candidate_moves: Vec<(usize, usize)> = (0..num_nurses)
                .flat_map(|j| {
                    let range = if i == j {
                        0..=individual[j].len().saturating_sub(1)
                    } else {
                        0..=individual[j].len()
                    };
                    range.map(move |k| (j, k))
                })
                .filter(|&(j, k)| !(i == j && k == patient_index))
                .collect();

            // STEP 3: Evaluate candidate moves in parallel to find the best relocation that improves fitness.
            let best_candidate = candidate_moves
                .par_iter()
                .filter_map(|&(j, k)| {
                    let mut new_individual = individual.clone();
                    new_individual[i].remove(patient_index);
                    new_individual[j].insert(k, patient);
                    let candidate_fitness = fitness(&new_individual, instance);
                    if candidate_fitness < original_fitness {
                        Some((j, k, candidate_fitness))
                    } else {
                        None
                    }
                })
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

            // STEP 4: If an improvement is found, update the individual with the best relocation.
            if let Some((best_nurse, best_position, _)) = best_candidate {
                individual[i].remove(patient_index);
                individual[best_nurse].insert(best_position, patient);
            }
        }
    }
}

/// Swap Mutation operator.
/// 
/// This operator randomly selects two distinct nurse routes from the solution and swaps one patient from each route.
/// The exchange introduces variation into the solution by altering the assignment of patients between routes.
/// 
/// # Parameters
/// - `individual` - A mutable reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
pub fn mutation_swap(individual: &mut Vec<Vec<usize>>) {

    // STEP 1: Determine the number of nurse routes.
    let mut rng = rand::rng();
    let num_nurses = individual.len();
    
    // STEP 2: Identify non-empty routes to ensure there are candidates for swapping.
    let non_empty: Vec<usize> = (0..num_nurses)
        .filter(|&i| !individual[i].is_empty())
        .collect();
    if non_empty.len() < 2 {
        return;
    }
    
    // STEP 3: Randomly select the first nurse (nurse_a) from non-empty routes.
    let nurse_a = non_empty[rng.random_range(0..non_empty.len())];
    
    // STEP 4: Randomly select the second nurse (nurse_b) ensuring it is different from nurse_a.
    let nurse_b = loop {
        let candidate = non_empty[rng.random_range(0..non_empty.len())];
        if candidate != nurse_a {
            break candidate;
        }
    };
    
    // STEP 5: Randomly select a patient index from each selected nurse's route.
    let idx_a = rng.random_range(0..individual[nurse_a].len());
    let idx_b = rng.random_range(0..individual[nurse_b].len());

    // STEP 6: Swap the selected patients between the two routes using safe mutable references.
    //         The split_at_mut method is used to obtain mutable slices without aliasing.
    if nurse_a < nurse_b {
        let (first, second) = individual.split_at_mut(nurse_b);
        let vec_a = &mut first[nurse_a];
        let vec_b = &mut second[0]; // nurse_b is the first element in the second slice.
        std::mem::swap(&mut vec_a[idx_a], &mut vec_b[idx_b]);
    } else {
        let (first, second) = individual.split_at_mut(nurse_a);
        let vec_b = &mut first[nurse_b];
        let vec_a = &mut second[0]; // nurse_a is the first element in the second slice.
        std::mem::swap(&mut vec_a[idx_a], &mut vec_b[idx_b]);
    }
}

/// Insert Mutation operator.
/// 
/// This operator removes a patient from one nurse route and reinserts it into a randomly chosen route (which may be the same route)
/// at a random position. This mutation alters the configuration of the routes while preserving the overall set of patients.
/// 
/// # Parameters
/// - `individual` - A mutable reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
pub fn mutation_insert(individual: &mut Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();

    // STEP 1: Identify non-empty routes and randomly select a source nurse and a patient from that route.
    let non_empty: Vec<usize> = (0..num_nurses)
        .filter(|&i| !individual[i].is_empty())
        .collect();
    if non_empty.is_empty() {
        return;
    }
    let source_nurse = non_empty[rng.random_range(0..non_empty.len())];
    let patient_index = rng.random_range(0..individual[source_nurse].len());
    let patient = individual[source_nurse][patient_index];
    
    // STEP 2: Remove the selected patient from the source route.
    individual[source_nurse].remove(patient_index);
    
    // STEP 3: Randomly select a target nurse and insertion position, then insert the patient.
    let target_nurse = rng.random_range(0..num_nurses);
    let insert_position = rng.random_range(0..=individual[target_nurse].len());
    individual[target_nurse].insert(insert_position, patient);
}

/// Scramble Mutation operator.
/// 
/// This operator selects a contiguous subsequence within a single nurse route and randomly shuffles the order of patients within that subsequence.
/// The scramble operation disrupts the existing ordering, thereby promoting exploration of alternative route configurations.
/// 
/// # Parameters
/// - `individual` - A mutable reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
pub fn mutation_scramble(individual: &mut Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();

    // STEP 1: Identify candidate routes with at least 2 patients.
    let candidates: Vec<usize> = (0..num_nurses)
        .filter(|&i| individual[i].len() >= 2)
        .collect();
    if candidates.is_empty() {
        return;
    }
    // STEP 2: Randomly select one candidate nurse route.
    let nurse = candidates[rng.random_range(0..candidates.len())];
    let route_len = individual[nurse].len();

    // STEP 3: Randomly choose two indices to define the subsequence boundaries.
    let idx1 = rng.random_range(0..route_len);
    let idx2 = rng.random_range(0..route_len);
    let (start, end) = if idx1 < idx2 { (idx1, idx2) } else { (idx2, idx1) };

    // STEP 4: Shuffle the selected subsequence within the nurse's route.
    individual[nurse][start..=end].shuffle(&mut rng);
}

/// Inversion Mutation operator.
/// 
/// This operator reverses a contiguous subsequence within a single nurse route. By inverting the order of a segment,
/// the operator explores alternative sequences that may lead to improved overall route performance.
/// 
/// # Parameters
/// - `individual` - A mutable reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
pub fn mutation_inversion(individual: &mut Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();

    // STEP 1: Identify candidate routes that have at least 2 patients.
    let candidates: Vec<usize> = (0..num_nurses)
        .filter(|&i| individual[i].len() >= 2)
        .collect();
    if candidates.is_empty() {
        return;
    }
    // STEP 2: Randomly select one candidate nurse route.
    let nurse = candidates[rng.random_range(0..candidates.len())];
    let route_len = individual[nurse].len();

    // STEP 3: Randomly choose two indices to define the subsequence boundaries.
    let idx1 = rng.random_range(0..route_len);
    let idx2 = rng.random_range(0..route_len);
    let (start, end) = if idx1 < idx2 { (idx1, idx2) } else { (idx2, idx1) };

    // STEP 4: Reverse the selected subsequence within the chosen nurse's route.
    individual[nurse][start..=end].reverse();
}

/// Meta Mutation operator.
/// 
/// With probability `mutation_rate`, this operator randomly selects one of the available mutation operators:
/// local search improvement,
///  swap,
///  insert,
///  scramble,
///  or inversion
/// and applies it to the solution. 
/// This combined approach enhances the exploration of the solution space by leveraging the strengths of multiple mutation strategies.
/// 
/// # Parameters
/// - `individual` - A mutable reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
/// - `mutation_rate` - The probability threshold for applying a mutation to the solution.
/// - `instance` - A reference to the problem instance containing travel times, patient data, and depot information.
pub fn meta_mutation(
    individual: &mut Vec<Vec<usize>>,
    mutation_rate: f64,
    instance: &Instance,
) {
    let mut rng = rand::rng();

    // STEP 1: Check if mutation should be applied based on the mutation_rate.
    if rng.random::<f64>() >= mutation_rate {
        return;
    }

    // STEP 2: Generate a random value to select the mutation operator.
    let op_choice = rng.random::<f64>();
    
    // STEP 3: Apply the chosen mutation operator based on the random selection.
    if op_choice < 0.2 {
        mutation_local_search(individual, mutation_rate, instance);
    }
    mutation_swap(individual);
    mutation_insert(individual);
    mutation_scramble(individual);
    mutation_inversion(individual);
}

