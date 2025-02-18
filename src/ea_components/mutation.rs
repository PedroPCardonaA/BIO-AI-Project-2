use rand::{seq::SliceRandom, Rng};
use crate::structs::instance::Instance;
use super::fitness::fitness;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Local Search Improvement operator.
/// For each route (nurse), with probability `mutation_rate`, a patient is
/// removed and reinserted in a location that improves the overall fitness.
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
            let patient_index = rng.random_range(0..individual[i].len());
            let patient = individual[i][patient_index];
            let original_fitness = fitness(&individual, instance);

            // Build candidate moves: each candidate is (target_nurse, insertion_index).
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

            if let Some((best_nurse, best_position, _)) = best_candidate {
                individual[i].remove(patient_index);
                individual[best_nurse].insert(best_position, patient);
            }
        }
    }
}

/// Swap Mutation operator.
/// Randomly selects two distinct routes (nurses) and swaps one patient from each.
pub fn mutation_swap(individual: &mut Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();
    let non_empty: Vec<usize> = (0..num_nurses)
        .filter(|&i| !individual[i].is_empty())
        .collect();
    if non_empty.len() < 2 {
        return;
    }
    let nurse_a = non_empty[rng.random_range(0..non_empty.len())];
    let nurse_b = loop {
        let candidate = non_empty[rng.random_range(0..non_empty.len())];
        if candidate != nurse_a {
            break candidate;
        }
    };
    let idx_a = rng.random_range(0..individual[nurse_a].len());
    let idx_b = rng.random_range(0..individual[nurse_b].len());

    // To swap between two different routes safely, use split_at_mut to get mutable references.
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
/// Removes a patient from one route and reinserts it into a random route (or the same route) at a random position.
pub fn mutation_insert(individual: &mut Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();
    let non_empty: Vec<usize> = (0..num_nurses)
        .filter(|&i| !individual[i].is_empty())
        .collect();
    if non_empty.is_empty() {
        return;
    }
    let source_nurse = non_empty[rng.random_range(0..non_empty.len())];
    let patient_index = rng.random_range(0..individual[source_nurse].len());
    let patient = individual[source_nurse][patient_index];
    individual[source_nurse].remove(patient_index);
    let target_nurse = rng.random_range(0..num_nurses);
    let insert_position = rng.random_range(0..=individual[target_nurse].len());
    individual[target_nurse].insert(insert_position, patient);
}

/// Scramble Mutation operator.
/// Randomly shuffles a contiguous subsequence within one route.
pub fn mutation_scramble(individual: &mut Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();
    let candidates: Vec<usize> = (0..num_nurses)
        .filter(|&i| individual[i].len() >= 2)
        .collect();
    if candidates.is_empty() {
        return;
    }
    let nurse = candidates[rng.random_range(0..candidates.len())];
    let route_len = individual[nurse].len();
    let idx1 = rng.random_range(0..route_len);
    let idx2 = rng.random_range(0..route_len);
    let (start, end) = if idx1 < idx2 { (idx1, idx2) } else { (idx2, idx1) };
    individual[nurse][start..=end].shuffle(&mut rng);
}

/// Inversion Mutation operator.
/// Reverses a contiguous subsequence within one route.
pub fn mutation_inversion(individual: &mut Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();
    let candidates: Vec<usize> = (0..num_nurses)
        .filter(|&i| individual[i].len() >= 2)
        .collect();
    if candidates.is_empty() {
        return;
    }
    let nurse = candidates[rng.random_range(0..candidates.len())];
    let route_len = individual[nurse].len();
    let idx1 = rng.random_range(0..route_len);
    let idx2 = rng.random_range(0..route_len);
    let (start, end) = if idx1 < idx2 { (idx1, idx2) } else { (idx2, idx1) };
    individual[nurse][start..=end].reverse();
}

/// Meta mutation operator.
/// With probability `mutation_rate`, randomly chooses one of the mutation operators
/// (local search improvement, swap, insert, scramble, inversion) and applies it.
pub fn meta_mutation(
    individual: &mut Vec<Vec<usize>>,
    mutation_rate: f64,
    instance: &Instance,
) {
    let mut rng = rand::rng();
    if rng.random::<f64>() >= mutation_rate {
        return;
    }
    let op_choice = rng.random::<f64>();
    if op_choice < 0.2 {
        mutation_local_search(individual, mutation_rate, instance);
    } else if op_choice < 0.4 {
        mutation_swap(individual);
    } else if op_choice < 0.6 {
        mutation_insert(individual);
    } else if op_choice < 0.8 {
        mutation_scramble(individual);
    } else {
        mutation_inversion(individual);
    }
}
