use rand::{seq::IndexedRandom, Rng};

use crate::structs::instance::Instance;
use super::fitness::fitness;


pub fn mutate_relocate_patient(
    individual: &mut Vec<Vec<usize>>,
    mutation_probability: f64,
) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();
    if num_nurses < 2 {
        return;
    }
    for i in 0..num_nurses {
        if !individual[i].is_empty() && rng.random::<f64>() < mutation_probability {
            let patient_index = rng.random_range(0..individual[i].len());
            let patient = individual[i].remove(patient_index);
            let other_nurses: Vec<usize> = (0..num_nurses).filter(|&j| j != i).collect();
            let target_nurse = *other_nurses.choose(&mut rng).unwrap();
            let insertion_index = rng.random_range(0..=individual[target_nurse].len());
            individual[target_nurse].insert(insertion_index, patient);
        }
    }
}



pub fn swap_mutation(individual: &mut Vec<Vec<usize>>, mutation_rate: f64) {
    let mut rng = rand::rng();
    let mut positions = Vec::new();

    if rng.random::<f64>() > mutation_rate {
        return;
    }

    // Collect the positions of all patients across routes.
    for (route_idx, route) in individual.iter().enumerate() {
        for pos_idx in 0..route.len() {
            positions.push((route_idx, pos_idx));
        }
    }

    // Need at least two patients to swap.
    if positions.len() < 2 {
        return;
    }

    // Select two distinct random positions.
    let idx1 = rng.random_range(0..positions.len());
    let mut idx2 = rng.random_range(0..positions.len());
    while idx2 == idx1 {
        idx2 = rng.random_range(0..positions.len());
    }

    let (route1, pos1) = positions[idx1];
    let (route2, pos2) = positions[idx2];

    // Swap the two patients.
    if route1 == route2 {
        // If in the same route, use the built-in swap.
        individual[route1].swap(pos1, pos2);
    } else {
        // If in different routes, swap using mem::swap.
        if route1 == route2 {
            individual[route1].swap(pos1, pos2);
        } else {
            let (left, right) = individual.split_at_mut(std::cmp::max(route1, route2));
            if route1 < route2 {
                std::mem::swap(&mut left[route1][pos1], &mut right[0][pos2]);
            } else {
                std::mem::swap(&mut right[0][pos1], &mut left[route2][pos2]);
            }
        }
    }
}

use rayon::prelude::*;
use std::cmp::Ordering;

pub fn mutate_local_improvement(
    individual: &mut Vec<Vec<usize>>,
    mutation_rate: f64,
    instance: &Instance,
) {
    // Use thread_rng instead of rand::rng() if possible.
    let mut rng = rand::rng();

    // With probability 1 - mutation_rate, do nothing.
    if rng.random::<f64>() > mutation_rate {
        return;
    }

    let num_nurses = individual.len();
    if num_nurses < 2 {
        return;
    }

    // Iterate over each nurse's route.
    for i in 0..num_nurses {
        if !individual[i].is_empty() && rng.random::<f64>() < mutation_rate {
            // Randomly select a patient from nurse i's route.
            let patient_index = rng.random_range(0..individual[i].len());
            let patient = individual[i][patient_index];
            let original_fitness = fitness(&individual, instance);

            // Build a list of candidate moves as (target_nurse, insertion_index).
            let candidate_moves: Vec<(usize, usize)> = (0..num_nurses)
                .flat_map(|j| {
                    // If moving within the same nurse, removal decreases the route length by 1.
                    let range = if i == j {
                        0..=individual[j].len().saturating_sub(1)
                    } else {
                        0..=individual[j].len()
                    };
                    range.map(move |k| (j, k))
                })
                // Skip the case where the patient would be inserted in the same location.
                .filter(|&(j, k)| !(i == j && k == patient_index))
                .collect();

            // Evaluate candidate moves in parallel.
            let best_candidate = candidate_moves
                .par_iter()
                .filter_map(|&(j, k)| {
                    let mut new_individual = individual.clone();
                    // Remove the patient from the original route.
                    new_individual[i].remove(patient_index);
                    // Insert the patient in the candidate route at position k.
                    new_individual[j].insert(k, patient);
                    let candidate_fitness = fitness(&new_individual, instance);
                    if candidate_fitness < original_fitness {
                        Some((j, k, candidate_fitness))
                    } else {
                        None
                    }
                })
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

            // If a better candidate move was found, update the individual.
            if let Some((best_nurse, best_position, _)) = best_candidate {
                individual[i].remove(patient_index);
                individual[best_nurse].insert(best_position, patient);
            }
        }
    }
}
