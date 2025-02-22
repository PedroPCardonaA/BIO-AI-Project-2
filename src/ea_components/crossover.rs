use std::{cmp::Ordering, collections::{HashMap, HashSet}};

use rand::{seq::{IndexedRandom, IteratorRandom, SliceRandom}, Rng};

use crate::structs::instance::Instance;


use rayon::prelude::*;
use std::sync::Arc;

use super::fitness::fitness;

pub fn edge_crossover(parent1: &Vec<Vec<usize>>, parent2: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut rng = rand::rng();
    let nurse_count = parent1.len();
    
    // Flatten parents into ordered lists of patients, while keeping nurse structure
    let p1_flat: Vec<usize> = parent1.iter().flatten().cloned().collect();
    let p2_flat: Vec<usize> = parent2.iter().flatten().cloned().collect();
    let patient_count = p1_flat.len();

    // Build adjacency edge map
    let mut edge_map: HashMap<usize, HashSet<usize>> = HashMap::new();
    
    for (p1, p2) in [(&p1_flat, &p2_flat), (&p2_flat, &p1_flat)].iter() {
        for i in 0..patient_count {
            let current = p1[i];
            let left = if i == 0 { p1[patient_count - 1] } else { p1[i - 1] };
            let right = if i == patient_count - 1 { p1[0] } else { p1[i + 1] };
            
            edge_map.entry(current).or_insert_with(HashSet::new).insert(left);
            edge_map.entry(current).or_insert_with(HashSet::new).insert(right);
        }
    }

    // Generate offspring as a valid permutation
    let mut offspring = Vec::new();
    let mut remaining: HashSet<usize> = p1_flat.iter().cloned().collect();
    
    let mut current = *p1_flat.choose(&mut rng).unwrap();
    offspring.push(current);
    remaining.remove(&current);

    while !remaining.is_empty() {
        // Remove current patient from all adjacency lists
        for neighbors in edge_map.values_mut() {
            neighbors.remove(&current);
        }

        // Choose the next patient
        let next = if let Some(neighbors) = edge_map.get(&current) {
            if !neighbors.is_empty() {
                // Prefer neighbors with fewer connections
                let mut sorted_neighbors: Vec<&usize> = neighbors.iter().collect();
                sorted_neighbors.sort_by_key(|n| edge_map.get(n).map_or(0, |s| s.len()));
                Some(*sorted_neighbors[0])
            } else {
                None
            }
        } else {
            None
        };

        // If no valid neighbor, pick randomly from remaining
        current = next.unwrap_or_else(|| *remaining.iter().choose(&mut rng).unwrap());
        offspring.push(current);
        remaining.remove(&current);
    }

    // **Redistribute offspring into nurses using parent1's structure**
    let mut index = 0;
    let distribution: Vec<usize> = parent1.iter().map(|n| n.len()).collect();
    let mut new_solution = vec![Vec::new(); nurse_count];

    for (i, &count) in distribution.iter().enumerate() {
        new_solution[i] = offspring[index..index + count].to_vec();
        index += count;
    }

    new_solution
}


pub fn route_preserving_crossover(
    parent1: &Vec<Vec<usize>>, 
    parent2: &Vec<Vec<usize>>, 
    instance: &Instance
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let nurse_count = parent1.len();
    let patient_count = instance.patients.len();

    // Initialize children with empty routes for each nurse.
    let mut child1 = vec![Vec::new(); nurse_count];
    let mut child2 = vec![Vec::new(); nurse_count];
    let mut used_patients: HashSet<usize> = HashSet::new();
    let mut assigned_nurses: HashSet<usize> = HashSet::new(); // Track assigned nurse indices

    // Step 1: Identify common routes (including mirrored routes)
    // Build a map for parent1 routes (route -> nurse index)
    let mut route_map = HashMap::new();
    for nurse in 0..nurse_count {
        // Actively store each nurse's route from parent1 for quick lookup.
        route_map.insert(parent1[nurse].clone(), nurse);
    }

    // Iterate over parent2 routes and check for identical or mirrored matches.
    for nurse2 in 0..nurse_count {
        // Check for an exact match between parent2 and parent1.
        if let Some(&nurse1) = route_map.get(&parent2[nurse2]) {
            if !assigned_nurses.contains(&nurse1) && !assigned_nurses.contains(&nurse2) {
                // Actively copy the identical route from parent1 to both children.
                child1[nurse1] = parent1[nurse1].clone();
                child2[nurse1] = parent2[nurse2].clone();
                used_patients.extend(&child1[nurse1]);
                assigned_nurses.insert(nurse1);
                assigned_nurses.insert(nurse2);
            }
        } else {
            // Check for a mirrored route: reverse the route from parent2 and compare.
            let mut reversed_route = parent2[nurse2].clone();
            reversed_route.reverse();
            if let Some(&nurse1) = route_map.get(&reversed_route) {
                if !assigned_nurses.contains(&nurse1) && !assigned_nurses.contains(&nurse2) {
                    // Actively copy the route from parent1 (using its orientation)
                    // when a mirrored match is detected.
                    child1[nurse1] = parent1[nurse1].clone();
                    child2[nurse1] = parent1[nurse1].clone();
                    used_patients.extend(&child1[nurse1]);
                    assigned_nurses.insert(nurse1);
                    assigned_nurses.insert(nurse2);
                }
            }
        }
    }

    // Step 2: Collect remaining unassigned patients.
    let mut remaining_patients: Vec<usize> = (1..=patient_count)
        .filter(|p| !used_patients.contains(p))
        .collect();
    remaining_patients.shuffle(&mut rng);

    // Step 3: Assign remaining patients using an insertion heuristic.
    for patient in remaining_patients {
        let mut best_nurse = 0;
        let mut best_increase = f64::MAX;

        for nurse in 0..nurse_count {
            let route = &child1[nurse];
            let last_patient = route.last().copied().unwrap_or(0); // 0 represents the depot.
            // Actively compute the cost increase by appending the patient.
            let increase = instance.travel_times[last_patient][patient] 
                         + instance.travel_times[patient][0];

            if increase < best_increase {
                best_increase = increase;
                best_nurse = nurse;
            }
        }
        child1[best_nurse].push(patient);
        child2[best_nurse].push(patient);
    }

    // Step 4: Ensure nurse capacities are respected.
    for nurse in 0..nurse_count {
        let mut total_demand: f64 = child1[nurse]
            .iter()
            .map(|p| instance.patients[&p.to_string()].demand)
            .sum();

        while total_demand > instance.nurses[0].get_capacity() as f64 {
            if let Some(moved_patient) = child1[nurse].pop() {
                let new_nurse = rng.random_range(0..nurse_count);
                child1[new_nurse].push(moved_patient);
                child2[new_nurse].push(moved_patient);

                // Actively update the total demand for this nurse.
                total_demand = child1[nurse]
                    .iter()
                    .map(|p| instance.patients[&p.to_string()].demand)
                    .sum();
            } else {
                break; // Prevent infinite loop if no more patients can be moved.
            }
        }
    }

    (child1, child2)
}



pub fn merge_and_split_crossover(
    parent1: &Vec<Vec<usize>>,
    parent2: &Vec<Vec<usize>>,
    instance: &Instance,
) -> Vec<Vec<usize>> {
    // Flatten both parents into one-dimensional vectors.
    let flat1: Vec<usize> = parent1.iter().flatten().cloned().collect();
    let flat2: Vec<usize> = parent2.iter().flatten().cloned().collect();
    let n_customers = flat1.len(); // Assuming customers are numbered 1..=n_customers

    // Build maps for the position (rank) of each customer in both parents.
    let mut pos1: HashMap<usize, usize> = HashMap::new();
    for (i, &cust) in flat1.iter().enumerate() {
        pos1.insert(cust, i);
    }
    let mut pos2: HashMap<usize, usize> = HashMap::new();
    for (i, &cust) in flat2.iter().enumerate() {
        pos2.insert(cust, i);
    }

    // Create a giant tour by sorting customers by the average of their positions in the two parents.
    let mut customers: Vec<usize> = (1..=n_customers).collect();
    customers.sort_by(|&a, &b| {
        let rank_a = (pos1.get(&a).unwrap() + pos2.get(&a).unwrap()) as f64 / 2.0;
        let rank_b = (pos1.get(&b).unwrap() + pos2.get(&b).unwrap()) as f64 / 2.0;
        rank_a.partial_cmp(&rank_b).unwrap_or(Ordering::Equal)
    });

    // --- Split the giant tour into routes ---
    // Here we use a simple greedy splitting based on capacity.
    // (In practice, you might want to use a dynamic programming split that also considers time windows.)
    let n_routes = instance.nurses.len();
    let capacity = instance.nurses[0].get_capacity() as f64; // assuming all nurses have the same capacity
    let mut routes: Vec<Vec<usize>> = Vec::new();
    let mut current_route: Vec<usize> = Vec::new();
    let mut current_load: f64 = 0.0;

    for &cust in customers.iter() {
        // Look up the demand for the customer.
        let demand = instance
            .patients
            .get(&cust.to_string())
            .map(|p| p.demand)
            .unwrap_or(0.0);
        // If adding the customer does not exceed capacity (or if the current route is empty),
        // add the customer to the current route.
        if current_route.is_empty() || current_load + demand <= capacity {
            current_route.push(cust);
            current_load += demand;
        } else {
            // Otherwise, finish the current route and start a new one.
            routes.push(current_route);
            current_route = vec![cust];
            current_load = demand;
        }
    }
    if !current_route.is_empty() {
        routes.push(current_route);
    }

    // --- Adjust the number of routes to match the number of vehicles (nurses) ---
    // If we have too few routes, add empty ones.
    while routes.len() < n_routes {
        routes.push(vec![]);
    }
    // If we have too many routes, merge some of them.
    while routes.len() > n_routes {
        // As a simple strategy, sort routes by total load and merge the two with the smallest loads.
        routes.sort_by(|a, b| {
            let load_a: f64 = a.iter().map(|&c| {
                instance
                    .patients
                    .get(&c.to_string())
                    .map(|p| p.demand)
                    .unwrap_or(0.0)
            }).sum();
            let load_b: f64 = b.iter().map(|&c| {
                instance
                    .patients
                    .get(&c.to_string())
                    .map(|p| p.demand)
                    .unwrap_or(0.0)
            }).sum();
            load_a.partial_cmp(&load_b).unwrap_or(Ordering::Equal)
        });
        // Merge the two smallest routes.
        let route1 = routes.remove(0);
        let route2 = routes.remove(0);
        let merged = [route1, route2].concat();
        routes.push(merged);
    }

    // Optionally, randomize the order of routes.
    let mut rng = rand::rng();
    routes.shuffle(&mut rng);
    routes
}


pub fn route_preserving_crossover_with_division(
    parent1: &Vec<Vec<usize>>, 
    parent2: &Vec<Vec<usize>>, 
    instance: &Instance,
    division_rate: f64
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    use rand::Rng; 
    let mut rng = rand::rng();
    let nurse_count = parent1.len();
    let patient_count = instance.patients.len();

    // Initialize children with empty routes for each nurse.
    let mut child1 = vec![Vec::new(); nurse_count];
    let mut child2 = vec![Vec::new(); nurse_count];
    let mut used_patients: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut assigned_nurses: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Step 1: Identify common routes (including mirrored routes)
    let mut route_map = std::collections::HashMap::new();
    for nurse in 0..nurse_count {
        route_map.insert(parent1[nurse].clone(), nurse);
    }

    for nurse2 in 0..nurse_count {
        // Check for an exact match between parent2 and parent1.
        if let Some(&nurse1) = route_map.get(&parent2[nurse2]) {
            if !assigned_nurses.contains(&nurse1) && !assigned_nurses.contains(&nurse2) {
                child1[nurse1] = parent1[nurse1].clone();
                child2[nurse1] = parent2[nurse2].clone();
                used_patients.extend(&child1[nurse1]);
                assigned_nurses.insert(nurse1);
                assigned_nurses.insert(nurse2);
            }
        } else {
            // Check for a mirrored route: reverse parent2's route and compare.
            let mut reversed_route = parent2[nurse2].clone();
            reversed_route.reverse();
            if let Some(&nurse1) = route_map.get(&reversed_route) {
                if !assigned_nurses.contains(&nurse1) && !assigned_nurses.contains(&nurse2) {
                    child1[nurse1] = parent1[nurse1].clone();
                    child2[nurse1] = parent1[nurse1].clone();
                    used_patients.extend(&child1[nurse1]);
                    assigned_nurses.insert(nurse1);
                    assigned_nurses.insert(nurse2);
                }
            }
        }
    }

    // Step 2: Collect remaining unassigned patients.
    let mut remaining_patients: Vec<usize> = (1..=patient_count)
        .filter(|p| !used_patients.contains(p))
        .collect();
    remaining_patients.shuffle(&mut rng);

    // Step 3: Assign remaining patients using an insertion heuristic.
    for patient in remaining_patients {
        let mut best_nurse = 0;
        let mut best_increase = f64::MAX;

        for nurse in 0..nurse_count {
            let route = &child1[nurse];
            let last_patient = route.last().copied().unwrap_or(0); // 0 represents the depot.
            let increase = instance.travel_times[last_patient][patient] 
                         + instance.travel_times[patient][0];
            if increase < best_increase {
                best_increase = increase;
                best_nurse = nurse;
            }
        }
        child1[best_nurse].push(patient);
        child2[best_nurse].push(patient);
    }

    // Step 4: Ensure nurse capacities are respected.
    for nurse in 0..nurse_count {
        let mut total_demand: f64 = child1[nurse]
            .iter()
            .map(|p| instance.patients[&p.to_string()].demand)
            .sum();

        while total_demand > instance.nurses[0].get_capacity() as f64 {
            if let Some(moved_patient) = child1[nurse].pop() {
                let new_nurse = rng.random_range(0..nurse_count);
                child1[new_nurse].push(moved_patient);
                child2[new_nurse].push(moved_patient);

                total_demand = child1[nurse]
                    .iter()
                    .map(|p| instance.patients[&p.to_string()].demand)
                    .sum();
            } else {
                break; // Prevent infinite loop if no more patients can be moved.
            }
        }
    }

    // Helper closure to compute route cost (depot -> first patient, between patients, and back to depot).
    let compute_route_cost = |route: &Vec<usize>| -> f64 {
        if route.is_empty() {
            0.0
        } else {
            let mut cost = instance.travel_times[0][route[0]];
            for window in route.windows(2) {
                cost += instance.travel_times[window[0]][window[1]];
            }
            cost + instance.travel_times[*route.last().unwrap()][0]
        }
    };

    // Step 5: (Probabilistic) Divide the longest route at a random point,
    // and assign one of the divided sub-routes to an unused nurse.
    if rng.random::<f64>() < division_rate {
        // Process child1.
        if let Some(unused_nurse_idx) = child1.iter().position(|r| r.is_empty()) {
            let mut longest_route_index: Option<usize> = None;
            let mut longest_cost = 0.0;
            for (i, route) in child1.iter().enumerate() {
                if !route.is_empty() && i != unused_nurse_idx && route.len() >= 2 {
                    let cost = compute_route_cost(route);
                    if cost > longest_cost {
                        longest_cost = cost;
                        longest_route_index = Some(i);
                    }
                }
            }
            if let Some(route_idx) = longest_route_index {
                let route = &mut child1[route_idx];
                let split_point = rng.random_range(1..route.len()); // ensures both segments are non-empty
                let new_route_segment = route.split_off(split_point);
                child1[unused_nurse_idx] = new_route_segment;
            }
        }

        // Process child2 similarly.
        if let Some(unused_nurse_idx) = child2.iter().position(|r| r.is_empty()) {
            let mut longest_route_index: Option<usize> = None;
            let mut longest_cost = 0.0;
            for (i, route) in child2.iter().enumerate() {
                if !route.is_empty() && i != unused_nurse_idx && route.len() >= 2 {
                    let cost = compute_route_cost(route);
                    if cost > longest_cost {
                        longest_cost = cost;
                        longest_route_index = Some(i);
                    }
                }
            }
            if let Some(route_idx) = longest_route_index {
                let route = &mut child2[route_idx];
                let split_point = rng.random_range(1..route.len());
                let new_route_segment = route.split_off(split_point);
                child2[unused_nurse_idx] = new_route_segment;
            }
        }
    }

    (child1, child2)
}


pub fn route_preserving_crossover_with_random_division(
    parent1: &Vec<Vec<usize>>, 
    parent2: &Vec<Vec<usize>>, 
    instance: &Instance,
    division_rate: f64
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    use rand::Rng;
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    let nurse_count = parent1.len();
    let patient_count = instance.patients.len();

    // Initialize children with empty routes for each nurse.
    let mut child1 = vec![Vec::new(); nurse_count];
    let mut child2 = vec![Vec::new(); nurse_count];
    let mut used_patients: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut assigned_nurses: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Step 1: Identify common routes (including mirrored routes)
    let mut route_map = std::collections::HashMap::new();
    for nurse in 0..nurse_count {
        route_map.insert(parent1[nurse].clone(), nurse);
    }

    for nurse2 in 0..nurse_count {
        // Check for an exact match between parent2 and parent1.
        if let Some(&nurse1) = route_map.get(&parent2[nurse2]) {
            if !assigned_nurses.contains(&nurse1) && !assigned_nurses.contains(&nurse2) {
                child1[nurse1] = parent1[nurse1].clone();
                child2[nurse1] = parent2[nurse2].clone();
                used_patients.extend(&child1[nurse1]);
                assigned_nurses.insert(nurse1);
                assigned_nurses.insert(nurse2);
            }
        } else {
            // Check for a mirrored route: reverse parent2's route and compare.
            let mut reversed_route = parent2[nurse2].clone();
            reversed_route.reverse();
            if let Some(&nurse1) = route_map.get(&reversed_route) {
                if !assigned_nurses.contains(&nurse1) && !assigned_nurses.contains(&nurse2) {
                    child1[nurse1] = parent1[nurse1].clone();
                    child2[nurse1] = parent1[nurse1].clone();
                    used_patients.extend(&child1[nurse1]);
                    assigned_nurses.insert(nurse1);
                    assigned_nurses.insert(nurse2);
                }
            }
        }
    }

    // Step 2: Collect remaining unassigned patients.
    let mut remaining_patients: Vec<usize> = (1..=patient_count)
        .filter(|p| !used_patients.contains(p))
        .collect();
    remaining_patients.shuffle(&mut rng);

    // Step 3: Assign remaining patients using an insertion heuristic.
    for patient in remaining_patients {
        let mut best_nurse = 0;
        let mut best_increase = f64::MAX;

        for nurse in 0..nurse_count {
            let route = &child1[nurse];
            let last_patient = route.last().copied().unwrap_or(0); // 0 represents the depot.
            let increase = instance.travel_times[last_patient][patient] 
                         + instance.travel_times[patient][0];
            if increase < best_increase {
                best_increase = increase;
                best_nurse = nurse;
            }
        }
        child1[best_nurse].push(patient);
        child2[best_nurse].push(patient);
    }

    // Step 4: Ensure nurse capacities are respected.
    for nurse in 0..nurse_count {
        let mut total_demand: f64 = child1[nurse]
            .iter()
            .map(|p| instance.patients[&p.to_string()].demand)
            .sum();

        while total_demand > instance.nurses[0].get_capacity() as f64 {
            if let Some(moved_patient) = child1[nurse].pop() {
                let new_nurse = rng.random_range(0..nurse_count);
                child1[new_nurse].push(moved_patient);
                child2[new_nurse].push(moved_patient);

                total_demand = child1[nurse]
                    .iter()
                    .map(|p| instance.patients[&p.to_string()].demand)
                    .sum();
            } else {
                break; // Prevent infinite loop if no more patients can be moved.
            }
        }
    }

    // Helper closure to compute route cost (depot -> first patient, between patients, and back to depot).
    let compute_route_cost = |route: &Vec<usize>| -> f64 {
        if route.is_empty() {
            0.0
        } else {
            let mut cost = instance.travel_times[0][route[0]];
            for window in route.windows(2) {
                cost += instance.travel_times[window[0]][window[1]];
            }
            cost + instance.travel_times[*route.last().unwrap()][0]
        }
    };

    // Step 5: (Probabilistic) Division:
    // With probability division_rate, split the longest route (with at least 2 patients)
    // and insert the removed segment into a randomly chosen target route.
    if rng.random::<f64>() < division_rate {
        // Process child1.
        {
            // Identify the longest route that can be split.
            let longest_route_idx = (0..nurse_count)
                .filter(|&i| child1[i].len() >= 2)
                .max_by(|&i, &j| {
                    compute_route_cost(&child1[i])
                        .partial_cmp(&compute_route_cost(&child1[j]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(split_idx) = longest_route_idx {
                let route_len = child1[split_idx].len();
                let split_point = rng.random_range(1..route_len); // Ensure both segments are non-empty.
                let relocate_front = rng.random_bool(0.5);
                let relocated_segment: Vec<usize>;
                if relocate_front {
                    relocated_segment = child1[split_idx].drain(0..split_point).collect();
                } else {
                    relocated_segment = child1[split_idx].split_off(split_point);
                }
                // Choose a target route (different from the one we just split).
                let target_candidates: Vec<usize> = (0..nurse_count)
                    .filter(|&i| i != split_idx)
                    .collect();
                let target_idx = *target_candidates.choose(&mut rng).unwrap();
                // Choose a random insertion position in the target route.
                let insert_pos = rng.random_range(0..=child1[target_idx].len());
                child1[target_idx].splice(insert_pos..insert_pos, relocated_segment);
            }
        }

        // Process child2 similarly.
        {
            let longest_route_idx = (0..nurse_count)
                .filter(|&i| child2[i].len() >= 2)
                .max_by(|&i, &j| {
                    compute_route_cost(&child2[i])
                        .partial_cmp(&compute_route_cost(&child2[j]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(split_idx) = longest_route_idx {
                let route_len = child2[split_idx].len();
                let split_point = rng.random_range(1..route_len);
                let relocate_front = rng.random_bool(0.5);
                let relocated_segment: Vec<usize>;
                if relocate_front {
                    relocated_segment = child2[split_idx].drain(0..split_point).collect();
                } else {
                    relocated_segment = child2[split_idx].split_off(split_point);
                }
                let target_candidates: Vec<usize> = (0..nurse_count)
                    .filter(|&i| i != split_idx)
                    .collect();
                let target_idx = *target_candidates.choose(&mut rng).unwrap();
                let insert_pos = rng.random_range(0..=child2[target_idx].len());
                child2[target_idx].splice(insert_pos..insert_pos, relocated_segment);
            }
        }
    }

    (child1, child2)
}




pub fn select_delete_fix_crossover(
    parent1: &Vec<Vec<usize>>,
    parent2: &Vec<Vec<usize>>,
    instance: &Instance,
    crossover_rate: f64,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {


    // Create a random number generator.
    let mut rng = rand::rng();

    // If the crossover rate is not met, return the parents as they are.
    if rng.random::<f64>() > crossover_rate {
        return (parent1.clone(), parent2.clone());
    }

    let nurse_count = parent1.len();

    // Select a random nurse route index from each parent.
    let route_idx1 = rng.random_range(0..nurse_count);
    let route_idx2 = rng.random_range(0..nurse_count);
    let selected_route_parent1 = parent1[route_idx1].clone();
    let selected_route_parent2 = parent2[route_idx2].clone();

    // Create children as clones of the parents.
    let mut child1 = parent1.clone();
    let mut child2 = parent2.clone();

    // For child1, remove from every route any patient that appears in parent2's selected route.
    // Store the removed patients so they can be reinserted.
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

    // For child2, remove from every route any patient that appears in parent1's selected route.
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

    // Wrap instance in an Arc for thread safety in parallel evaluations.
    let instance_arc = Arc::new(instance.clone());

    // "Fix" child1: reinsert each missing patient at the best insertion (lowest overall fitness).
    for patient in missing_from_child1 {
        // Build candidate insertion positions as (route_index, insertion_position).
        let candidate_positions: Vec<(usize, usize)> = (0..child1.len())
            .flat_map(|r_idx| {
                // Allow insertion at any position in route, including at the beginning or end.
                (0..=child1[r_idx].len()).map(move |pos| (r_idx, pos))
            })
            .collect();

        // Evaluate candidates in parallel.
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

    // "Fix" child2 by reinserting each missing patient using the fitness function.
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


pub fn random_route_mixing_crossover(
    parent1: &Vec<Vec<usize>>, 
    parent2: &Vec<Vec<usize>>,
    crossover_rate: f64,
) -> Vec<Vec<usize>> {
    let mut rng = rand::rng();
    // With probability not meeting the crossover rate, return a clone of parent1.
    if rng.random::<f64>() > crossover_rate {
        return parent1.clone();
    }

    // All patients should appear exactly once across the routes.
    // We flatten one parent's routes (they should represent the same set as parent2).
    let mut all_patients: Vec<usize> = parent1.iter().flatten().cloned().collect();
    // Shuffle the list completely to break any route structure.
    all_patients.shuffle(&mut rng);

    // Determine the number of nurses (routes).
    let nurse_count = parent1.len();
    let total_patients = all_patients.len();

    // Compute the average route size for each nurse from the two parents.
    let mut avg_sizes = Vec::with_capacity(nurse_count);
    for i in 0..nurse_count {
        // Note: if parent routes have different lengths, take the average.
        let size = (parent1[i].len() + parent2[i].len()) / 2;
        avg_sizes.push(size);
    }
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

    // Partition the shuffled patient list according to route_sizes.
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
    // In case there are fewer routes than nurses, add empty routes.
    while new_solution.len() < nurse_count {
        new_solution.push(Vec::new());
    }

    new_solution
}


pub fn meta_crossover(
    parent1: &Vec<Vec<usize>>,
    parent2: &Vec<Vec<usize>>,
    instance: &Instance,
    crossover_rate: f64,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    if rng.random::<f64>() < 0.7 {
        // Use select-delete-fix crossover.
        select_delete_fix_crossover(parent1, parent2, instance, crossover_rate)
    } else {
        // Use random-route-mixing crossover and duplicate the result.
        let offspring = random_route_mixing_crossover(parent1, parent2, crossover_rate);
        (offspring.clone(), offspring)
    }
}
