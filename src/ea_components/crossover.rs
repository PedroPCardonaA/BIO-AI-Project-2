use std::{cmp::Ordering, collections::{HashMap, HashSet}};

use rand::{seq::{IndexedRandom, IteratorRandom, SliceRandom}, Rng};

use crate::structs::instance::Instance;

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

