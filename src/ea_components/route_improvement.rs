use rayon::prelude::*;
use dashmap::DashMap;
use rand::Rng;
use crate::ea_components::fitness::fitness;
use crate::structs::instance::Instance;

// =============== MAIN WRAPPER FUNCTION ===============

/// Iteratively improves the offspring solution by applying intra-route and inter-route improvement passes.
/// 
/// This function repeatedly applies the intra-route and inter-route improvement passes to the current solution
/// until neither pass yields a further improvement. The improvements are performed using local search operators
/// that adjust the routes to reduce the overall fitness value.
/// 
/// # Parameters
/// - `instance` - A reference to the problem instance containing travel times, patient data, and depot details.
/// - `offspring` - A mutable reference to the current solution, represented as a vector of routes (each route is a vector of patient IDs).
/// - `cache` - A shared cache for storing computed fitness values to avoid redundant evaluations.
pub fn route_improvement(
    instance: &Instance,
    offspring: &mut Vec<Vec<usize>>,
    cache: &DashMap<String, f64>,
) {
    loop {
        let improved_intra = intra_route_improvement_pass(instance, offspring, cache);
        let improved_inter = inter_route_improvement_pass(instance, offspring, cache);

        // If neither pass found any improvement, stop the loop.
        if !improved_intra && !improved_inter {
            break;
        }
    }
}

// =============== PASS #1: INTRA-ROUTE IMPROVEMENT ===============

/// Performs an intra-route improvement pass on the given solution.
/// 
/// This function samples a subset of routes and attempts local improvements on each by applying intra-route operators,
/// such as 2-opt and Or-opt moves, to reduce the overall fitness value. The first improvement found is applied to update the solution.
/// 
/// # Parameters
/// - `instance` - A reference to the problem instance.
/// - `offspring` - A mutable reference to the solution, represented as a vector of routes.
/// - `cache` - A cache for storing fitness evaluations.
/// 
/// # Returns
/// Returns `true` if an improvement is found and applied; otherwise, returns `false`.
fn intra_route_improvement_pass(
    instance: &Instance,
    offspring: &mut Vec<Vec<usize>>,
    cache: &DashMap<String, f64>,
) -> bool {

    // STEP 1: Determine the number of routes in the current solution.
    let n_routes = offspring.len();
    if n_routes == 0 {
        return false;
    }

    // STEP 2: Sample a subset of routes for improvement.
    // About 5% of the routes (at least one) are sampled to limit the computational load.
    let mut rng = rand::rng();
    let sample_size = std::cmp::max(1, (n_routes as f64 * 0.05).ceil() as usize); // e.g., sample 5% of routes
    let routes: Vec<usize> = (0..sample_size)
        .map(|_| rng.random_range(0..n_routes))
        .collect();

    // STEP 3: Search for an improvement in parallel among the sampled routes.
    // For each sampled route, attempt intra-route improvements using 2-opt or Or-opt moves.
    let candidate = routes.par_iter().find_map_any(|&r_i| {

        // STEP 3.1: Skip routes that are too short for improvement.
        if offspring[r_i].len() < 3 {
            return None;
        }
        // STEP 3.2: Attempt a 2-opt improvement on the selected route.
        if let Some(new_offspring) = try_2opt_intra_route(r_i, offspring, instance, cache) {
            return Some(new_offspring);
        }
        // STEP 3.3: If 2-opt fails, attempt an Or-opt improvement on the same route.
        if let Some(new_offspring) = try_or_opt_intra_route(r_i, offspring, instance, cache) {
            return Some(new_offspring);
        }
        None
    });

    // STEP 4: If an improved solution is found, update the offspring and signal improvement.
    if let Some(new_offspring) = candidate {
        *offspring = new_offspring;
        true
    } else {
        false
    }
}

// =============== PASS #2: INTER-ROUTE IMPROVEMENT ===============

/// Performs an inter-route improvement pass on the given solution.
/// 
/// This function samples pairs of distinct routes and applies inter-route operators, such as relocation and swap moves,
/// to explore improvements between routes. If an improvement is identified, the solution is updated accordingly.
/// 
/// # Parameters
/// - `instance` - A reference to the problem instance.
/// - `offspring` - A mutable reference to the solution, represented as a vector of routes.
/// - `cache` - A cache for storing fitness evaluations.
/// 
/// # Returns
/// Returns `true` if an improvement is found and applied; otherwise, returns `false`.
fn inter_route_improvement_pass(
    instance: &Instance,
    offspring: &mut Vec<Vec<usize>>,
    cache: &DashMap<String, f64>,
) -> bool {

    // STEP 1: Verify that there are at least two routes to consider for inter-route improvements.
    let n_routes = offspring.len();
    if n_routes < 2 {
        return false;
    }

    // STEP 2: Create a random sample of distinct route pairs.
    // Sample about 1% of the total routes (at least one pair) to limit computational overhead.
    let mut rng = rand::rng();
    let sample_size = std::cmp::max(1, (n_routes as f64 * 0.01).ceil() as usize); // e.g., sample 2% pairs
    let pairs: Vec<(usize, usize)> = (0..sample_size)
        .map(|_| {
            let r_i = rng.random_range(0..n_routes);
            let mut r_j = rng.random_range(0..n_routes);
            // ensure r_j != r_i
            while r_j == r_i {
                r_j = rng.random_range(0..n_routes);
            }
            (r_i, r_j)
        })
        .collect();

    // STEP 3: In parallel, explore potential improvements for each sampled pair.
    let candidate = pairs.par_iter().find_map_any(|&(r_i, r_j)| {

        // STEP 3.1: Ensure both routes have sufficient length for improvement.
        if offspring[r_i].len() < 3 || offspring[r_j].len() < 3 {
            return None;
        }
        // STEP 3.2: Attempt to improve via relocation between the two routes.
        if let Some(new_offspring) = try_all_relocations(r_i, r_j, offspring, instance, cache) {
            return Some(new_offspring);
        }
        // STEP 3.3: If relocation fails, attempt to improve via swapping patients between the routes.
        if let Some(new_offspring) = try_all_swaps(r_i, r_j, offspring, instance, cache) {
            return Some(new_offspring);
        }
        None
    });

    // STEP 4: If an improved solution is found, update the offspring and indicate success.
    if let Some(new_offspring) = candidate {
        *offspring = new_offspring;
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// 2-OPT (Intra-route)
// ---------------------------------------------------------------------------

/// Attempts a 2-opt intra-route improvement on the specified route.
/// 
/// This function iterates over pairs of edges in the route and reverses the segment between them to potentially lower the total cost.
/// If a move results in an improved fitness value, the updated solution is returned.
/// 
/// # Parameters
/// - `route_idx` - The index of the route within the solution to be improved.
/// - `offspring` - A reference to the current solution, represented as a vector of routes.
/// - `instance` - A reference to the problem instance.
/// - `cache` - A cache for storing computed fitness values.
/// 
/// # Returns
/// Returns an improved solution as an `Option`; returns `None` if no improvement is found.
fn try_2opt_intra_route(
    route_idx: usize,
    offspring: &Vec<Vec<usize>>,
    instance: &Instance,
    cache: &DashMap<String, f64>,
) -> Option<Vec<Vec<usize>>> {

    // STEP 1: Retrieve the target route and its length.
    let route = &offspring[route_idx];
    let len = route.len();
    
    // STEP 2: Iterate over candidate edge pairs for a 2-opt move.
    // We choose indices i and j such that reversing the segment between i+1 and j may yield an improvement.
    // (Note: The depot is assumed to be at index 0 and at the end, so these positions are skipped.)
    for i in 1..(len - 2) {
        for j in (i + 2)..(len - 1) {

            // STEP 3: Clone the current solution to create a candidate solution.
            let mut new_offspring = offspring.clone();
            let mut new_route = new_offspring[route_idx].clone();

            // STEP 4: Reverse the segment from index i+1 to j (inclusive) as a candidate 2-opt move.
            new_route[i+1..=j].reverse();
            new_offspring[route_idx] = new_route;

            // STEP 5: Evaluate the fitness of the candidate solution.
            let new_cost = fitness(&new_offspring, instance);
            let key = format!("{:?}", new_offspring);

            // STEP 6: Compare the candidate solution's fitness with any cached fitness.
            // If the candidate is an improvement, update the cache and return the candidate solution.
            if let Some(old_cost) = cache.get(&key) {
                if new_cost < *old_cost {
                    cache.insert(key, new_cost);
                    return Some(new_offspring);
                }
            } else {
                cache.insert(key, new_cost);
                return Some(new_offspring);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Or-Opt (Intra-route): move a chain of length 1..3 to a new position
// ---------------------------------------------------------------------------

/// Attempts an Or-opt intra-route improvement on the specified route.
/// 
/// This function considers removing a contiguous chain of 1 to 3 patients from the route and reinserting it at a different position.
/// If the move reduces the overall fitness value, the updated solution is returned.
/// 
/// # Parameters
/// - `route_idx` - The index of the route to be improved.
/// - `offspring` - A reference to the current solution, represented as a vector of routes.
/// - `instance` - A reference to the problem instance.
/// - `cache` - A cache for storing fitness evaluations.
/// 
/// # Returns
/// Returns an improved solution as an `Option`; returns `None` if no improvement is found.
fn try_or_opt_intra_route(
    route_idx: usize,
    offspring: &Vec<Vec<usize>>,
    instance: &Instance,
    cache: &DashMap<String, f64>,
) -> Option<Vec<Vec<usize>>> {

    // STEP 1: Retrieve the target route and determine its length.
    let route = &offspring[route_idx];
    let len = route.len();

    // STEP 2: Iterate over possible chain sizes (from 1 to 3) to remove a contiguous segment.
    for chain_size in 1..=3 {

        // STEP 2.1: If the chain size exceeds the interior of the route, break out.
        if chain_size >= len - 1 {
            break;
        }

        // STEP 3: For each possible starting index of the chain in the route.
        for start in 1..(len - chain_size) {
            let end = start + chain_size - 1;

            // STEP 3.1: For each possible insertion position for the removed chain.
            for insert_pos in 1..(len - chain_size) {
                // STEP 3.2: Skip if the insertion position is within the original chain location.
                if insert_pos >= start && insert_pos <= end {
                    continue;
                }
                // STEP 4: Clone the current solution and extract the route to be modified.
                let mut new_offspring = offspring.clone();
                let mut new_route = new_offspring[route_idx].clone();

                // STEP 5: Remove the chain from the route.
                let chain: Vec<_> = new_route.drain(start..=end).collect();

                // STEP 6: Adjust the insertion index if the removal affected the positions.
                let adjusted_insert = if insert_pos > end {
                    insert_pos - chain_size
                } else {
                    insert_pos
                };

                // STEP 7: Reinsert the removed chain at the new insertion position.
                for (i, val) in chain.iter().enumerate() {
                    new_route.insert(adjusted_insert + i, *val);
                }
                new_offspring[route_idx] = new_route;

                // STEP 8: Evaluate the fitness of the candidate solution.
                let new_cost = fitness(&new_offspring, instance);
                let key = format!("{:?}", new_offspring);

                // STEP 9: If the candidate solution improves the fitness, update the cache and return it.
                if let Some(old_cost) = cache.get(&key) {
                    if new_cost < *old_cost {
                        cache.insert(key, new_cost);
                        return Some(new_offspring);
                    }
                } else {
                    cache.insert(key, new_cost);
                    return Some(new_offspring);
                }
            }
        }
    }
    None
}

/// Attempts to improve the solution by swapping patients between two routes.
/// 
/// This function iterates over potential swap positions between routes `r_i` and `r_j`. For each candidate swap,
/// it evaluates the resulting solution's fitness and returns the updated solution if an improvement is found.
/// 
/// # Parameters
/// - `r_i` - The index of the first route.
/// - `r_j` - The index of the second route.
/// - `offspring` - A reference to the current solution, represented as a vector of routes.
/// - `instance` - A reference to the problem instance.
/// - `cache` - A cache for storing fitness evaluations.
/// 
/// # Returns
/// Returns an improved solution as an `Option`; returns `None` if no improvement is found.
fn try_all_swaps(
    r_i: usize,
    r_j: usize,
    offspring: &Vec<Vec<usize>>,
    instance: &Instance,
    cache: &DashMap<String, f64>,
) -> Option<Vec<Vec<usize>>> {

    // STEP 1: Ensure both selected routes have sufficient elements (at least 3) for swapping.
    if offspring[r_i].len() < 3 || offspring[r_j].len() < 3 {
        return None;
    }

    // STEP 2: Iterate over possible swap positions in the two routes.
    for i in 1..offspring[r_i].len() - 1 {
        for j in 1..offspring[r_j].len() - 1 {
            // STEP 2.1: Clone the current solution to create a candidate offspring.
            let mut new_offspring = offspring.clone();
            if r_i == r_j {
                // STEP 2.2a: For the same route, perform an in-place swap to avoid shifting issues.
                new_offspring[r_i].swap(i, j);
            } else {
                // STEP 2.2b: For different routes, remove and reinsert patients to swap them.
                let mut route_i = new_offspring[r_i].clone();
                let mut route_j = new_offspring[r_j].clone();
                let patient_i = route_i.remove(i);
                let patient_j = route_j.remove(j);
                route_i.insert(i, patient_j);
                route_j.insert(j, patient_i);
                new_offspring[r_i] = route_i;
                new_offspring[r_j] = route_j;
            }

            // STEP 3: Evaluate the fitness of the candidate solution.
            let new_cost = fitness(&new_offspring, instance);
            let key = format!("{:?}", new_offspring);

            // STEP 4: If the candidate improves the fitness, update the cache and return the new solution.
            if let Some(old_cost) = cache.get(&key) {
                if new_cost < *old_cost {
                    cache.insert(key, new_cost);
                    return Some(new_offspring);
                }
            } else {
                cache.insert(key, new_cost);
                return Some(new_offspring);
            }
        }
    }
    None
}


/// Attempts to improve the solution by relocating a patient between two routes.
/// 
/// This function explores potential moves by removing a patient from route `r_i` and reinserting it into route `r_j` (or within the same route)
/// at different positions. It returns the updated solution if a relocation leads to a reduced fitness value.
/// 
/// # Parameters
/// - `r_i` - The index of the source route.
/// - `r_j` - The index of the target route.
/// - `offspring` - A reference to the current solution, represented as a vector of routes.
/// - `instance` - A reference to the problem instance.
/// - `cache` - A cache for storing fitness evaluations.
/// 
/// # Returns
/// Returns an improved solution as an `Option`; returns `None` if no improvement is found.
fn try_all_relocations(
    r_i: usize,
    r_j: usize,
    offspring: &Vec<Vec<usize>>,
    instance: &Instance,
    cache: &DashMap<String, f64>,
) -> Option<Vec<Vec<usize>>> {

    // STEP 1: Check that both selected routes have sufficient elements for relocation.
    if offspring[r_i].len() < 3 || offspring[r_j].len() < 3 {
        return None;
    }

    // STEP 2: Iterate over potential removal positions in route r_i.
    for i in 1..offspring[r_i].len() - 1 {
        // STEP 3: Iterate over potential insertion positions in route r_j.
        for j in 1..offspring[r_j].len() - 1 {
            // STEP 4: Clone the current solution to create a candidate offspring.
            let mut new_offspring = offspring.clone();
            if r_i == r_j {
                // STEP 4.1: For relocation within the same route, remove the patient at position i.
                let mut route = new_offspring[r_i].clone();
                let patient = route.remove(i);
                // STEP 4.2: Adjust the insertion index to account for the removal.
                let insert_index = if i < j { j - 1 } else { j };
                // STEP 4.3: Reinsert the patient at the new position within the same route.
                route.insert(insert_index, patient);
                new_offspring[r_i] = route;
            } else {
                // STEP 4.4: For relocation between different routes, remove the patient from route r_i.
                let mut route_i = new_offspring[r_i].clone();
                let mut route_j = new_offspring[r_j].clone();
                let patient = route_i.remove(i);
                // STEP 4.5: Insert the patient into route r_j at position j.
                route_j.insert(j, patient);
                new_offspring[r_i] = route_i;
                new_offspring[r_j] = route_j;
            }

            // STEP 5: Evaluate the candidate solution's fitness.
            let new_cost = fitness(&new_offspring, instance);
            let key = format!("{:?}", new_offspring);

            // STEP 6: Compare with cached fitness and update the cache if improvement is found.
            if let Some(old_cost) = cache.get(&key) {
                if new_cost < *old_cost {
                    cache.insert(key, new_cost);
                    return Some(new_offspring);
                }
            } else {
                cache.insert(key, new_cost);
                return Some(new_offspring);
            }
        }
    }
    None
}
