use rayon::prelude::*;
use dashmap::DashMap;
use rand::Rng;
use crate::ea_components::fitness::fitness;
use crate::structs::instance::Instance;

pub fn route_improvement(
    instance: &Instance,
    offspring: &mut Vec<Vec<usize>>,
    cache: &DashMap<String, f64>,
) {
    loop {
        let n_routes = offspring.len();
        if n_routes == 0 {
            break;
        }
        // Compute total possible pairs (r_i, r_j) with r_i ≤ r_j.
        let total_pairs = n_routes * (n_routes + 1) / 2;
        // Sample 1% of these pairs (at least one pair).
        let sample_size = std::cmp::max(1, ((total_pairs as f64) * 0.01).ceil() as usize);
        let mut rng = rand::thread_rng();
        let sample: Vec<(usize, usize)> = (0..sample_size)
            .map(|_| {
                let r_i = rng.gen_range(0..n_routes);
                // Ensure r_j is in [r_i, n_routes)
                let r_j = rng.gen_range(r_i..n_routes);
                (r_i, r_j)
            })
            .collect();

        // In parallel, try to find any improvement candidate among the sampled pairs.
        let candidate = sample.par_iter().find_map_any(|&(r_i, r_j)| {
            // If both routes are too short, skip.
            if offspring[r_i].len() < 3 || offspring[r_j].len() < 3 {
                return None;
            }
            
            if r_i == r_j {
                // ============= Intra-route improvements ============= //
                // 1) 2-Opt
                if let Some(new_offspring) = try_2opt_intra_route(r_i, offspring, instance, cache) {
                    return Some(new_offspring);
                }
                // 2) Or-Opt (small chain removal)
                if let Some(new_offspring) = try_or_opt_intra_route(r_i, offspring, instance, cache) {
                    return Some(new_offspring);
                }
                // (You could add 3‐Opt, etc. here as well)
            } else {
                // ============= Inter-route improvements ============= //
                // 1) Relocation
                if let Some(new_offspring) = try_all_relocations(r_i, r_j, offspring, instance, cache) {
                    return Some(new_offspring);
                }
                // 2) Swap
                if let Some(new_offspring) = try_all_swaps(r_i, r_j, offspring, instance, cache) {
                    return Some(new_offspring);
                }
            }
            None
        });

        // If an improvement candidate was found, update offspring and repeat;
        // otherwise, break the loop.
        if let Some(new_offspring) = candidate {
            *offspring = new_offspring;
        } else {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// 2-OPT (Intra-route)
// ---------------------------------------------------------------------------
fn try_2opt_intra_route(
    route_idx: usize,
    offspring: &Vec<Vec<usize>>,
    instance: &Instance,
    cache: &DashMap<String, f64>,
) -> Option<Vec<Vec<usize>>> {
    let route = &offspring[route_idx];
    let len = route.len();
    // For a route with N stops (including start/end if included), you can 2-Opt
    // edges (i, i+1) and (j, j+1) for 1 <= i < j-1 <= len-2
    // This code uses a typical VRP convention that 0 and last are depots,
    // so we skip them for 2-Opt. Adjust if your indexing differs.
    for i in 1..(len - 2) {
        for j in (i + 2)..(len - 1) {
            let mut new_offspring = offspring.clone();
            let mut new_route = new_offspring[route_idx].clone();
            // Reverse the segment [i+1..=j]
            new_route[i+1..=j].reverse();
            new_offspring[route_idx] = new_route;

            // Evaluate cost
            let new_cost = fitness(&new_offspring, instance);
            let key = format!("{:?}", new_offspring);
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
fn try_or_opt_intra_route(
    route_idx: usize,
    offspring: &Vec<Vec<usize>>,
    instance: &Instance,
    cache: &DashMap<String, f64>,
) -> Option<Vec<Vec<usize>>> {
    let route = &offspring[route_idx];
    let len = route.len();

    // Or-opt tries removing a chain of size k from [1..k..(len-2)]
    // then reinserting it elsewhere in the route.
    for chain_size in 1..=3 {
        if chain_size >= len - 1 {
            // If chain_size is bigger than route interior, skip
            break;
        }
        for start in 1..(len - chain_size) {
            let end = start + chain_size - 1;
            // The segment we remove is [start..=end].
            for insert_pos in 1..(len - chain_size) {
                // Skip the trivial case of re‐inserting exactly where it was.
                if insert_pos >= start && insert_pos <= end {
                    continue;
                }
                let mut new_offspring = offspring.clone();
                let mut new_route = new_offspring[route_idx].clone();

                // Remove the chain
                let chain: Vec<_> = new_route.drain(start..=end).collect();
                // Because we removed elements before insert_pos if insert_pos > end
                // we adjust the insertion index if necessary:
                let adjusted_insert = if insert_pos > end {
                    insert_pos - chain_size
                } else {
                    insert_pos
                };
                // Insert the chain
                for (i, val) in chain.iter().enumerate() {
                    new_route.insert(adjusted_insert + i, *val);
                }
                new_offspring[route_idx] = new_route;

                // Evaluate cost
                let new_cost = fitness(&new_offspring, instance);
                let key = format!("{:?}", new_offspring);
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

fn try_all_swaps(
    r_i: usize,
    r_j: usize,
    offspring: &Vec<Vec<usize>>,
    instance: &Instance,
    cache: &DashMap<String, f64>,
) -> Option<Vec<Vec<usize>>> {
    // Ensure both routes have enough elements.
    if offspring[r_i].len() < 3 || offspring[r_j].len() < 3 {
        return None;
    }
    for i in 1..offspring[r_i].len() - 1 {
        for j in 1..offspring[r_j].len() - 1 {
            let mut new_offspring = offspring.clone();
            if r_i == r_j {
                // For the same route, use in-place swap to avoid shifting issues.
                new_offspring[r_i].swap(i, j);
            } else {
                // For different routes, remove and insert as before.
                let mut route_i = new_offspring[r_i].clone();
                let mut route_j = new_offspring[r_j].clone();
                let patient_i = route_i.remove(i);
                let patient_j = route_j.remove(j);
                route_i.insert(i, patient_j);
                route_j.insert(j, patient_i);
                new_offspring[r_i] = route_i;
                new_offspring[r_j] = route_j;
            }
            let new_cost = fitness(&new_offspring, instance);
            let key = format!("{:?}", new_offspring);
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

// (Optional) If you still want to use relocations, make sure to adjust index shifting
// when moving elements within the same route.
fn try_all_relocations(
    r_i: usize,
    r_j: usize,
    offspring: &Vec<Vec<usize>>,
    instance: &Instance,
    cache: &DashMap<String, f64>,
) -> Option<Vec<Vec<usize>>> {
    if offspring[r_i].len() < 3 || offspring[r_j].len() < 3 {
        return None;
    }
    for i in 1..offspring[r_i].len() - 1 {
        for j in 1..offspring[r_j].len() - 1 {
            let mut new_offspring = offspring.clone();
            if r_i == r_j {
                // When relocating within the same route, adjust insertion index.
                let mut route = new_offspring[r_i].clone();
                let patient = route.remove(i);
                let insert_index = if i < j { j - 1 } else { j };
                route.insert(insert_index, patient);
                new_offspring[r_i] = route;
            } else {
                let mut route_i = new_offspring[r_i].clone();
                let mut route_j = new_offspring[r_j].clone();
                let patient = route_i.remove(i);
                route_j.insert(j, patient);
                new_offspring[r_i] = route_i;
                new_offspring[r_j] = route_j;
            }
            let new_cost = fitness(&new_offspring, instance);
            let key = format!("{:?}", new_offspring);
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
