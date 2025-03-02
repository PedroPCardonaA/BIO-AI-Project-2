use std::sync::{Arc, Mutex};
use std::thread;
use dashmap::DashMap;
use rand::{seq::IteratorRandom, Rng};
use crate::{
    structs::instance::Instance,
    utils::plot_metrics::plot_fitness,
};
use super::route_improvement;
use super::selection::tournament_selection_index;
use super::{
    crossover::{meta_crossover, select_delete_fix_crossover},
    fitness::fitness,
    generate_population::generate_population_combined,
    mutation::meta_mutation,
    selection::tournament_selection,
    route_improvement::route_improvement,
};

pub struct IslandResult {
    pub best_solution: Vec<Vec<usize>>,
    pub fitness_history: Vec<f64>,
}

/// Helper function to compute the distance between two individuals.
fn distance(ind1: &Vec<Vec<usize>>, ind2: &Vec<Vec<usize>>) -> usize {
    let mut diff = 0;
    for (route1, route2) in ind1.iter().zip(ind2.iter()) {
        let paired_diff = route1.iter().zip(route2.iter()).filter(|(a, b)| a != b).count();
        diff += paired_diff;
        diff += if route1.len() > route2.len() {
            route1.len() - route2.len()
        } else {
            route2.len() - route1.len()
        };
    }
    diff += if ind1.len() > ind2.len() {
        ind1.len() - ind2.len()
    } else {
        ind2.len() - ind1.len()
    };
    diff
}

/// Evolutionary algorithm using migration with DashMap caching.
pub fn evolutionary_algorithm(
    instance: &Instance,
    population_size: usize,
    generations: usize,
    tournament_size: usize,
    mutation_probability: f64,
    lambda: f64,
    generation_to_print: usize,
    num_islands: usize,
    migration_interval: usize,
) -> Vec<Vec<usize>> {
    let sub_population_size = population_size / num_islands;
    let instance_arc = Arc::new(instance.clone());
    let fitness_cache: Arc<DashMap<String, f64>> = Arc::new(DashMap::new());
    let migration_pool: Arc<Mutex<Vec<Vec<Vec<usize>>>>> = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();
    for island_id in 0..num_islands {
        let instance = instance_arc.clone();
        let fitness_cache = fitness_cache.clone();
        let migration_pool = migration_pool.clone();
        let handle = thread::spawn(move || {
            let mut sub_population = generate_population_combined(sub_population_size, &instance);
            let mut fitness_values: Vec<f64> = sub_population.iter().map(|individual| {
                let key = format!("{:?}", individual);
                if let Some(val) = fitness_cache.get(&key) {
                    *val
                } else {
                    let fit = fitness(individual, &instance);
                    fitness_cache.insert(key, fit);
                    fit
                }
            }).collect();

            let get_fitness = |ind: &Vec<Vec<usize>>| -> f64 {
                let key = format!("{:?}", ind);
                if let Some(val) = fitness_cache.get(&key) {
                    *val
                } else {
                    let fit = fitness(ind, &instance);
                    fitness_cache.insert(key, fit);
                    fit
                }
            };

            let mut fitness_history = Vec::new();
            for gen in 0..generations {
                // Migration every migration_interval generations (after generation 0)
                if gen % migration_interval == 0 && gen > 0 {
                    // Deposit the best individual from this island.
                    let best_index = fitness_values
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;
                    let best_individual = sub_population[best_index].clone();

                    // Lock the migration pool once: deposit and choose a migrant.
                    let (worst_index_opt, best_migrant_opt) = {
                        let mut pool = migration_pool.lock().unwrap();
                        pool.push(best_individual);
                        if pool.is_empty() {
                            (None, None)
                        } else {
                            let worst_index = fitness_values
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .unwrap()
                                .0;
                            let best_migrant = pool.iter()
                                .min_by(|a, b| get_fitness(a).partial_cmp(&get_fitness(b)).unwrap())
                                .cloned();
                            (Some(worst_index), best_migrant)
                        }
                    };

                    if let (Some(worst_index), Some(best_migrant)) = (worst_index_opt, best_migrant_opt) {
                        sub_population[worst_index] = best_migrant;
                        let key = format!("{:?}", sub_population[worst_index]);
                        let new_fit = if let Some(val) = fitness_cache.get(&key) {
                            *val
                        } else {
                            let fit = fitness(&sub_population[worst_index], &instance);
                            fitness_cache.insert(key, fit);
                            fit
                        };
                        fitness_values[worst_index] = new_fit;
                    }
                }

                // Create a new population (elitism + variation)
                let mut new_population = Vec::with_capacity(sub_population_size);
                let best_index = fitness_values
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                new_population.push(sub_population[best_index].clone());

                while new_population.len() < sub_population_size {
                    let parent1 = tournament_selection(&sub_population, &fitness_values, tournament_size);
                    let parent2 = tournament_selection(&sub_population, &fitness_values, tournament_size);
                    let (mut child1, mut child2) = select_delete_fix_crossover(&parent1, &parent2, &instance, 0.2);
                    meta_mutation(&mut child1, 1.0, &instance);
                    meta_mutation(&mut child2, 1.0, &instance);
                    new_population.push(child1);
                    if new_population.len() < sub_population_size {
                        new_population.push(child2);
                    }
                }
                sub_population = new_population;
                fitness_values = sub_population.iter().map(|individual| {
                    let key = format!("{:?}", individual);
                    if let Some(val) = fitness_cache.get(&key) {
                        *val
                    } else {
                        let fit = fitness(individual, &instance);
                        fitness_cache.insert(key, fit);
                        fit
                    }
                }).collect();

                if gen % generation_to_print == 0 {
                    let best_fit = fitness_values.iter().cloned().fold(f64::INFINITY, f64::min);
                    println!("Island {} Generation {}: Best fitness = {}", island_id, gen, best_fit);
                }
                fitness_history.push(fitness_values.iter().cloned().fold(f64::INFINITY, f64::min));
            }

            IslandResult {
                best_solution: {
                    let best_index = fitness_values.iter().enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap().0;
                    sub_population[best_index].clone()
                },
                fitness_history,
            }
        });
        handles.push(handle);
    }

    let mut overall_best_solution = None;
    let mut best_fitness = f64::INFINITY;
    let mut island_results = Vec::new();
    for (island_id, handle) in handles.into_iter().enumerate() {
        let result: IslandResult = handle.join().unwrap();
        island_results.push((island_id, result.fitness_history.clone()));
        let sol_fit = fitness(&result.best_solution, instance);
        if sol_fit < best_fitness {
            best_fitness = sol_fit;
            overall_best_solution = Some(result.best_solution);
        }
    }

    plot_fitness(&island_results);
    overall_best_solution.unwrap()
}

/// Evolutionary algorithm with adaptive θ-crowding and migration (using DashMap)
pub fn evolutionary_algorithm_crowding(
    instance: &Instance,
    population_size: usize,
    generations: usize,
    tournament_size: usize,
    mutation_probability: f64,
    lambda: f64,
    generation_to_print: usize,
    num_islands: usize,
    migration_interval: usize,
) -> Vec<Vec<usize>> {
    let sub_population_size = population_size / num_islands;
    let instance_arc = Arc::new(instance.clone());
    let fitness_cache: Arc<DashMap<String, f64>> = Arc::new(DashMap::new());
    let migration_pool: Arc<Mutex<Vec<Vec<Vec<usize>>>>> = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();
    for island_id in 0..num_islands {
        let instance = instance_arc.clone();
        let fitness_cache = fitness_cache.clone();
        let migration_pool = migration_pool.clone();
        let handle = thread::spawn(move || {
            let mut sub_population = generate_population_combined(sub_population_size, &instance);
            let mut fitness_values: Vec<f64> = sub_population
                .iter()
                .map(|individual| {
                    let key = format!("{:?}", individual);
                    if let Some(val) = fitness_cache.get(&key) {
                        *val
                    } else {
                        let fit = fitness(individual, &instance);
                        fitness_cache.insert(key, fit);
                        fit
                    }
                })
                .collect();

            let get_fitness = |ind: &Vec<Vec<usize>>| -> f64 {
                let key = format!("{:?}", ind);
                if let Some(val) = fitness_cache.get(&key) {
                    *val
                } else {
                    let fit = fitness(ind, &instance);
                    fitness_cache.insert(key, fit);
                    fit
                }
            };

            let mut fitness_history = Vec::new();
            for gen in 0..generations {
                // ----- Migration block -----
                if gen % migration_interval == 0 && gen > 0 {
                    // Find best individual in the current island.
                    let best_index = fitness_values
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;
                    let best_individual = sub_population[best_index].clone();

                    let (worst_index_opt, best_migrant_opt) = {
                        let mut pool = migration_pool.lock().unwrap();
                        pool.push(best_individual);
                        // Choose the worst individual from the current island…
                        let worst_index = fitness_values
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap()
                            .0;
                        // …and pick the best migrant from the migration pool.
                        if let Some((migrant_idx, _)) = pool
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| get_fitness(a).partial_cmp(&get_fitness(b)).unwrap())
                        {
                            let best_migrant = pool.remove(migrant_idx);
                            (Some(worst_index), Some(best_migrant))
                        } else {
                            (None, None)
                        }
                    };

                    if let (Some(worst_index), Some(best_migrant)) = (worst_index_opt, best_migrant_opt) {
                        sub_population[worst_index] = best_migrant;
                        let key = format!("{:?}", sub_population[worst_index]);
                        let new_fit = if let Some(val) = fitness_cache.get(&key) {
                            *val
                        } else {
                            let fit = fitness(&sub_population[worst_index], &instance);
                            fitness_cache.insert(key, fit);
                            fit
                        };
                        fitness_values[worst_index] = new_fit;
                    }
                }
                // -----------------------------

                // Adaptive θ-crowding: using theta to decide offspring survival.
                let mut new_population = Vec::with_capacity(sub_population_size);
                let theta = 1.0 - (gen as f64) / (generations as f64);
                while new_population.len() < sub_population_size {
                    let parent1 = tournament_selection(&sub_population, &fitness_values, tournament_size);
                    let parent2 = tournament_selection(&sub_population, &fitness_values, tournament_size);
                    // Use provided parameters here:
                    let (mut child1, mut child2) = meta_crossover(&parent1, &parent2, &instance, 1.0);
                    meta_mutation(&mut child1, 0.8, &instance);
                    meta_mutation(&mut child2, 0.8, &instance);
                    
                    // Pass the cache reference (via &*fitness_cache) so that route_improvement works with a &DashMap.
                    route_improvement(&instance, &mut child1, &*fitness_cache);
                    route_improvement(&instance, &mut child2, &*fitness_cache);

                    let pairing1 = distance(&parent1, &child1) + distance(&parent2, &child2);
                    let pairing2 = distance(&parent1, &child2) + distance(&parent2, &child1);
                    let mut rng = rand::thread_rng();
                    if pairing1 <= pairing2 {
                        let winner1 = if theta > 0.0 {
                            let diff = get_fitness(&child1) - get_fitness(&parent1);
                            let p = 1.0 / (1.0 + (diff / theta).exp());
                            if rng.gen::<f64>() < p { child1 } else { parent1 }
                        } else {
                            if get_fitness(&child1) < get_fitness(&parent1) { child1 } else { parent1 }
                        };
                        let winner2 = if theta > 0.0 {
                            let diff = get_fitness(&child2) - get_fitness(&parent2);
                            let p = 1.0 / (1.0 + (diff / theta).exp());
                            if rng.gen::<f64>() < p { child2 } else { parent2 }
                        } else {
                            if get_fitness(&child2) < get_fitness(&parent2) { child2 } else { parent2 }
                        };
                        new_population.push(winner1);
                        if new_population.len() < sub_population_size {
                            new_population.push(winner2);
                        }
                    } else {
                        let winner1 = if theta > 0.0 {
                            let diff = get_fitness(&child2) - get_fitness(&parent1);
                            let p = 1.0 / (1.0 + (diff / theta).exp());
                            if rng.gen::<f64>() < p { child2 } else { parent1 }
                        } else {
                            if get_fitness(&child2) < get_fitness(&parent1) { child2 } else { parent1 }
                        };
                        let winner2 = if theta > 0.0 {
                            let diff = get_fitness(&child1) - get_fitness(&parent2);
                            let p = 1.0 / (1.0 + (diff / theta).exp());
                            if rng.gen::<f64>() < p { child1 } else { parent2 }
                        } else {
                            if get_fitness(&child1) < get_fitness(&parent2) { child1 } else { parent2 }
                        };
                        new_population.push(winner1);
                        if new_population.len() < sub_population_size {
                            new_population.push(winner2);
                        }
                    }
                }
                sub_population = new_population;
                fitness_values = sub_population
                    .iter()
                    .map(|individual| {
                        let key = format!("{:?}", individual);
                        if let Some(val) = fitness_cache.get(&key) {
                            *val
                        } else {
                            let fit = fitness(individual, &instance);
                            fitness_cache.insert(key, fit);
                            fit
                        }
                    })
                    .collect();

                if gen % generation_to_print == 0 {
                    let best_fit = fitness_values.iter().cloned().fold(f64::INFINITY, f64::min);
                    println!("Island {} Generation {}: Best fitness = {}", island_id, gen, best_fit);
                }
                fitness_history.push(fitness_values.iter().cloned().fold(f64::INFINITY, f64::min));
            }

            IslandResult {
                best_solution: {
                    let best_index = fitness_values
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;
                    sub_population[best_index].clone()
                },
                fitness_history,
            }
        });
        handles.push(handle);
    }

    let mut overall_best_solution = None;
    let mut best_fitness = f64::INFINITY;
    let mut island_results = Vec::new();
    for (island_id, handle) in handles.into_iter().enumerate() {
        let result: IslandResult = handle.join().unwrap();
        island_results.push((island_id, result.fitness_history.clone()));
        let sol_fit = fitness(&result.best_solution, instance);
        if sol_fit < best_fitness {
            best_fitness = sol_fit;
            overall_best_solution = Some(result.best_solution);
        }
    }

    plot_fitness(&island_results);
    overall_best_solution.unwrap()
}


pub fn evolutionary_algorithm_crowding_one(
    instance: &Instance,
    population_size: usize,
    generations: usize,
    tournament_size: usize,
    mutation_probability: f64,
    lambda: f64,
    generation_to_print: usize,
    num_islands: usize,
    migration_interval: usize,
) -> Vec<Vec<usize>> {
    let sub_population_size = population_size / num_islands;
    let instance_arc = Arc::new(instance.clone());
    let fitness_cache: Arc<DashMap<String, f64>> = Arc::new(DashMap::new());
    let migration_pool: Arc<Mutex<Vec<Vec<Vec<usize>>>>> =
        Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();
    for island_id in 0..num_islands {
        let instance = instance_arc.clone();
        let fitness_cache = fitness_cache.clone();
        let migration_pool = migration_pool.clone();
        let handle = thread::spawn(move || {
            // Generate the initial sub-population.
            let mut sub_population = generate_population_combined(sub_population_size, &instance);
            let mut fitness_values: Vec<f64> = sub_population
                .iter()
                .map(|individual| {
                    let key = format!("{:?}", individual);
                    if let Some(val) = fitness_cache.get(&key) {
                        *val
                    } else {
                        let fit = fitness(individual, &instance);
                        fitness_cache.insert(key, fit);
                        fit
                    }
                })
                .collect();

            // Helper closure to evaluate fitness with cache.
            let get_fitness = |ind: &Vec<Vec<usize>>| -> f64 {
                let key = format!("{:?}", ind);
                if let Some(val) = fitness_cache.get(&key) {
                    *val
                } else {
                    let fit = fitness(ind, &instance);
                    fitness_cache.insert(key, fit);
                    fit
                }
            };

            let generate_random_individual = ||{
                let mut pop  = generate_population_combined(1, &instance);
                pop.remove(0)
            };

            

            let mut fitness_history = Vec::new();


            for gen in 0..generations {
                // ----- Migration block (unchanged) -----
                if gen % migration_interval == 0 && gen > 0 {
                    // Find best individual in the current island.
                    let best_index = fitness_values
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;
                    let best_individual = sub_population[best_index].clone();

                    let (worst_index_opt, best_migrant_opt) = {
                        let mut pool = migration_pool.lock().unwrap();
                        pool.push(best_individual);
                        let worst_index = fitness_values
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap()
                            .0;
                        if let Some((migrant_idx, _)) = pool
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                get_fitness(a).partial_cmp(&get_fitness(b)).unwrap()
                            })
                        {
                            let best_migrant = pool.remove(migrant_idx);
                            (Some(worst_index), Some(best_migrant))
                        } else {
                            (None, None)
                        }
                    };

                    if let (Some(worst_index), Some(best_migrant)) =
                        (worst_index_opt, best_migrant_opt)
                    {
                        sub_population[worst_index] = best_migrant;
                        let key = format!("{:?}", sub_population[worst_index]);
                        let new_fit = if let Some(val) = fitness_cache.get(&key) {
                            *val
                        } else {
                            let fit = fitness(&sub_population[worst_index], &instance);
                            fitness_cache.insert(key, fit);
                            fit
                        };
                        fitness_values[worst_index] = new_fit;
                    }
                }

                                // ~~~~~~~~~~~~~~~~ DIVERSITY STEP ~~~~~~~~~~~~~~~~~ //
                // Every 5% of the total generations, randomize the worst 80%.
                // (Check that generations is at least 20, etc.)
                if gen > 0 && generations >= 20 && gen % (generations / 20) == 0 {
                    // How many individuals to replace
                    let to_replace = (sub_population_size as f64 * 0.80).ceil() as usize;
                    // Sort by fitness descending => worst first
                    let mut indices: Vec<usize> = (0..sub_population_size).collect();
                    indices.sort_by(|&i, &j| {
                        // compare fitness_values[j] with fitness_values[i] for descending
                        fitness_values[j]
                            .partial_cmp(&fitness_values[i])
                            .unwrap()
                    });

                    // Replace the 'to_replace' worst individuals with random solutions
                    for k in 0..to_replace {
                        let idx = indices[k];
                        let new_sol = generate_random_individual();
                        sub_population[idx] = new_sol;
                        // Recompute fitness
                        let key = format!("{:?}", sub_population[idx]);
                        let new_fit = if let Some(val) = fitness_cache.get(&key) {
                            *val
                        } else {
                            let fit = fitness(&sub_population[idx], &instance);
                            fitness_cache.insert(key, fit);
                            fit
                        };
                        fitness_values[idx] = new_fit;
                    }

                    // Print a message to see the effect
                    println!(
                        "Island {} Generation {}: Replaced {} worst individuals with random solutions.",
                        island_id, gen, to_replace
                    );
                }
                // ----------------------------------------

                // Adaptive theta for replacement.
                let theta = 1.0 - (gen as f64) / (generations as f64);

                // --- Selection, Crossover, and Mutation ---
                // Here we use a tournament selection that returns (index, individual).
                let (idx1, parent1) =
                    tournament_selection_index(&sub_population, &fitness_values, tournament_size);
                let (idx2, parent2) =
                    tournament_selection_index(&sub_population, &fitness_values, tournament_size);

                let (mut child1, mut child2) = meta_crossover(&parent1, &parent2, &instance, lambda);
                meta_mutation(&mut child1, mutation_probability, &instance);
                meta_mutation(&mut child2, mutation_probability, &instance);

                // Improve routes for each offspring.
                route_improvement(&instance, &mut child1, &*fitness_cache);
                route_improvement(&instance, &mut child2, &*fitness_cache);

                // --- Crowding Replacement ---
                // Compute the pairing distances.
                let pairing1 = distance(&parent1, &child1) + distance(&parent2, &child2);
                let pairing2 = distance(&parent1, &child2) + distance(&parent2, &child1);
                let mut rng = rand::thread_rng();

                let new_ind1;
                let new_ind2;
                if pairing1 <= pairing2 {
                    new_ind1 = if theta > 0.0 {
                        let diff = get_fitness(&child1) - get_fitness(&parent1);
                        let p = 1.0 / (1.0 + (diff / theta).exp());
                        if rng.gen::<f64>() < p {
                            child1
                        } else {
                            parent1
                        }
                    } else {
                        if get_fitness(&child1) < get_fitness(&parent1) {
                            child1
                        } else {
                            parent1
                        }
                    };
                    new_ind2 = if theta > 0.0 {
                        let diff = get_fitness(&child2) - get_fitness(&parent2);
                        let p = 1.0 / (1.0 + (diff / theta).exp());
                        if rng.gen::<f64>() < p {
                            child2
                        } else {
                            parent2
                        }
                    } else {
                        if get_fitness(&child2) < get_fitness(&parent2) {
                            child2
                        } else {
                            parent2
                        }
                    };
                } else {
                    new_ind1 = if theta > 0.0 {
                        let diff = get_fitness(&child2) - get_fitness(&parent1);
                        let p = 1.0 / (1.0 + (diff / theta).exp());
                        if rng.gen::<f64>() < p {
                            child2
                        } else {
                            parent1
                        }
                    } else {
                        if get_fitness(&child2) < get_fitness(&parent1) {
                            child2
                        } else {
                            parent1
                        }
                    };
                    new_ind2 = if theta > 0.0 {
                        let diff = get_fitness(&child1) - get_fitness(&parent2);
                        let p = 1.0 / (1.0 + (diff / theta).exp());
                        if rng.gen::<f64>() < p {
                            child1
                        } else {
                            parent2
                        }
                    } else {
                        if get_fitness(&child1) < get_fitness(&parent2) {
                            child1
                        } else {
                            parent2
                        }
                    };
                }
                // Replace the two selected individuals in the population.
                sub_population[idx1] = new_ind1;
                sub_population[idx2] = new_ind2;
                // Update fitness values accordingly.
                {
                    let key = format!("{:?}", sub_population[idx1]);
                    let new_fit = if let Some(val) = fitness_cache.get(&key) {
                        *val
                    } else {
                        let fit = fitness(&sub_population[idx1], &instance);
                        fitness_cache.insert(key, fit);
                        fit
                    };
                    fitness_values[idx1] = new_fit;
                }
                {
                    let key = format!("{:?}", sub_population[idx2]);
                    let new_fit = if let Some(val) = fitness_cache.get(&key) {
                        *val
                    } else {
                        let fit = fitness(&sub_population[idx2], &instance);
                        fitness_cache.insert(key, fit);
                        fit
                    };
                    fitness_values[idx2] = new_fit;
                }

                if gen % generation_to_print == 0 {
                    let best_fit = fitness_values
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min);
                    println!(
                        "Island {} Generation {}: Best fitness = {}",
                        island_id, gen, best_fit
                    );
                }
                fitness_history.push(
                    fitness_values
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min),
                );
            }

            IslandResult {
                best_solution: {
                    let best_index = fitness_values
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;
                    sub_population[best_index].clone()
                },
                fitness_history,
            }
        });
        handles.push(handle);
    }

    let mut overall_best_solution = None;
    let mut best_fitness = f64::INFINITY;
    let mut island_results = Vec::new();
    for (island_id, handle) in handles.into_iter().enumerate() {
        let result: IslandResult = handle.join().unwrap();
        island_results.push((island_id, result.fitness_history.clone()));
        let sol_fit = fitness(&result.best_solution, instance);
        if sol_fit < best_fitness {
            best_fitness = sol_fit;
            overall_best_solution = Some(result.best_solution);
        }
    }

    plot_fitness(&island_results);
    overall_best_solution.unwrap()
}
