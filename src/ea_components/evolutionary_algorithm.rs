use std::{collections::HashMap, sync::{Arc, Mutex, RwLock}, thread};

use rand::seq::{IndexedRandom, IteratorRandom};

use crate::{structs::instance::Instance, utils::plot_metrics::plot_fitness};

use super::{crossover::{route_preserving_crossover, select_delete_fix_crossover}, fitness::fitness, generate_population::{generate_population_combined, generate_population_heuristic_with_workload}, mutation::{mutate_local_improvement, mutate_relocate_patient}, niching::fitness_sharing_adjustment, selection::tournament_selection};


pub struct IslandResult {
    pub best_solution: Vec<Vec<usize>>,
    pub fitness_history: Vec<f64>,
}

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
    // Each island gets its own subpopulation.
    let sub_population_size = population_size / num_islands;

    // Wrap instance in an Arc so that it can be shared across threads.
    let instance_arc = Arc::new(instance.clone());

    // Shared fitness cache: maps a (stringified) solution to its fitness value.
    let fitness_cache: Arc<RwLock<HashMap<String, f64>>> = Arc::new(RwLock::new(HashMap::new()));
    // Shared migration pool (for islands to deposit their best individuals).
    let migration_pool: Arc<Mutex<Vec<Vec<Vec<usize>>>>> = Arc::new(Mutex::new(Vec::new()));

    // Launch one thread per island.
    let mut handles = Vec::new();
    for island_id in 0..num_islands {
        let instance = instance_arc.clone();
        let fitness_cache = fitness_cache.clone();
        let migration_pool = migration_pool.clone();
        let handle = thread::spawn(move || {
            // Generate an initial subpopulation for this island.
            let mut sub_population =
                generate_population_combined(sub_population_size, &instance);
            let mut fitness_values: Vec<f64> = sub_population
                .iter()
                .map(|individual| {
                    // Create a unique key for the solution.
                    let key = format!("{:?}", individual);
                    {
                        let cache_read = fitness_cache.read().unwrap();
                        if let Some(&cached_fit) = cache_read.get(&key) {
                            return cached_fit;
                        }
                    }
                    let fit = fitness(individual, &instance);
                    let mut cache_write = fitness_cache.write().unwrap();
                    cache_write.insert(key, fit);
                    fit
                })
                .collect();

            // Record the best fitness per generation.
            let mut fitness_history = Vec::new();

            // Main loop for this island.
            for gen in 0..generations {
                // Every migration_interval generations, perform migration.
                if gen % migration_interval == 0 && gen > 0 {
                    // Deposit the best individual of this island in the shared migration pool.
                    let best_index = fitness_values
                        .iter()
                        .enumerate()
                        .min_by(|(_, &fit_a), (_, &fit_b)| {
                            fit_a.partial_cmp(&fit_b).unwrap()
                        })
                        .unwrap()
                        .0;
                    let best_individual = sub_population[best_index].clone();
                    {
                        let mut pool = migration_pool.lock().unwrap();
                        pool.push(best_individual);
                    }
                    // Then, if there is any migrant available, replace our worst individual.
                    {
                        let pool = migration_pool.lock().unwrap();
                        if !pool.is_empty() {
                            // Find the worst individual in the island.
                            let worst_index = fitness_values
                                .iter()
                                .enumerate()
                                .max_by(|(_, &fit_a), (_, &fit_b)| {
                                    fit_a.partial_cmp(&fit_b).unwrap()
                                })
                                .unwrap()
                                .0;
                            // Choose a random migrant from the pool.
                            let mut rng = rand::rng();
                            if let Some(migrant) = pool.choose(&mut rng) {
                                sub_population[worst_index] = migrant.clone();
                                // Recalculate fitness for the replaced solution using the cache.
                                let key = format!("{:?}", sub_population[worst_index]);
                                let new_fit = {
                                    let cache_read = fitness_cache.read().unwrap();
                                    if let Some(&cached_fit) = cache_read.get(&key) {
                                        cached_fit
                                    } else {
                                        drop(cache_read);
                                        let fit = fitness(&sub_population[worst_index], &instance);
                                        let mut cache_write = fitness_cache.write().unwrap();
                                        cache_write.insert(key, fit);
                                        fit
                                    }
                                };
                                fitness_values[worst_index] = new_fit;
                            }
                        }
                    }
                }

                // Generate a new population for the island.
                let mut new_population = Vec::with_capacity(sub_population_size);
                // Elitism: carry over the best individual.
                let best_index = fitness_values
                    .iter()
                    .enumerate()
                    .min_by(|(_, &fit_a), (_, &fit_b)| {
                        fit_a.partial_cmp(&fit_b).unwrap()
                    })
                    .unwrap()
                    .0;
                new_population.push(sub_population[best_index].clone());

                // Generate offspring until the subpopulation is filled.
                while new_population.len() < sub_population_size {
                    // Selection: tournament selection.
                    let parent1 =
                        tournament_selection(&sub_population, &fitness_values, tournament_size);
                    let parent2 =
                        tournament_selection(&sub_population, &fitness_values, tournament_size);
                    // Crossover: route-preserving crossover.
                    let (mut child1, mut child2) =
                        select_delete_fix_crossover(&parent1, &parent2, &instance, 0.2);
                    // Mutation: relocate a patient.
                    mutate_local_improvement(&mut child1, 0.5, &instance);
                    mutate_local_improvement(&mut child2, 0.5, &instance);
                    new_population.push(child1);
                    if new_population.len() < sub_population_size {
                        new_population.push(child2);
                    }
                }
                sub_population = new_population;
                // Recalculate fitness values for the new generation.
                fitness_values = sub_population
                    .iter()
                    .map(|individual| {
                        let key = format!("{:?}", individual);
                        {
                            let cache_read = fitness_cache.read().unwrap();
                            if let Some(&cached_fit) = cache_read.get(&key) {
                                return cached_fit;
                            }
                        }
                        let fit = fitness(individual, &instance);
                        let mut cache_write = fitness_cache.write().unwrap();
                        cache_write.insert(key, fit);
                        fit
                    })
                    .collect();

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
                // Records the best fitness of this generation.
                let best_fit = fitness_values
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                fitness_history.push(best_fit);
            }

            // Returns the best solution from this island along with its fitness history.
            let best_index = fitness_values
                .iter()
                .enumerate()
                .min_by(|(_, &fit_a), (_, &fit_b)| fit_a.partial_cmp(&fit_b).unwrap())
                .unwrap()
                .0;
            IslandResult {
                best_solution: sub_population[best_index].clone(),
                fitness_history,
            }
        });
        handles.push(handle);
    }

    // Waits for all island threads to finish and select the overall best solution.
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


fn distance(ind1: &Vec<Vec<usize>>, ind2: &Vec<Vec<usize>>) -> usize {
    let mut diff = 0;
    // Compare corresponding routes.
    for (route1, route2) in ind1.iter().zip(ind2.iter()) {
        // Count differences elementwise.
        let paired_diff = route1.iter().zip(route2.iter()).filter(|(a, b)| a != b).count();
        diff += paired_diff;
        // Also account for length differences.
        diff += if route1.len() > route2.len() {
            route1.len() - route2.len()
        } else {
            route2.len() - route1.len()
        };
    }
    // If the outer lists are of different lengths, add that difference.
    diff += if ind1.len() > ind2.len() {
        ind1.len() - ind2.len()
    } else {
        ind2.len() - ind1.len()
    };
    diff
}

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
    // Each island gets its own subpopulation.
    let sub_population_size = population_size / num_islands;

    // Wrap instance in an Arc so that it can be shared across threads.
    let instance_arc = Arc::new(instance.clone());

    // Shared fitness cache: maps a (stringified) solution to its fitness value.
    let fitness_cache: Arc<RwLock<HashMap<String, f64>>> = Arc::new(RwLock::new(HashMap::new()));
    // Shared migration pool (for islands to deposit their best individuals).
    let migration_pool: Arc<Mutex<Vec<Vec<Vec<usize>>>>> = Arc::new(Mutex::new(Vec::new()));

    // Launch one thread per island.
    let mut handles = Vec::new();
    for island_id in 0..num_islands {
        let instance = instance_arc.clone();
        let fitness_cache = fitness_cache.clone();
        let migration_pool = migration_pool.clone();
        let handle = thread::spawn(move || {
            // Generate an initial subpopulation for this island.
            let mut sub_population =
                generate_population_combined(sub_population_size, &instance);
            let mut fitness_values: Vec<f64> = sub_population
                .iter()
                .map(|individual| {
                    let key = format!("{:?}", individual);
                    {
                        let cache_read = fitness_cache.read().unwrap();
                        if let Some(&cached_fit) = cache_read.get(&key) {
                            return cached_fit;
                        }
                    }
                    let fit = fitness(individual, &instance);
                    let mut cache_write = fitness_cache.write().unwrap();
                    cache_write.insert(key, fit);
                    fit
                })
                .collect();

            // Closure to get the fitness of an individual using the cache.
            let get_fitness = |ind: &Vec<Vec<usize>>| -> f64 {
                let key = format!("{:?}", ind);
                {
                    let cache_read = fitness_cache.read().unwrap();
                    if let Some(&cached_fit) = cache_read.get(&key) {
                        return cached_fit;
                    }
                }
                let fit = fitness(ind, &instance);
                let mut cache_write = fitness_cache.write().unwrap();
                cache_write.insert(key, fit);
                fit
            };

            // Record the best fitness per generation.
            let mut fitness_history = Vec::new();

            // Main loop for this island.
            for gen in 0..generations {
                // Migration: every migration_interval generations, perform migration.
                if gen % migration_interval == 0 && gen > 0 {
                    // Deposit the best individual of this island in the shared migration pool.
                    let best_index = fitness_values
                        .iter()
                        .enumerate()
                        .min_by(|(_, &fit_a), (_, &fit_b)| {
                            fit_a.partial_cmp(&fit_b).unwrap()
                        })
                        .unwrap()
                        .0;
                    let best_individual = sub_population[best_index].clone();
                    {
                        let mut pool = migration_pool.lock().unwrap();
                        pool.push(best_individual);
                    }
                    // Then, if there is any migrant available, replace our worst individual.
                    {
                        let mut pool = migration_pool.lock().unwrap();
                        if !pool.is_empty() {
                            let worst_index = fitness_values
                                .iter()
                                .enumerate()
                                .max_by(|(_, &fit_a), (_, &fit_b)| {
                                    fit_a.partial_cmp(&fit_b).unwrap()
                                })
                                .unwrap()
                                .0;
                            let mut rng = rand::thread_rng();
                            if let Some(migrant) = pool.iter().choose(&mut rng) {
                                sub_population[worst_index] = migrant.clone();
                                // Recalculate fitness using the cache.
                                let key = format!("{:?}", sub_population[worst_index]);
                                let new_fit = {
                                    let cache_read = fitness_cache.read().unwrap();
                                    if let Some(&cached_fit) = cache_read.get(&key) {
                                        cached_fit
                                    } else {
                                        drop(cache_read);
                                        let fit = fitness(&sub_population[worst_index], &instance);
                                        let mut cache_write = fitness_cache.write().unwrap();
                                        cache_write.insert(key, fit);
                                        fit
                                    }
                                };
                                fitness_values[worst_index] = new_fit;
                            }
                        }
                    }
                }

                // Create a new population using deterministic crowding (no elitism).
                let mut new_population = Vec::with_capacity(sub_population_size);
                // Fill new_population until we reach the desired size.
                while new_population.len() < sub_population_size {
                    // Selection: choose two parents.
                    let parent1 =
                        tournament_selection(&sub_population, &fitness_values, tournament_size);
                    let parent2 =
                        tournament_selection(&sub_population, &fitness_values, tournament_size);
                    // Crossover: route-preserving (or select-delete-fix) crossover.
                    let (mut child1, mut child2) =
                        select_delete_fix_crossover(&parent1, &parent2, &instance, 0.2);
                    // Mutation.
                    mutate_local_improvement(&mut child1, 0.5, &instance);
                    mutate_local_improvement(&mut child2, 0.5, &instance);

                    // Determine the best pairing based on similarity.
                    let pairing1 = distance(&parent1, &child1) + distance(&parent2, &child2);
                    let pairing2 = distance(&parent1, &child2) + distance(&parent2, &child1);
                    if pairing1 <= pairing2 {
                        // For pairing1, each offspring competes with its corresponding parent.
                        let winner1 = if get_fitness(&child1) < get_fitness(&parent1) {
                            child1
                        } else {
                            parent1
                        };
                        let winner2 = if get_fitness(&child2) < get_fitness(&parent2) {
                            child2
                        } else {
                            parent2
                        };
                        new_population.push(winner1);
                        if new_population.len() < sub_population_size {
                            new_population.push(winner2);
                        }
                    } else {
                        // For pairing2, swap the comparisons.
                        let winner1 = if get_fitness(&child2) < get_fitness(&parent1) {
                            child2
                        } else {
                            parent1
                        };
                        let winner2 = if get_fitness(&child1) < get_fitness(&parent2) {
                            child1
                        } else {
                            parent2
                        };
                        new_population.push(winner1);
                        if new_population.len() < sub_population_size {
                            new_population.push(winner2);
                        }
                    }
                }
                sub_population = new_population;
                // Recalculate fitness values for the new generation.
                fitness_values = sub_population
                    .iter()
                    .map(|individual| {
                        let key = format!("{:?}", individual);
                        {
                            let cache_read = fitness_cache.read().unwrap();
                            if let Some(&cached_fit) = cache_read.get(&key) {
                                return cached_fit;
                            }
                        }
                        let fit = fitness(individual, &instance);
                        let mut cache_write = fitness_cache.write().unwrap();
                        cache_write.insert(key, fit);
                        fit
                    })
                    .collect();

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
                // Record the best fitness of this generation.
                let best_fit = fitness_values
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                fitness_history.push(best_fit);
            }

            // Return the best solution from this island along with its fitness history.
            let best_index = fitness_values
                .iter()
                .enumerate()
                .min_by(|(_, &fit_a), (_, &fit_b)| fit_a.partial_cmp(&fit_b).unwrap())
                .unwrap()
                .0;
            IslandResult {
                best_solution: sub_population[best_index].clone(),
                fitness_history,
            }
        });
        handles.push(handle);
    }

    // Wait for all island threads to finish and select the overall best solution.
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