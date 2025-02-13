use std::{collections::HashMap, sync::{Arc, Mutex, RwLock}, thread};

use rand::seq::{IndexedRandom, IteratorRandom};

use crate::{structs::instance::Instance, utils::plot_metrics::plot_fitness};

use super::{crossover::route_preserving_crossover, fitness::fitness, generate_population::{generate_population_combined, generate_population_heuristic_with_workload}, mutation::{mutate_local_improvement, mutate_relocate_patient}, niching::fitness_sharing_adjustment, selection::tournament_selection};


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
                        route_preserving_crossover(&parent1, &parent2, &instance);
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

pub fn evolutionary_algorithm_niching(
    instance: &Instance,
    population_size: usize,
    generations: usize,
    tournament_size: usize,
    mutation_probability: f64,
    lambda: f64,
    generation_to_print: usize,
    num_islands: usize,
) -> Vec<Vec<usize>> {
    // Migration occurs every 50 generations.
    let migration_interval = 50;
    // Each island gets a subpopulation.
    let sub_population_size = population_size / num_islands;

    let instance_arc = Arc::new(instance.clone());
    let fitness_cache: Arc<RwLock<HashMap<String, f64>>> = Arc::new(RwLock::new(HashMap::new()));
    // Shared migration pool for islands.
    let migration_pool: Arc<Mutex<Vec<Vec<Vec<usize>>>>> = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();
    for island_id in 0..num_islands {
        let instance = instance_arc.clone();
        let fitness_cache = fitness_cache.clone();
        let migration_pool = migration_pool.clone();
        let handle = thread::spawn(move || {
            // Generate an initial subpopulation.
            let mut sub_population =
                generate_population_heuristic_with_workload(sub_population_size, &instance);
            // Parameters for fitness sharing.
            let sigma = 0.3;  // Adjust niche radius as needed.
            let alpha = 1.0;  // Adjust shape parameter as needed.

            // Compute initial raw fitness values.
            let raw_fitness: Vec<f64> = sub_population
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
            // Compute initial shared (niche-adjusted) fitness.
            let mut shared_fitness = fitness_sharing_adjustment(&sub_population, &raw_fitness, sigma, alpha);

            // To record the best raw fitness per generation.
            let mut fitness_history = Vec::new();

            for gen in 0..generations {
                // Migration: every migration_interval generations (except generation 0).
                if gen % migration_interval == 0 && gen > 0 {
                    // Deposit best individual from this island into the shared migration pool.
                    let best_index = raw_fitness
                        .iter()
                        .enumerate()
                        .min_by(|(_, &fit_a), (_, &fit_b)| fit_a.partial_cmp(&fit_b).unwrap())
                        .unwrap()
                        .0;
                    let best_individual = sub_population[best_index].clone();
                    {
                        let mut pool = migration_pool.lock().unwrap();
                        pool.push(best_individual);
                    }
                    // If any migrant is available, replace the worst individual.
                    {
                        let pool = migration_pool.lock().unwrap();
                        if !pool.is_empty() {
                            let worst_index = raw_fitness
                                .iter()
                                .enumerate()
                                .max_by(|(_, &fit_a), (_, &fit_b)| fit_a.partial_cmp(&fit_b).unwrap())
                                .unwrap()
                                .0;
                            let mut rng = rand::rng();
                            if let Some(migrant) = pool.iter().choose(&mut rng) {
                                sub_population[worst_index] = migrant.clone();
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
                                // (The raw fitness for this individual will be updated below.)
                            }
                        }
                    }
                }

                // Recalculate raw fitness values for the current population.
                let raw_fitness: Vec<f64> = sub_population
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
                // Adjust fitness using fitness sharing.
                let shared_fitness = fitness_sharing_adjustment(&sub_population, &raw_fitness, sigma, alpha);

                // Record the best (raw) fitness for this generation.
                let best_fit = raw_fitness.iter().cloned().fold(f64::INFINITY, f64::min);
                fitness_history.push(best_fit);

                if gen % generation_to_print == 0 {
                    println!(
                        "Island {} Generation {}: Best raw fitness = {}",
                        island_id, gen, best_fit
                    );
                }

                // Create new subpopulation.
                let mut new_population = Vec::with_capacity(sub_population_size);
                // Elitism: keep the best individual (using the shared fitness values).
                let best_index = shared_fitness
                    .iter()
                    .enumerate()
                    .min_by(|(_, &fit_a), (_, &fit_b)| fit_a.partial_cmp(&fit_b).unwrap())
                    .unwrap()
                    .0;
                new_population.push(sub_population[best_index].clone());

                // Generate offspring until the subpopulation is filled.
                while new_population.len() < sub_population_size {
                    // Use tournament selection based on the niche-adjusted fitness.
                    let parent1 =
                        tournament_selection(&sub_population, &shared_fitness, tournament_size);
                    let parent2 =
                        tournament_selection(&sub_population, &shared_fitness, tournament_size);
                    // Crossover (route-preserving).
                    let (mut child1, mut child2) =
                        route_preserving_crossover(&parent1, &parent2, &instance);
                    // Mutation.
                    mutate_relocate_patient(&mut child1, mutation_probability);
                    mutate_relocate_patient(&mut child2, mutation_probability);
                    new_population.push(child1);
                    if new_population.len() < sub_population_size {
                        new_population.push(child2);
                    }
                }
                sub_population = new_population;
            }

            // After all generations, return the best solution from this island.
            let raw_fitness: Vec<f64> = sub_population
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
            let best_index = raw_fitness
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

    // Wait for all islands to complete and pick the overall best solution.
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
