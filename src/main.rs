use rand::seq::IndexedRandom;
use structs::instance::Instance;
use ea_components::{crossover::route_preserving_crossover, generate_population::generate_population_heuristic_with_workload, mutation::mutate_relocate_patient, selection::tournament_selection};
use utils::plot_map::plot_map;
use std::{collections::HashMap, sync::{Arc, Mutex, RwLock}, thread};
use utils::plot_metrics::plot_fitness;
use std::time::Instant;

mod structs;
mod utils;
mod ea_components;

fn main() {
    let instance = utils::parse_data::parse_data("src/data/train/train_0.json");
    
    let start_time = Instant::now();
    let best_solution = evolutionary_algorithm(
            &instance,
            100,
            10000,
            5,
            0.7,
            1.2,
            1000,
            20
        );

    // Calculates the elapsed time since the timer started.
    let duration = start_time.elapsed();

    // Converts the duration into seconds as a f64.
    let secs = duration.as_secs_f64();

    // Prints the elapsed time with 4 digits after the decimal point.
    println!("Evolutionary algorithm training completed in: {:.4} seconds", secs);

    plot_map(&best_solution, &instance.patients, &instance.depot);
    let _ = utils::create_file::save_solution_to_file(&best_solution, "output/solution.json");
}



fn fitness(solution: &Vec<Vec<usize>>, instance: &Instance) -> f64 {
    let mut total_travel_time = 0.0;
    let mut total_penalty = 0.0;
    let penalty_factor = 1.0; // Higher value means higher penalty
    let penalty_factor_time = 10.0; // Higher value means higher penalty
    let penalty_factor_violation = 100.0; // Higher value means higher penalty

    // Calculate the total travel time for each nurse
    let mut nurses = instance.nurses.clone();
    for (nurse, route) in nurses.iter_mut().zip(solution.iter()) {
        let mut last_patient = 0; // The depot is the first patient

        //println!("New nurse");

        // Calculate the travel time and capacity for each patient in the route
        for patient_id in route {
            let mut wait_time = 0.0;
            let patient = &instance.patients[&patient_id.to_string()];
            // Print the nurse and patient ID
            //println!("Last patient: {:?}, Current patient: {:?}", last_patient, patient_id);

            // Calculate the travel time from the last patient to the current patient
            let travel_time = instance.travel_times[last_patient][*patient_id];

            // Add the travel time as a penalty, since then nurses will be penalized for traveling too much between patients
            total_penalty += travel_time * penalty_factor_time;

            // Check if the nurse visits the patient too early
            if patient.start_time > (nurse.get_current_travel_time() + travel_time) {
                wait_time = patient.start_time - (nurse.get_current_travel_time() + travel_time);
            }

            // Add the travel time to the nurse's current travel time
            nurse.set_current_travel_time(nurse.get_current_travel_time() + travel_time + patient.care_time + wait_time);

            // Check if the nurse visits the patient too late
            if patient.end_time < nurse.get_current_travel_time() {
                total_penalty += penalty_factor_violation * (nurse.get_current_travel_time() - patient.end_time);
            }

            // Add the patient's demand to the nurse's current load
            nurse.set_current_load(nurse.get_current_load() + patient.demand as u32);

            // Set the current patient as the last patient
            last_patient = *patient_id;
        }

        // Add the travel time from the last patient to the depot
        let travel_time = instance.travel_times[last_patient][0];
        nurse.set_current_travel_time(nurse.get_current_travel_time() + travel_time);

        // Check if the nurses capacity is exceeded
        if nurse.get_current_load() as f64 > nurse.get_capacity() as f64 {
            total_penalty += penalty_factor_violation * (nurse.get_current_load() as f64 - nurse.get_capacity() as f64);
        }

        // Check if the nurse returns to the depot too late
        if nurse.get_current_travel_time() > instance.depot.return_time {
            total_penalty += penalty_factor * (nurse.get_current_travel_time() - instance.depot.return_time);
        }

        // Add the nurse's travel time to the total travel time
        total_travel_time += nurse.get_current_travel_time();
    }

    total_travel_time + total_penalty
}

/// Structure to hold the result from an island.
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
) -> Vec<Vec<usize>> {
    // Parameters for island model
    let migration_interval = 50;
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
                generate_population_heuristic_with_workload(sub_population_size, &instance);
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
                            let mut rng = rand::thread_rng();
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
                    mutate_relocate_patient(&mut child1, mutation_probability);
                    mutate_relocate_patient(&mut child2, mutation_probability);
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

