use std::sync::{Arc, Mutex};
use std::thread;
use dashmap::DashMap;
use rand::Rng;
use crate::{
    structs::instance::Instance,
    utils::plot_metrics::plot_fitness,
};
use super::selection::tournament_selection_index;
use super::{
    crossover::meta_crossover,
    fitness::fitness,
    generate_population::generate_population_combined,
    mutation::meta_mutation,
    route_improvement::route_improvement,
};

/// Represents the outcome of the evolutionary process on an island.
///
/// This structure holds the best solution found by the island's evolutionary algorithm
/// and records the corresponding fitness values over the generations.
/// 
/// - `best_solution`: The best solution discovered, represented as a vector of routes, where each route is a vector of patient IDs.
/// - `fitness_history`: A record of the best fitness value per generation during the evolution.
pub struct IslandResult {
    pub best_solution: Vec<Vec<usize>>,
    pub fitness_history: Vec<f64>,
}

/// Computes a distance metric between two individuals by comparing their respective routes.
///
/// Each individual is represented as a vector of routes where each inner vector
/// denotes a route as a sequence of patients' identifiers. The function calculates the distance by:
/// - Iterating over paired routes from both individuals.
/// - Counting the number of mismatched elements in the corresponding positions.
/// - Adding the difference in lengths of the routes to account for any extra elements.
/// - Incorporating differences if the individuals have a different number of routes.
///
/// This metric is used in the crowding replacement process to assess the similarity between solutions.
///
/// # Arguments
///
/// * `ind1` - A reference to the first individual's set of routes.
/// * `ind2` - A reference to the second individual's set of routes.
///
/// # Returns
///
/// Returns an `usize` value representing the computed distance between the two individuals.
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

/// Steady-State Memetic Genetic Algorithm (SSMGA)
/// The final algorithm produced in project 2 in Bio-AI.
/// This algorithm is a memetic algorithm (being an extension of the evolutionary algorithm with a local search heuristic added)
///  as well as being a genetic algorithm which is run to perform evolution of the individuals on a steady-state basis. 
/// Executes a parallelized evolutionary algorithm with crowding replacement for solving the home-care routing problem.
///
/// The function implements a genetic algorithm that minimizes the total travel time for home-care nurses
/// by evolving a population of candidate solutions. It employs tournament selection, meta crossover, and meta mutation
/// operators, along with an adaptive crowding replacement strategy to choose between parents and offspring based on fitness.
/// Additionally, it periodically applies a diversity step by randomly replacing a portion of the worst individuals,
/// which helps prevent premature convergence. 
///
/// # Parameters
/// - `instance` - A reference to the problem instance containing depot, patient, nurse, and travel time data.
/// - `population_size` - The total number of candidate solutions in the population.
/// - `generations` - The number of generations (iterations) for which the population is evolved.
/// - `tournament_size` - The number of individuals used in tournament selection for choosing parents.
/// - `mutation_probability` - The probability of applying mutation to an offspring.
/// - `lambda` - A parameter that influences the behavior of the meta crossover operator (e.g., the crossover rate).
/// - `generation_to_print` - The frequency (in generations) at which the algorithm prints the best fitness value.
/// - `num_islands` - The number of parallel sub-populations to evolve (typically set to 1 for a single population).
/// - `migration_interval` - The interval, in generations, at which migration between sub-populations is performed (applicable when using multiple islands).
///
/// # Returns
/// Returns the best overall solution found as a vector of routes, where each route is represented as a vector
/// of patient IDs. Each route implicitly starts and ends at the depot, which is managed externally.
/// SSMGA
pub fn ssmga(
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

    // STEP 1: Initialize shared parameters and resources.
    let sub_population_size = population_size / num_islands;
    let instance_arc = Arc::new(instance.clone());
    let fitness_cache: Arc<DashMap<String, f64>> = Arc::new(DashMap::new());
    let migration_pool: Arc<Mutex<Vec<Vec<Vec<usize>>>>> =
        Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();
    for island_id in 0..num_islands {

        // STEP 2: Clone shared resources for use in the spawned thread.
        let instance = instance_arc.clone();
        let fitness_cache = fitness_cache.clone();
        let migration_pool = migration_pool.clone();
        let handle = thread::spawn(move || {

            // STEP 3: Generate the initial sub-population and compute initial fitness values.
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

            // STEP 4: Define helper closures for fitness evaluation and random individual generation.
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

            // STEP 5: Evolutionary loop over generations.
            for gen in 0..generations {

                // STEP 5.1: Migration Step - exchange best individual with worst from migration pool.
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

                // STEP 5.2: Diversity Step - periodically replace the worst 80% of individuals.
                if gen > 0 && generations >= 20 && gen % (generations / 20) == 0 {
                    // Determine the number of individuals to replace.
                    let to_replace = (sub_population_size as f64 * 0.80).ceil() as usize;
                    // Sort indices so that the worst individuals come first.
                    let mut indices: Vec<usize> = (0..sub_population_size).collect();
                    indices.sort_by(|&i, &j| {
                        fitness_values[j]
                            .partial_cmp(&fitness_values[i])
                            .unwrap()
                    });

                    // Replace the 'to_replace' worst individuals with random solutions
                    for k in 0..to_replace {
                        let idx = indices[k];
                        let new_sol = generate_random_individual();
                        sub_population[idx] = new_sol;
                        // Recompute fitness for the new solution.
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

                    // Log the outcome of the diversity step.
                    println!(
                        "Island {} Generation {}: Replaced {} worst individuals with random solutions.",
                        island_id, gen, to_replace
                    );
                }

                // STEP 5.3: Adaptive Parameter Calculation - update theta for crowding replacement.
                let theta = 1.0 - (gen as f64) / (generations as f64);

                // STEP 5.4: Parent Selection, Crossover, and Mutation.
                let (idx1, parent1) =
                    tournament_selection_index(&sub_population, &fitness_values, tournament_size);
                let (idx2, parent2) =
                    tournament_selection_index(&sub_population, &fitness_values, tournament_size);

                let (mut child1, mut child2) = meta_crossover(&parent1, &parent2, &instance, lambda);
                meta_mutation(&mut child1, mutation_probability, &instance);
                meta_mutation(&mut child2, mutation_probability, &instance);

                // STEP 5.5: Route Improvement - apply local improvements to offspring.
                route_improvement(&instance, &mut child1, &*fitness_cache);
                route_improvement(&instance, &mut child2, &*fitness_cache);

                // STEP 5.6: Crowding Replacement - determine which individuals to retain.
                let pairing1 = distance(&parent1, &child1) + distance(&parent2, &child2);
                let pairing2 = distance(&parent1, &child2) + distance(&parent2, &child1);
                let mut rng = rand::rng();

                let new_ind1;
                let new_ind2;
                if pairing1 <= pairing2 {
                    new_ind1 = if theta > 0.0 {
                        let diff = get_fitness(&child1) - get_fitness(&parent1);
                        let p = 1.0 / (1.0 + (diff / theta).exp());
                        if rng.random::<f64>() < p {
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
                        if rng.random::<f64>() < p {
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
                        if rng.random::<f64>() < p {
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
                        if rng.random::<f64>() < p {
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

                // STEP 5.7: Update Population - replace the selected individuals and update their fitness.
                sub_population[idx1] = new_ind1;
                sub_population[idx2] = new_ind2;
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

                // STEP 5.8: Logging - print the best fitness at specified generation intervals.
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

                // STEP 5.9: Record the current generation's best fitness.
                fitness_history.push(
                    fitness_values
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min),
                );
            }

            // STEP 6: Return the best solution and fitness history for this thread.
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

    // STEP 7: Aggregate results from all threads and determine the overall best solution.
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

    // STEP 8: Plot the fitness evolution and return the best overall solution.
    plot_fitness(&island_results);
    overall_best_solution.unwrap()
}
