use rand::Rng;

/// Selects one individual from the population using tournament selection.
///
/// # Arguments
///
/// * `population` - A reference to the population (each individual is a Vec<Vec<usize>>).
/// * `fitness` - A reference to a vector of fitness values corresponding to each individual.
///               Lower fitness is considered better.
/// * `tournament_size` - The number of individuals in each tournament.
///
/// # Returns
///
/// A clone of the selected individual.
pub fn tournament_selection(
    population: &Vec<Vec<Vec<usize>>>,
    fitness: &Vec<f64>,
    tournament_size: usize,
) -> Vec<Vec<usize>> {
    let mut rng = rand::rng();
    let pop_size = population.len();
    let mut best_index = None;
    for _ in 0..tournament_size {
        let idx = rng.random_range(0..pop_size);
        best_index = match best_index {
            Some(current_best) => {
                // Lower fitness is better.
                if fitness[idx] < fitness[current_best] {
                    Some(idx)
                } else {
                    Some(current_best)
                }
            }
            None => Some(idx),
        };
    }
    population[best_index.unwrap()].clone()
}

/// Performs tournament selection on the population.
/// 
/// This function randomly selects a subset of individuals from the population (of size `tournament_size`)
/// and returns the individual with the lowest fitness value, as lower fitness is considered better.
/// The function returns a tuple containing the index of the selected individual and a clone of its chromosome.
/// 
/// # Parameters
/// - `population` - A reference to the population, where each individual is represented as a vector of routes.
/// - `fitness` - A vector containing the fitness values corresponding to each individual in the population.
/// - `tournament_size` - The number of random individuals to consider during the tournament selection.
/// 
/// # Returns
/// A tuple containing:
/// - The index of the selected individual.
/// - A clone of the selected individual's chromosome (a vector of routes).
pub fn tournament_selection_index(
    population: &Vec<Vec<Vec<usize>>>,
    fitness: &Vec<f64>,
    tournament_size: usize,
) -> (usize, Vec<Vec<usize>>) {
    // STEP 1: Initialize the random number generator and determine the population size.
    let mut rng = rand::rng();
    let pop_size = population.len();
    
    // STEP 2: Initialize best_index to track the index of the best individual found.
    let mut best_index = None;
    
    // STEP 3: Iterate for the specified tournament size, randomly selecting individuals.
    for _ in 0..tournament_size {
        let idx = rng.random_range(0..pop_size);
        best_index = match best_index {
            Some(current_best) => {
                // Lower fitness is better.
                if fitness[idx] < fitness[current_best] {
                    Some(idx)
                } else {
                    Some(current_best)
                }
            }
            None => Some(idx),
        };
    }
    
    // STEP 4: Return the index and a clone of the selected individual's chromosome.
    (best_index.unwrap(), population[best_index.unwrap()].clone())
}
