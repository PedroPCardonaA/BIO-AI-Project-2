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

    // Randomly select indices for the tournament.
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

    // Return the best individual from the tournament.
    population[best_index.unwrap()].clone()
}


pub fn exponential_rank_wheel_selection(
    population: &Vec<Vec<Vec<usize>>>,
    fitness: &Vec<f64>,
    lambda: f64,
) -> Vec<Vec<usize>> {
    // Create a vector of indices and sort them by fitness (ascending).
    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.sort_by(|&a, &b| fitness[a].partial_cmp(&fitness[b]).unwrap());

    // Compute exponential weights based on rank (best individual has rank 0).
    // weight = exp(-lambda * rank)
    let weights: Vec<f64> = indices
        .iter()
        .enumerate()
        .map(|(rank, _)| (-lambda * (rank as f64)).exp())
        .collect();

    // Calculate total weight.
    let total_weight: f64 = weights.iter().sum();

    // Build cumulative weights for the roulette wheel.
    let mut cumulative_weights = Vec::with_capacity(weights.len());
    let mut cumulative = 0.0;
    for w in &weights {
        cumulative += *w;
        cumulative_weights.push(cumulative);
    }

    // Generate a random number in [0, total_weight).
    let mut rng = rand::rng();
    let r: f64 = rng.random_range(0.0..total_weight);

    // Find the first rank where the cumulative weight exceeds r.
    let selected_rank = cumulative_weights
        .iter()
        .position(|&cw| cw >= r)
        .unwrap();
    let selected_index = indices[selected_rank];

    // Return the selected individual.
    population[selected_index].clone()
}
