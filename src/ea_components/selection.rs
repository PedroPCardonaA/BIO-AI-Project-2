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
