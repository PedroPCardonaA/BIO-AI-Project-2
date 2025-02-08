use std::collections::HashSet;

/// Returns the set of edges in a solution. Each route produces edges from depot (0) to the first patient,
/// between consecutive patients, and from the last patient back to the depot.
fn get_edges(solution: &Vec<Vec<usize>>) -> HashSet<(usize, usize)> {
    let mut edges = HashSet::new();
    for route in solution {
        if !route.is_empty() {
            // Edge from depot to first patient.
            edges.insert((0, route[0]));
            // Edges between patients.
            for window in route.windows(2) {
                edges.insert((window[0], window[1]));
            }
            // Edge from last patient back to depot.
            edges.insert((route[route.len() - 1], 0));
        }
    }
    edges
}

/// Computes a normalized distance between two solutions based on their edge sets.
/// The distance is 0 if they share all edges and 1 if they share none.
fn edge_distance(sol1: &Vec<Vec<usize>>, sol2: &Vec<Vec<usize>>) -> f64 {
    let edges1 = get_edges(sol1);
    let edges2 = get_edges(sol2);
    let common = edges1.intersection(&edges2).count() as f64;
    let total = (edges1.len() + edges2.len()) as f64;
    if total == 0.0 {
        0.0
    } else {
        // This gives a value between 0 and 1.
        (total - 2.0 * common) / total
    }
}

/// Adjusts raw fitness values via fitness sharing.
/// For each individual, we compute a sharing sum over all individuals in the population.
/// The sharing function is defined as:
///   sh(d) = 1 - (d/sigma)^alpha, if d < sigma, and 0 otherwise.
/// The shared fitness is then raw_fitness / sharing_sum.
pub fn fitness_sharing_adjustment(
    population: &Vec<Vec<Vec<usize>>>, 
    raw_fitness: &Vec<f64>,
    sigma: f64,
    alpha: f64,
) -> Vec<f64> {
    let n = population.len();
    let mut shared_fitness = vec![0.0; n];
    for i in 0..n {
        let mut sharing_sum = 0.0;
        for j in 0..n {
            let d = edge_distance(&population[i], &population[j]);
            let sh = if d < sigma { 1.0 - (d / sigma).powf(alpha) } else { 0.0 };
            sharing_sum += sh;
        }
        // Avoid division by zero.
        shared_fitness[i] = raw_fitness[i] / if sharing_sum > 0.0 { sharing_sum } else { 1.0 };
    }
    shared_fitness
}
