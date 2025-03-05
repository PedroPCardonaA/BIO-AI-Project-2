use std::time::Instant;
use serde::Serialize;

use crate::ea_components::fitness::fitness;
use crate::utils::create_file::{save_json, save_solution_to_file};
use crate::utils::textual_answer::save_textual_solution_to_file;
use crate::utils::plot_map::plot_map_with_path;
use crate::utils::parse_data::parse_data;
use crate::structs::instance::Instance;

/// Represents the score for one training instance.
/// 
/// This struct stores scoring information for a single training instance, including the instance name,
/// benchmark objective value, computed objective value, assigned score based on the difference between the two,
/// and the percentage difference from the benchmark.
/// 
/// # Fields
/// - `instance_name`: The name of the instance (e.g., "train_0").
/// - `benchmark`: The benchmark objective value for the instance.
/// - `objective_value`: The objective value (total duration) computed by the solution.
/// - `score`: The score assigned based on the difference between the computed value and the benchmark.
/// - `percent_difference`: The percentage difference between the computed objective value and the benchmark.
#[derive(Serialize, Clone)]
struct ScoreEntry {
    /// The name of the instance (e.g., "train_0").
    instance_name: String,
    /// The benchmark objective value for the instance.
    benchmark: f64,
    /// The objective value (total duration) computed by the solution.
    objective_value: f64,
    /// The score assigned based on the difference between the computed value and the benchmark.
    score: f64,
    /// The percentage difference between the computed objective value and the benchmark.
    percent_difference: f64,
}

/// Represents the overall scoreboard for multiple training instances.
/// 
/// This struct aggregates the scoring data for all processed training instances, providing a list of
/// individual score entries along with summary metrics such as the average score, total score, and the maximum possible score.
/// 
/// # Fields
/// - `scores`: A vector of individual instance score entries.
/// - `average_score`: The average score over all processed instances.
/// - `total_score`: The sum of scores for all instances.
/// - `max_possible_score`: The maximum possible score (e.g., if all instances achieved full score).
#[derive(Serialize)]
struct ScoreBoard {
    /// A list of individual instance score entries.
    scores: Vec<ScoreEntry>,
    /// The average score over all processed instances.
    average_score: f64,
    /// The sum of scores for all instances.
    total_score: f64,
    /// The maximum possible score (e.g., if all instances achieved full score).
    max_possible_score: f64,
}

/// Runs the provided evolutionary algorithm on a specified range of train_x.json files in an endless loop,
/// saving each run's solution (JSON, TXT, and PNG) to the instance-specific directory (e.g., output/scoring/train_0/),
/// and continually updating the best solution for each instance in a "current_best" subdirectory.
/// 
/// For each instance provided in the benchmarks parameter, the function:
/// 1. Parses the instance from a JSON file.
/// 2. Runs the evolutionary algorithm and times its execution.
/// 3. Saves the current run's solution to JSON, TXT, and PNG files in the instance's output directory,
///    overwriting previous files.
/// 4. Evaluates the solution's objective value using the fitness function.
/// 5. If the current solution is better than the best so far for that instance, updates the best solution
///    files in the "current_best" subdirectory (overwriting previous bests).
/// 6. Repeats the process indefinitely until the process is manually stopped.
/// 
/// # Parameters:
/// - `alg`: The evolutionary algorithm function to use. It should have the signature:
///   `fn(&Instance, usize, usize, usize, f64, f64, usize, usize, usize) -> Vec<Vec<usize>>`.
/// - `benchmarks`: A vector of tuples `(instance_name, benchmark)` for the instances to process.
/// - `population_size`: The population size used in the evolutionary algorithm.
/// - `generations`: The number of generations the algorithm will run for.
/// - `tournament_size`: The number of individuals in each tournament for selection.
/// - `mutation_probability`: The probability of mutation being applied.
/// - `lambda`: A scaling parameter used by the algorithm.
/// - `generation_to_print`: How frequently (in generations) progress is printed.
/// - `num_islands`: The number of islands (subpopulations) used in the algorithm.
/// - `migration_interval`: The interval (in generations) at which migration between islands occurs.
pub fn run_trains_range<F>(
    alg: F,
    benchmarks: Vec<(&str, f64)>,
    population_size: usize,
    generations: usize,
    tournament_size: usize,
    mutation_probability: f64,
    lambda: f64,
    generation_to_print: usize,
    num_islands: usize,
    migration_interval: usize,
)
where
    F: Fn(
            &Instance,
            usize,
            usize,
            usize,
            f64,
            f64,
            usize,
            usize,
            usize,
        ) -> Vec<Vec<usize>>,
{
    use std::collections::HashMap;
    // Maintain a HashMap to store the best solution for each instance.
    let mut best_solutions: HashMap<String, (f64, Vec<Vec<usize>>, Instance)> = HashMap::new();

    // Endless loop: runs until the user stops the process.
    loop {
        // Process each benchmark instance provided.
        for (idx, (instance_name, benchmark)) in benchmarks.iter().enumerate() {
            // STEP: Build the file path for the instance JSON file.
            let file_path = format!("src/data/train/{}.json", instance_name);
            println!("Processing instance: {}", instance_name);

            let instance = parse_data(&file_path);

            // STEP 1: Run the evolutionary algorithm and time its execution.
            let start_time = Instant::now();
            let best_solution = alg(
                &instance,
                population_size,
                generations,
                tournament_size,
                mutation_probability,
                lambda,
                generation_to_print,
                num_islands,
                migration_interval,
            );
            let duration = start_time.elapsed();
            println!(
                "Instance {} solved in {:.4} seconds",
                instance_name,
                duration.as_secs_f64()
            );

            // STEP 2: Create the output directory for this instance.
            let output_dir = format!("output/scoring/{}", instance_name);
            std::fs::create_dir_all(&output_dir)
                .expect("Unable to create output directory");

            // STEP 3: Define file paths for the current run's solution (which will be overwritten on each run).
            let sol_file_json = format!("{}/solution.json", output_dir);
            let sol_file_txt = format!("{}/solution.txt", output_dir);
            let sol_file_png = format!("{}/solution.png", output_dir);

            // STEP 4: Save the current run's solution.
            match save_solution_to_file(&best_solution, &sol_file_json) {
                Ok(_) => println!("Solution for {} updated in {}", instance_name, sol_file_json),
                Err(e) => eprintln!("Error saving solution for {}: {}", instance_name, e),
            }
            save_textual_solution_to_file(&sol_file_txt, &best_solution, &instance);
            plot_map_with_path(&best_solution, &instance.patients, &instance.depot, &sol_file_png);

            // STEP 5: Evaluate the objective value using the fitness function.
            let obj_value = fitness(&best_solution, &instance);
            println!("Objective value for {}: {:.2}", instance_name, obj_value);

            // STEP: Update the best solution for this instance if the current solution is better.

            ///TODO: ADD SCOREBOARD|
            let entry = best_solutions
                .entry(instance_name.to_string())
                .or_insert((f64::INFINITY, Vec::new(), instance.clone()));
            if obj_value < entry.0 {
                *entry = (obj_value, best_solution.clone(), instance.clone());
                // STEP: Save the new best solution in the instance's "current_best" subdirectory.
                let current_best_dir = format!("{}/current_best", output_dir);
                std::fs::create_dir_all(&current_best_dir)
                    .expect("Unable to create current_best directory");
                let best_sol_file_json = format!("{}/best_solution.json", current_best_dir);
                let best_sol_file_txt = format!("{}/best_solution.txt", current_best_dir);
                let best_sol_file_png = format!("{}/best_solution.png", current_best_dir);
                match save_solution_to_file(&best_solution, &best_sol_file_json) {
                    Ok(_) => println!(
                        "Current best solution for {} updated in {}",
                        instance_name, best_sol_file_json
                    ),
                    Err(e) => eprintln!(
                        "Error saving current best solution for {}: {}",
                        instance_name, e
                    ),
                }
                save_textual_solution_to_file(&best_sol_file_txt, &best_solution, &instance);
                plot_map_with_path(&best_solution, &instance.patients, &instance.depot, &best_sol_file_png);
            }
        }
        println!("Completed one full iteration over all instances. Starting next iteration...");
    }
}
