use std::collections::HashMap;
use std::fs;
use std::time::Instant;
use serde::Serialize;

use crate::ea_components::fitness::fitness;
use crate::utils::create_file::{cleanup_current_best_folder, save_best_solution_files, save_current_solution_files, save_json};
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

/// Scans the current_best folder and updates the best_solutions map with the lowest stored objective value.
///
/// # Parameters
/// - `instance_name`: The name of the instance.
/// - `current_best_dir`: The path to the instance's current_best folder.
/// - `instance`: A reference to the parsed instance.
/// - `best_solutions`: A mutable reference to the map storing the best solution for each instance.
///
/// # Remarks
/// This function reads file names in the format "best_solution_<score>.json" and updates the in‑memory best
/// solution if a lower objective value is found.
fn update_best_solution_from_folder(
    instance_name: &str,
    current_best_dir: &str,
    instance: &Instance,
    best_solutions: &mut HashMap<String, (f64, Vec<Vec<usize>>, Instance)>,
) {
    if let Ok(entries) = fs::read_dir(current_best_dir) {
        for entry in entries.flatten() {
            if let Ok(file_name) = entry.file_name().into_string() {
                if let Some(score_str) = file_name.strip_prefix("best_solution_")
                    .and_then(|s| s.strip_suffix(".json"))
                {
                    if let Ok(parsed_score) = score_str.parse::<u32>() {
                        let stored_obj_value = parsed_score as f64 / 100.0;
                        best_solutions
                            .entry(instance_name.to_string())
                            .and_modify(|(curr_best, _sol, _inst)| {
                                if stored_obj_value < *curr_best {
                                    *curr_best = stored_obj_value;
                                }
                            })
                            .or_insert((stored_obj_value, Vec::new(), instance.clone()));
                    }
                }
            }
        }
    }
}

/// Continuously runs the evolutionary algorithm on the provided benchmark instances.
///
/// For each instance, the function:
/// 1. Parses the instance from a JSON file.
/// 2. Cleans the instance's "current_best" folder (once per outer loop) by removing any file whose
///    stored objective value is more than 10% above the benchmark value.
/// 3. Runs the evolutionary algorithm and times its execution.
/// 4. Saves the current run's solution to JSON, TXT, and PNG files (overwriting previous current run files).
/// 5. Evaluates the solution's objective value using the fitness function.
/// 6. If a new best solution is found (i.e. its objective value is lower than the stored best), it:
///    - Cleans the current_best folder again (removing any file whose stored value is above benchmark * 1.10),
///    - Updates the in‑memory best solution, and
///    - Saves the new best solution with a unique file name.
/// 7. Updates and saves the overall scoreboard with the current run's scores.
/// 8. Repeats the process indefinitely.
///
/// # Parameters
/// - `alg`: The evolutionary algorithm function to use. It should have the signature:
///   `fn(&Instance, usize, usize, usize, f64, f64, usize, usize, usize) -> Vec<Vec<usize>>`.
/// - `benchmarks`: A vector of tuples `(instance_name, benchmark)`.
/// - `population_size`: The population size for the algorithm.
/// - `generations`: The number of generations to run.
/// - `tournament_size`: The number of individuals in each tournament for selection.
/// - `mutation_probability`: The probability of mutation.
/// - `lambda`: A scaling parameter used by the algorithm.
/// - `generation_to_print`: The frequency (in generations) for printing progress.
/// - `num_islands`: The number of islands (subpopulations) used by the algorithm.
/// - `migration_interval`: The interval (in generations) for migration.
///
/// # Remarks
/// The function continuously trains the evolutionary algorithm on the provided instances until manually stopped.
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
    // Persistent map for each instance's best solution.
    let mut best_solutions: HashMap<String, (f64, Vec<Vec<usize>>, Instance)> = HashMap::new();

    

    // Endless loop: the training process repeats indefinitely.
    loop {
        // --- One-Time Cleanup per Outer Loop Iteration ---
        // For each instance, clean up its current_best folder.
        for (instance_name, benchmark) in &benchmarks {
            let current_best_dir = format!("output/scoring_test/{}/current_best", instance_name);
            cleanup_current_best_folder(&current_best_dir, *benchmark);
        }
        // --- End One-Time Cleanup ---

        // Reinitialize scoreboard data.
        let mut scores: Vec<ScoreEntry> = Vec::new();
        let mut total_score = 0.0;
        let max_possible = benchmarks.len() as f64 * 8.33;

        // Process each benchmark instance.
        for (instance_name, benchmark) in benchmarks.iter() {
            // Parse the instance from its JSON file.
            let file_path = format!("src/data/test/{}.json", instance_name);
            println!("Processing instance: {}", instance_name);
            let instance = parse_data(&file_path);

            // Create output directories.
            let output_dir = format!("output/scoring_test/{}", instance_name);
            let current_best_dir = format!("{}/current_best", output_dir);
            fs::create_dir_all(&output_dir).expect("Unable to create output directory");

            // Update in-memory best solution info from the current_best folder.
            update_best_solution_from_folder(instance_name, &current_best_dir, &instance, &mut best_solutions);

            // Run the evolutionary algorithm and time its execution.
            let start_time = Instant::now();
            let current_solution = alg(
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

            // Save the current solution files.
            save_current_solution_files(&output_dir, &current_solution, &instance);

            // Evaluate the objective value of the current solution.
            let obj_value = fitness(&current_solution, &instance);
            println!("Objective value for {}: {:.2}", instance_name, obj_value);

            // If a new best solution is found, update and save it.
            let best_entry = best_solutions
                .entry(instance_name.to_string())
                .or_insert((f64::INFINITY, Vec::new(), instance.clone()));
            if obj_value < best_entry.0 {
                // Clean up any bad solutions in the current_best folder.
                cleanup_current_best_folder(&current_best_dir, *benchmark);

                // Update the best solution in memory.
                *best_entry = (obj_value, current_solution.clone(), instance.clone());
                fs::create_dir_all(&current_best_dir).expect("Unable to create current_best directory");
                save_best_solution_files(&current_best_dir, obj_value, &current_solution, &instance);
            }

            // Compute the percentage difference and score for the instance.
            let percent_diff = ((obj_value - benchmark) / benchmark) * 100.0;
            let instance_score = if obj_value <= benchmark * 1.05 {
                8.33
            } else if obj_value <= benchmark * 1.10 {
                6.5
            } else if obj_value <= benchmark * 1.20 {
                4.0
            } else if obj_value <= benchmark * 1.30 {
                2.0
            } else {
                0.0
            };
            total_score += instance_score;

            // Create a score entry for the current instance.
            scores.push(ScoreEntry {
                instance_name: instance_name.to_string(),
                benchmark: *benchmark,
                objective_value: obj_value,
                score: instance_score,
                percent_difference: percent_diff,
            });

            // Update and save the overall scoreboard.
            let average_score = total_score / scores.len() as f64;
            let scoreboard = ScoreBoard {
                scores: scores.clone(),
                average_score,
                total_score,
                max_possible_score: max_possible,
            };
            match save_json(&scoreboard, "output/scoring_test/scoreboard.json") {
                Ok(_) => println!("Scoreboard updated ({} instances processed).", scores.len()),
                Err(e) => eprintln!("Error updating scoreboard: {}", e),
            }
        }
        println!("Completed one full iteration over all instances. Starting next iteration...");
    }
}