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

/// Runs the provided evolutionary algorithm on each train_x.json file, evaluates the result,
/// saves a JSON, an image (png) and a textual representation of the solution, and iteratively updates a scoreboard.
/// 
/// The evolutionary algorithm function is provided as a parameter, allowing for modularity.
/// The function should have the signature:
/// `fn(&Instance, usize, usize, usize, f64, f64, usize, usize, usize) -> Vec<Vec<usize>>`.
/// 
/// # Parameters:
/// - `alg`: The evolutionary algorithm function to use.
/// - `population_size`: The size of the population used in the evolutionary algorithm.
/// - `generations`: The number of generations the algorithm will run for.
/// - `tournament_size`: The number of individuals in each tournament for selection.
/// - `mutation_probability`: The probability of mutation being applied.
/// - `lambda`: A parameter used for scaling.
/// - `generation_to_print`: How frequently progress is printed (in generations).
/// - `num_islands`: The number of islands (subpopulations) used in the algorithm.
/// - `migration_interval`: The interval (in generations) at which migration between islands occurs.
pub fn run_all_trains<F>(
    alg: F,
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
    let benchmarks = vec![
        ("train_0", 827.0),
        ("train_1", 589.0),
        ("train_2", 1258.0),
        ("train_3", 1132.0),
        ("train_4", 1261.0),
        ("train_5", 1092.0),
        ("train_6", 924.0),
        ("train_7", 870.0),
        ("train_8", 731.0),
        ("train_9", 855.0),
    ];

    let mut scores: Vec<ScoreEntry> = Vec::new();
    let mut total_score = 0.0;
    let max_possible = benchmarks.len() as f64 * 8.33;
    let iteration_started_time = Instant::now();

    // Process each benchmark instance.
    for (idx, (instance_name, benchmark)) in benchmarks.iter().enumerate() {
        let file_path = format!("src/data/train/{}.json", instance_name);
        println!("Processing instance: {}", instance_name);

        let instance = parse_data(&file_path);

        // STEP 1: Run the evolutionary algorithm and time the execution.
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

        // STEP 2: Create output directories for the instance.
        let output_dir = format!("output/scoring/{}", instance_name);
        std::fs::create_dir_all(&output_dir).expect("Unable to create output directory");

        let sol_file_json = format!("{}/solution_{}.json", output_dir, idx);
        let sol_file_txt = format!("{}/solution_{}.txt", output_dir, idx);
        let sol_file_png = format!("{}/solution_{}.png", output_dir, idx);

        // STEP 3: Save the best solution in JSON, TXT and PNG. 
        match save_solution_to_file(&best_solution, &sol_file_json) {
            Ok(_) => println!("Solution for {} saved to {}", instance_name, sol_file_json),
            Err(e) => eprintln!("Error saving solution for {}: {}", instance_name, e),
        }
        save_textual_solution_to_file(&sol_file_txt, &best_solution, &instance);

        plot_map_with_path(&best_solution, &instance.patients, &instance.depot, &sol_file_png);

        // STEP 4: Evaluate the objective value using the fitness function.
        let obj_value = fitness(&best_solution, &instance);

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

        // STEP 5: Create a score entry for this instance.
        let entry = ScoreEntry {
            instance_name: instance_name.to_string(),
            benchmark: *benchmark,
            objective_value: obj_value,
            score: instance_score,
            percent_difference: percent_diff,
        };
        scores.push(entry);

        // STEP 6: Update the overall scoreboard.
        let average_score = total_score / scores.len() as f64;
        let scoreboard = ScoreBoard {
            scores: scores.clone(),
            average_score,
            total_score,
            max_possible_score: max_possible,
        };

        let duration_of_all = iteration_started_time.elapsed();
        println!(
            "All instances processed in {:.4} seconds",
            duration_of_all.as_secs_f64()
        );

        match save_json(&scoreboard, "output/scoring/scoreboard.json") {
            Ok(_) => println!("Scoreboard updated ({} instances processed).", scores.len()),
            Err(e) => eprintln!("Error updating scoreboard: {}", e),
        }
    }
}