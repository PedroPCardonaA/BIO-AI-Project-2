
use std::time::Instant;

use ea_components::evolutionary_algorithm::ssmga;
use utils::{
    plot_map::plot_map_with_path, 
    textual_answer::save_textual_solution_to_file, 
    create_file::save_solution_to_file, 
    score_recorder::run_trains_range
};
mod structs;
mod utils;
mod ea_components;

/// Main method.
/// 
/// This function performs the following steps:
/// 
/// 1. Parses a problem instance from a JSON file (e.g., "src/data/train/train_2.json") which contains all necessary
///    data for the nurse routing problem.
/// 2. Executes the `evolutionary_algorithm_crowding_one` function with fixed parameters to compute the best solution.
/// 3. Measures and prints the elapsed time for the evolutionary algorithm run.
/// 4. Plots the best solution on a map and saves the diagram as an image.
/// 5. Saves both a textual representation and a JSON representation of the best solution to output files.
/// 
/// Optionally, commented-out code is provided to run the evolutionary algorithm on multiple training instances
/// and iteratively update a scoreboard.
fn main() {
    /*
    let instance = utils::parse_data::parse_data("src/data/train/train_9.json");
    
    let start_time = Instant::now();
    let best_solution = evolutionary_algorithm_crowding_one(
            &instance,
            30,
            30000,
            5,
            0.2,
            1.2,
            1000,
            1,
            10000,
        );

    // Calculates the elapsed time since the timer started.
    let duration = start_time.elapsed();

    // Converts the duration into seconds as a f64.
    let secs = duration.as_secs_f64();

    // Prints the elapsed time with 4 digits after the decimal point.
    println!("Evolutionary algorithm training completed in: {:.4} seconds", secs);

    plot_map_with_path(&best_solution, &instance.patients, &instance.depot, "output/solution.png");
    save_textual_solution_to_file("output/solution.txt", &best_solution, &instance);

    let _ = save_solution_to_file(&best_solution, "output/solution.json");
    // Run all training instances using the crowding algorithm.
    //TODO: Uncomment this line to run all training instances
    
    */
    let benchmarks = vec![
        //("train_0", 827.0),
        //("train_1", 589.0),
        ("train_2", 1258.0),
        //("train_3", 1132.0),
        //("train_4", 1261.0),
        //("train_5", 1092.0),
        //("train_6", 924.0),
        //("train_7", 870.0),
        //("train_8", 731.0),
        ("train_9", 855.0),
    ];

    // Run the evolutionary algorithm (being the designed Steady-State Memetic Genetic Algorithm) on the specified benchmarks.
    run_trains_range(
        ssmga, 
        benchmarks,                                           // benchmark range of instances
        40,                                  // population_size
        15000,                                   // generations
        5,                                   // tournament_size
        0.2,                            // mutation_probability
        1.2,                                          // lambda
        2000,                            // generation_to_print
        1,                                       // num_islands
        10000                             // migration_interval
    );
}