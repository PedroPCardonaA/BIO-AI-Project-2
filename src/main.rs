
use std::time::Instant;

use ea_components::evolutionary_algorithm::{evolutionary_algorithm, evolutionary_algorithm_crowding};
use utils::{plot_map::plot_map, textual_answer::save_textual_solution_to_file};
mod structs;
mod utils;
mod ea_components;

fn main() {
    let instance = utils::parse_data::parse_data("src/data/train/train_9.json");
    
    let start_time = Instant::now();
    let best_solution = evolutionary_algorithm_crowding(
            &instance,
            111,
            300,
            5,
            0.2,
            1.2,
            25,
            3,
            75,
        );

    // Calculates the elapsed time since the timer started.
    let duration = start_time.elapsed();

    // Converts the duration into seconds as a f64.
    let secs = duration.as_secs_f64();

    // Prints the elapsed time with 4 digits after the decimal point.
    println!("Evolutionary algorithm training completed in: {:.4} seconds", secs);

    plot_map(&best_solution, &instance.patients, &instance.depot);
    save_textual_solution_to_file("output/solution.txt", &best_solution, &instance);

    let _ = utils::create_file::save_solution_to_file(&best_solution, "output/solution.json");
}

