
use std::time::Instant;

use ea_components::evolutionary_algorithm::{evolutionary_algorithm, evolutionary_algorithm_crowding, evolutionary_algorithm_crowding_one};
use utils::{
    plot_map::plot_map, 
    textual_answer::save_textual_solution_to_file, 
    create_file::save_solution_to_file, 
    score_recorder::run_all_trains
};
mod structs;
mod utils;
mod ea_components;

fn main() {
 

    let instance = utils::parse_data::parse_data("src/data/train/train_2.json");
    
    let start_time = Instant::now();
    let best_solution = evolutionary_algorithm_crowding_one(
            &instance,
            30,
            50000,
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

    plot_map(&best_solution, &instance.patients, &instance.depot);
    save_textual_solution_to_file("output/solution.txt", &best_solution, &instance);

    let _ = save_solution_to_file(&best_solution, "output/solution.json");
    /* 
    // Run all training instances using the crowding algorithm.
    //TODO: Uncomment this line to run all training instances
    run_all_trains( 
        evolutionary_algorithm_crowding_one,
        30,
            50000,
            5,
            0.2,
            1.2,
            2000,
            1,
            10000,   // migration_interval
    );
*/
}

