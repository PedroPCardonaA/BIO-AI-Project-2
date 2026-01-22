
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
mod examples;

/// Test function to validate a solution
#[allow(dead_code)]
fn test_solution_validator() {
    use utils::parse_data::parse_data;
    use utils::solution_validator::validate_and_print;
    use std::fs;
    
    println!("\n=== Testing Solution Validator ===\n");
    
    // Load the instance
    let instance = parse_data("src/data/test/test_1.json");
    
    // Load the solution from JSON
    let solution_json = fs::read_to_string("output/scoring_test/test_1/solution.json")
        .expect("Failed to read solution file");
    let solution: Vec<Vec<usize>> = serde_json::from_str(&solution_json)
        .expect("Failed to parse solution JSON");
    
    println!("Validating solution for test_1...\n");
    
    // Validate and print result
    let is_feasible = validate_and_print(&solution, &instance);
    
    if is_feasible {
        println!("\nThe solution satisfies all constraints!");
    } else {
        println!("\nThe solution violates some constraints. See details above.");
    }
    
    println!("\n=== Validation Complete ===\n");
}

/// Test function to demonstrate instance generator
#[allow(dead_code)]
fn test_instance_generator() {
    use utils::instance_generator::{generate_and_save_instance, InstanceConfig};
    
    println!("\n=== Generating 5 Random Instances ===\n");
    
    // Create output directory
    std::fs::create_dir_all("output/random_instances").ok();
    
    for i in 1..=5 {
        println!("{}. Generating random instance {}...", i, i);
        let config = InstanceConfig {
            instance_name: format!("random_instance_{}", i),
            ..Default::default()
        };
        
        match generate_and_save_instance(
            config,
            &format!("output/random_instances/random_instance_{}.json", i),
            true,
            Some(&format!("output/random_instances/random_instance_{}_map.png", i)),
        ) {
            Ok(instance) => {
                println!("   ✓ Generated: {} nurses, {} patients, capacity: {}", 
                         instance.nbr_nurses, instance.patients.len(), instance.capacity_nurse);
            }
            Err(e) => eprintln!("   ✗ Error: {}", e),
        }
    }
    
    println!("\n=== Instance Generation Complete ===");
    println!("Generated 5 instances in output/random_instances/");
    println!("- random_instance_1.json (with map)");
    println!("- random_instance_2.json (with map)");
    println!("- random_instance_3.json (with map)");
    println!("- random_instance_4.json (with map)");
    println!("- random_instance_5.json (with map)\n");
}

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
    // Optimize the new random instances with integer values
    let new_instances = vec![
        ("random_instance_1", 0.0),
        ("random_instance_2", 0.0),
        ("random_instance_3", 0.0),
        ("random_instance_4", 0.0),
        ("random_instance_5", 0.0),
    ];

    // Run the evolutionary algorithm on the new instances
    run_trains_range(
        ssmga, 
        new_instances,              // new random instances
        30,                         // population_size
        100000,                     // generations
        6,                          // tournament_size
        0.2,                        // mutation_probability
        1.2,                        // lambda
        1000,                       // generation_to_print
        1,                          // num_islands
        10000                       // migration_interval
    );
} 