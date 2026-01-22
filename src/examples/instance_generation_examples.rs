/// Example usage of the instance generator
/// 
/// This file demonstrates how to generate random instances with different configurations.
/// You can run specific examples by uncommenting them in the main function.

use crate::utils::instance_generator::{
    generate_random_instance, 
    generate_train_like_instance, 
    generate_and_save_instance,
    save_instance_to_json,
    plot_instance_map,
    InstanceConfig
};

#[allow(dead_code)]
/// Example 1: Generate a random instance with default configuration
pub fn example_default_instance() {
    println!("\n=== Example 1: Default Configuration ===");
    
    let mut instance = generate_random_instance(InstanceConfig::default());
    instance.add_nurses();
    
    println!("Generated instance: {}", instance.instance_name);
    println!("Number of nurses: {}", instance.nbr_nurses);
    println!("Nurse capacity: {}", instance.capacity_nurse);
    println!("Number of patients: {}", instance.patients.len());
    println!("Depot return time: {}", instance.depot.return_time);
    
    // Save to JSON
    let _ = save_instance_to_json(&instance, "output/random_instances/default_instance.json");
    
    // Plot the instance map
    plot_instance_map(&instance, "output/random_instances/default_instance_map.png");
}

#[allow(dead_code)]
/// Example 2: Generate a custom instance with specific ranges
pub fn example_custom_instance() {
    println!("\n=== Example 2: Custom Configuration ===");
    
    let custom_config = InstanceConfig {
        instance_name: "custom_nurse_routing_instance".to_string(),
        nbr_nurses_range: (15, 20),
        capacity_nurse_range: (180, 220),
        depot_return_time_range: (1100.0, 1300.0),
        depot_x_range: (38.0, 42.0),
        depot_y_range: (48.0, 52.0),
        nbr_patients_range: (60, 80),
        patient_x_range: (0.0, 100.0),
        patient_y_range: (0.0, 100.0),
        patient_demand_range: (5.0, 50.0),
        patient_care_time_range: (70.0, 110.0),
        time_window_start_range: (0.0, 900.0),
        time_window_duration_range: (100.0, 1300.0),
        benchmark: Some(500.0),
    };
    
    let mut instance = generate_random_instance(custom_config);
    instance.add_nurses();
    
    println!("Generated instance: {}", instance.instance_name);
    println!("Number of nurses: {}", instance.nbr_nurses);
    println!("Nurse capacity: {}", instance.capacity_nurse);
    println!("Number of patients: {}", instance.patients.len());
    
    // Save and visualize
    let _ = save_instance_to_json(&instance, "output/random_instances/custom_instance.json");
    plot_instance_map(&instance, "output/random_instances/custom_instance_map.png");
}

#[allow(dead_code)]
/// Example 3: Generate train-like instances with variations
pub fn example_train_variants() {
    println!("\n=== Example 3: Train-like Variants ===");
    
    let variations = vec![0.1, 0.2, 0.3]; // 10%, 20%, 30% variation
    
    for (idx, variation) in variations.iter().enumerate() {
        let mut instance = generate_train_like_instance(*variation);
        instance.add_nurses();
        
        println!("\nVariant {} ({}% variation):", idx + 1, (variation * 100.0) as i32);
        println!("  Nurses: {}", instance.nbr_nurses);
        println!("  Capacity: {}", instance.capacity_nurse);
        println!("  Patients: {}", instance.patients.len());
        println!("  Depot return: {}", instance.depot.return_time);
        println!("  Benchmark: {}", instance.benchmark);
        
        let json_path = format!("output/random_instances/train_variant_{}.json", idx + 1);
        let image_path = format!("output/random_instances/train_variant_{}_map.png", idx + 1);
        
        let _ = save_instance_to_json(&instance, &json_path);
        plot_instance_map(&instance, &image_path);
    }
}

#[allow(dead_code)]
/// Example 4: Generate and save instance in one call with visualization
pub fn example_generate_and_save() {
    println!("\n=== Example 4: Generate and Save with Visualization ===");
    
    let config = InstanceConfig {
        instance_name: "quick_instance".to_string(),
        nbr_nurses_range: (25, 25),  // Exactly 25 nurses
        capacity_nurse_range: (200, 200),  // Exactly 200 capacity
        depot_return_time_range: (1236.0, 1236.0),
        depot_x_range: (40.0, 40.0),
        depot_y_range: (50.0, 50.0),
        nbr_patients_range: (100, 100),  // Exactly 100 patients
        patient_x_range: (0.0, 100.0),
        patient_y_range: (0.0, 100.0),
        patient_demand_range: (10.0, 40.0),
        patient_care_time_range: (90.0, 90.0),
        time_window_start_range: (0.0, 800.0),
        time_window_duration_range: (150.0, 1220.0),
        benchmark: Some(827.3),
    };
    
    match generate_and_save_instance(
        config,
        "output/random_instances/quick_instance.json",
        true,  // Enable visualization
        Some("output/random_instances/quick_instance_map.png"),
    ) {
        Ok(instance) => {
            println!("Successfully generated and saved instance!");
            println!("Instance: {}", instance.instance_name);
            println!("Nurses: {}, Capacity: {}", instance.nbr_nurses, instance.capacity_nurse);
            println!("Patients: {}", instance.patients.len());
        }
        Err(e) => {
            eprintln!("Error generating instance: {}", e);
        }
    }
}

#[allow(dead_code)]
/// Example 5: Generate multiple instances for batch testing
pub fn example_batch_generation() {
    println!("\n=== Example 5: Batch Generation ===");
    
    let batch_size = 5;
    
    for i in 1..=batch_size {
        let config = InstanceConfig {
            instance_name: format!("batch_instance_{}", i),
            nbr_nurses_range: (20, 30),
            capacity_nurse_range: (150, 250),
            depot_return_time_range: (1000.0, 1500.0),
            depot_x_range: (30.0, 50.0),
            depot_y_range: (40.0, 60.0),
            nbr_patients_range: (50, 100),
            patient_x_range: (0.0, 100.0),
            patient_y_range: (0.0, 100.0),
            patient_demand_range: (10.0, 40.0),
            patient_care_time_range: (60.0, 120.0),
            time_window_start_range: (0.0, 800.0),
            time_window_duration_range: (100.0, 1200.0),
            benchmark: None,
        };
        
        let mut instance = generate_random_instance(config);
        instance.add_nurses();
        
        println!("Generated batch instance {}/{}: {} patients, {} nurses", 
                 i, batch_size, instance.patients.len(), instance.nbr_nurses);
        
        let json_path = format!("output/random_instances/batch/instance_{}.json", i);
        let _ = save_instance_to_json(&instance, &json_path);
    }
    
    println!("\nBatch generation complete!");
}

#[allow(dead_code)]
/// Run all examples
pub fn run_all_examples() {
    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output/random_instances/batch").ok();
    
    example_default_instance();
    example_custom_instance();
    example_train_variants();
    example_generate_and_save();
    example_batch_generation();
}

// Uncomment this if you want to run examples directly
// fn main() {
//     run_all_examples();
// }
