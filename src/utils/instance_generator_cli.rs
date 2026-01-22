/// Command-line interface for the instance generator
/// 
/// Add this to your main.rs to enable CLI generation of instances

use std::env;
use crate::utils::instance_generator::{
    generate_random_instance,
    generate_train_like_instance,
    generate_and_save_instance,
    InstanceConfig,
};

#[allow(dead_code)]
pub fn run_cli() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return;
    }
    
    match args[1].as_str() {
        "default" => generate_default_instance(&args),
        "train" => generate_train_variant(&args),
        "custom" => generate_custom_instance(&args),
        "batch" => generate_batch_instances(&args),
        "help" => print_usage(),
        _ => {
            println!("Unknown command: {}", args[1]);
            print_usage();
        }
    }
}

fn generate_default_instance(args: &[String]) {
    let output = if args.len() > 2 {
        &args[2]
    } else {
        "output/random_instances/default_instance.json"
    };
    
    println!("Generating default instance...");
    
    match generate_and_save_instance(
        InstanceConfig::default(),
        output,
        true,
        Some(&format!("{}.png", output.trim_end_matches(".json"))),
    ) {
        Ok(instance) => {
            println!("✓ Successfully generated instance!");
            println!("  Nurses: {}", instance.nbr_nurses);
            println!("  Capacity: {}", instance.capacity_nurse);
            println!("  Patients: {}", instance.patients.len());
            println!("  Saved to: {}", output);
        }
        Err(e) => eprintln!("✗ Error: {}", e),
    }
}

fn generate_train_variant(args: &[String]) {
    let variation = if args.len() > 2 {
        args[2].parse::<f64>().unwrap_or(0.2)
    } else {
        0.2
    };
    
    let output = if args.len() > 3 {
        &args[3]
    } else {
        "output/random_instances/train_variant.json"
    };
    
    println!("Generating train-like variant with {}% variation...", (variation * 100.0) as i32);
    
    let mut instance = generate_train_like_instance(variation);
    instance.add_nurses();
    
    match crate::utils::instance_generator::save_instance_to_json(&instance, output) {
        Ok(_) => {
            println!("✓ Successfully generated instance!");
            println!("  Nurses: {}", instance.nbr_nurses);
            println!("  Capacity: {}", instance.capacity_nurse);
            println!("  Patients: {}", instance.patients.len());
            println!("  Saved to: {}", output);
            
            crate::utils::instance_generator::plot_instance_map(
                &instance,
                &format!("{}.png", output.trim_end_matches(".json"))
            );
        }
        Err(e) => eprintln!("✗ Error: {}", e),
    }
}

fn generate_custom_instance(args: &[String]) {
    if args.len() < 6 {
        println!("Usage: cargo run -- custom <nurses> <capacity> <patients> <output>");
        println!("Example: cargo run -- custom 25 200 100 output/my_instance.json");
        return;
    }
    
    let nurses = args[2].parse::<u32>().unwrap_or(25);
    let capacity = args[3].parse::<u32>().unwrap_or(200);
    let patients = args[4].parse::<usize>().unwrap_or(100);
    let output = &args[5];
    
    println!("Generating custom instance...");
    println!("  Nurses: {}", nurses);
    println!("  Capacity: {}", capacity);
    println!("  Patients: {}", patients);
    
    let config = InstanceConfig {
        instance_name: format!("custom_{}n_{}c_{}p", nurses, capacity, patients),
        nbr_nurses_range: (nurses, nurses),
        capacity_nurse_range: (capacity, capacity),
        nbr_patients_range: (patients, patients),
        ..Default::default()
    };
    
    match generate_and_save_instance(
        config,
        output,
        true,
        Some(&format!("{}.png", output.trim_end_matches(".json"))),
    ) {
        Ok(_) => println!("✓ Successfully generated and saved to: {}", output),
        Err(e) => eprintln!("✗ Error: {}", e),
    }
}

fn generate_batch_instances(args: &[String]) {
    let count = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(5)
    } else {
        5
    };
    
    let output_dir = if args.len() > 3 {
        &args[3]
    } else {
        "output/random_instances/batch"
    };
    
    println!("Generating {} instances...", count);
    std::fs::create_dir_all(output_dir).ok();
    
    for i in 1..=count {
        let config = InstanceConfig {
            instance_name: format!("batch_instance_{}", i),
            ..Default::default()
        };
        
        let output = format!("{}/instance_{}.json", output_dir, i);
        
        match generate_and_save_instance(config, &output, true, None) {
            Ok(instance) => {
                println!("  [{}/{}] Generated: {} nurses, {} patients", 
                         i, count, instance.nbr_nurses, instance.patients.len());
            }
            Err(e) => eprintln!("  [{}/{}] Error: {}", i, count, e),
        }
    }
    
    println!("✓ Batch generation complete! Check {}/", output_dir);
}

fn print_usage() {
    println!("Instance Generator CLI");
    println!("=====================");
    println!();
    println!("Usage:");
    println!("  cargo run -- <command> [args...]");
    println!();
    println!("Commands:");
    println!("  default [output]");
    println!("      Generate instance with default parameters");
    println!("      Example: cargo run -- default output/my_instance.json");
    println!();
    println!("  train <variation> [output]");
    println!("      Generate train-like variant (variation: 0.0-1.0)");
    println!("      Example: cargo run -- train 0.2 output/variant.json");
    println!();
    println!("  custom <nurses> <capacity> <patients> <output>");
    println!("      Generate instance with specific parameters");
    println!("      Example: cargo run -- custom 25 200 100 output/custom.json");
    println!();
    println!("  batch <count> [output_dir]");
    println!("      Generate multiple instances");
    println!("      Example: cargo run -- batch 10 output/batch");
    println!();
    println!("  help");
    println!("      Show this help message");
    println!();
}
