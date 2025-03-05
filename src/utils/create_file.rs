use serde_json;
use std::fs::{self, File};
use std::io::Write;
use serde::Serialize;

use crate::structs::instance::Instance;

use super::plot_map::plot_map_with_path;
use super::textual_answer::save_textual_solution_to_file;

/// Saves the given solution to a file in pretty-printed JSON format.
/// 
/// # Parameters
/// - `instance`: A reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
/// - `file_path`: The file path where the JSON representation of the solution will be saved.
/// 
/// # Returns
/// A `std::io::Result<()>` indicating whether the file was written successfully.
pub fn save_solution_to_file(instance: &Vec<Vec<usize>>, file_path: &str) -> std::io::Result<()> {
    let mut file = File::create(file_path)?;
    let instance_json = serde_json::to_string_pretty(instance).unwrap();
    file.write_all(instance_json.as_bytes())?;
    Ok(())
}

/// Saves any serializable data to a file in pretty-printed JSON format.
///
/// # Parameters
///
/// * `data` - A reference to the data to be saved. It must implement the `Serialize` trait.
/// * `file_path` - The path where the JSON representation of the data will be written.
///
/// # Returns
///
/// A `std::io::Result<()>` indicating whether the file was written successfully.
///
pub fn save_json<T: Serialize>(data: &T, file_path: &str) -> std::io::Result<()> {
    let mut file = File::create(file_path)?;
    let json = serde_json::to_string_pretty(data).unwrap();
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Cleans up a current_best folder by removing all solution files (and their associated TXT/PNG files)
/// whose stored objective value (extracted from the file name) exceeds benchmark * 1.10.
///
/// # Parameters
/// - `current_best_dir`: The path to the current_best folder.
/// - `benchmark`: The benchmark objective value for the instance.
///
/// # Remarks
/// Files are expected to be named in the format "best_solution_<score>.json", where `<score>` is an integer.
/// The stored objective value is interpreted as `<score> / 100.0`.
pub fn cleanup_current_best_folder(current_best_dir: &str, benchmark: f64) {
    if let Ok(entries) = fs::read_dir(current_best_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(score_str) = file_name.strip_prefix("best_solution_")
                    .and_then(|s| s.strip_suffix(".json"))
                {
                    if let Ok(parsed_score) = score_str.parse::<u32>() {
                        let stored_obj_value = parsed_score as f64 / 100.0;
                        if stored_obj_value > benchmark * 1.10 {
                            if let Err(e) = fs::remove_file(&path) {
                                eprintln!("Error removing file {:?}: {}", path, e);
                            }
                            // Remove associated TXT and PNG files.
                            let base_file = file_name.strip_suffix(".json").unwrap();
                            let txt_file = format!("{}/{}.txt", current_best_dir, base_file);
                            let png_file = format!("{}/{}.png", current_best_dir, base_file);
                            let _ = fs::remove_file(txt_file);
                            let _ = fs::remove_file(png_file);
                        }
                    }
                }
            }
        }
    }
}

/// Saves the current run's solution files (JSON, TXT, and PNG) to the specified output directory.
///
/// # Parameters
/// - `output_dir`: The directory in which to save the current solution.
/// - `current_solution`: A reference to the current solution (a vector of routes).
/// - `instance`: A reference to the parsed instance.
///
/// # Remarks
/// The solution is saved to "solution.json", "solution.txt", and "solution.png", overwriting any existing files.
pub fn save_current_solution_files(output_dir: &str, current_solution: &Vec<Vec<usize>>, instance: &Instance) {
    let sol_file_json = format!("{}/solution.json", output_dir);
    let sol_file_txt = format!("{}/solution.txt", output_dir);
    let sol_file_png = format!("{}/solution.png", output_dir);
    match save_solution_to_file(&current_solution, &sol_file_json) {
        Ok(_) => println!("Solution updated in {}", sol_file_json),
        Err(e) => eprintln!("Error saving solution in {}: {}", sol_file_json, e),
    }
    save_textual_solution_to_file(&sol_file_txt, &current_solution, instance);
    plot_map_with_path(&current_solution, &instance.patients, &instance.depot, &sol_file_png);
}

/// Saves the new best solution files (JSON, TXT, and PNG) to the specified current_best folder.
///
/// # Parameters
/// - `current_best_dir`: The directory in which to save the best solution.
/// - `obj_value`: The objective value of the new best solution.
/// - `current_solution`: A reference to the new best solution.
/// - `instance`: A reference to the parsed instance.
///
/// # Remarks
/// A unique file name is generated in the format "best_solution_<score>.json" (and corresponding TXT/PNG files),
/// where `<score>` is computed as `(obj_value * 100.0).round()` as an integer.
pub fn save_best_solution_files(
    current_best_dir: &str,
    obj_value: f64,
    current_solution: &Vec<Vec<usize>>,
    instance: &Instance,
) {
    let file_suffix = (obj_value * 100.0).round() as u32;
    let best_sol_file_json = format!("{}/best_solution_{}.json", current_best_dir, file_suffix);
    let best_sol_file_txt = format!("{}/best_solution_{}.txt", current_best_dir, file_suffix);
    let best_sol_file_png = format!("{}/best_solution_{}.png", current_best_dir, file_suffix);
    match save_solution_to_file(&current_solution, &best_sol_file_json) {
        Ok(_) => println!("Current best solution updated in {}", best_sol_file_json),
        Err(e) => eprintln!("Error saving current best solution in {}: {}", best_sol_file_json, e),
    }
    save_textual_solution_to_file(&best_sol_file_txt, &current_solution, instance);
    plot_map_with_path(&current_solution, &instance.patients, &instance.depot, &best_sol_file_png);
}
