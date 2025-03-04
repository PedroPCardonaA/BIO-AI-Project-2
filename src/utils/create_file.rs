use serde_json;
use std::fs::File;
use std::io::Write;
use serde::Serialize;

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
/// # Arguments
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
