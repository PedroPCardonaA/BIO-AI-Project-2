use serde_json;
use std::fs::File;
use std::io::Write;

pub fn save_solution_to_file(instance: &Vec<Vec<usize>>, file_path: &str) -> std::io::Result<()> {
    let mut file = File::create(file_path)?;
    let instance_json = serde_json::to_string_pretty(instance).unwrap();
    file.write_all(instance_json.as_bytes())?;
    Ok(())
}