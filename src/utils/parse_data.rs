use crate::structs::instance::Instance;
use std::fs::File;
use std::io::BufReader;

/// Parses a JSON file containing all necessary information for the nurse problem instance into an `Instance`.
/// 
/// This function opens the file at the specified path and wraps it in a buffered reader. It then deserializes
/// the JSON content into an `Instance` struct using Serde. The JSON file is expected to contain all the required
/// data for the nurse problem instance, including depot details, patient information, and other parameters.
/// After deserialization, the function initializes the nurse vector by invoking the `add_nurses` method on the instance.
/// 
/// # Parameters
/// - `path`: A string slice representing the path to the JSON file with the nurse problem instance data.
/// 
/// # Returns
/// An `Instance` populated with the data parsed from the JSON file.
pub fn parse_data(path: &str) -> Instance {
    let file = File::open(path).expect("File not found");
    let reader = BufReader::new(file);
    let mut instance: Instance = serde_json::from_reader(reader).expect("Error while reading file");
    instance.add_nurses();
    instance
}
